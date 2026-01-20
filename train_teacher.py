import json
import time
import os
import math
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW

import deepspeed
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoProcessor,
    HfArgumentParser,
)

from src.arguments import DataArguments, TrainingArguments, ModelArguments
from src.distiller import DistillationDataset, DistillationCollator
from src.model.modality_gated_pooling import ModalityGatedPooling
from src.utils import print_rank
from src import model

def contrastive_loss(z_v, z_t, temperature=0.07):
    """
    z_v, z_t: [B, D]
    """
    z_v = nn.functional.normalize(z_v, dim=-1)
    z_t = nn.functional.normalize(z_t, dim=-1)

    logits = z_v @ z_t.T / temperature
    labels = torch.arange(z_v.size(0), device=z_v.device)

    loss_v2t = nn.functional.cross_entropy(logits, labels)
    loss_t2v = nn.functional.cross_entropy(logits.T, labels)

    return 0.5 * (loss_v2t + loss_t2v)


def get_pooling_optimizer(pool_v, pool_t, training_args):
    params = list(pool_v.parameters()) + list(pool_t.parameters())

    return AdamW(
        params,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

def train_teacher_stage1(
    teacher,
    pool_v,
    pool_t,
    train_dataset,
    collator,
    optimizer,
    training_args,
    device,
):
    sampler = DistributedSampler(
        train_dataset,
        shuffle=True,
        drop_last=True,
        rank=dist.get_rank(),
        num_replicas=dist.get_world_size(),
    )

    dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
    )

    teacher.eval()        # 🔒 teacher frozen
    pool_v.train()
    pool_t.train()

    for epoch in range(training_args.num_train_epochs):
        sampler.set_epoch(epoch)
        print_rank(f"[Teacher Stage-1 | Frozen] Epoch {epoch + 1}")

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # ----------------------------------
            # Teacher forward (NO GRAD)
            # ----------------------------------
            with torch.no_grad():
                outputs = teacher(**batch)
                loss_task = outputs["loss"]  # supervision signal only

                H_T_v = outputs["vision_hidden_states"]  # [B, Nv, D]
                H_T_t = outputs["text_hidden_states"]    # [B, Nt, D]

            # ----------------------------------
            # Learnable gated pooling
            # ----------------------------------
            z_T_v, g_v = pool_v(H_T_v)
            z_T_t, g_t = pool_t(H_T_t)

            # ----------------------------------
            # Contrastive loss
            # ----------------------------------
            loss_con = contrastive_loss(
                z_T_v,
                z_T_t,
                temperature=training_args.temperature,
            )

            # ----------------------------------
            # Final loss (pooling only)
            # ----------------------------------
            loss = loss_task + training_args.lambda_contrastive * loss_con

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print_rank(
            f"[Epoch {epoch + 1}] "
            f"task={loss_task.item():.4f} | "
            f"contrastive={loss_con.item():.4f}"
        )
def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    deepspeed.init_distributed(timeout=timedelta(minutes=1))
    device = torch.cuda.current_device()

    # Dataset
    train_dataset = DistillationDataset(data_args, model_args)

    # Collator (teacher only)
    collator = DistillationCollator(
        student_processor=None,
        teacher_processor=model.build_processor(model_args),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )

    # -----------------------------
    # Teacher model (FROZEN)
    # -----------------------------
    teacher = model.build_teacher_model(model_args).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    # -----------------------------
    # Learnable pooling heads
    # -----------------------------
    hidden_dim = teacher.config.hidden_size
    pool_v = ModalityGatedPooling(hidden_dim).to(device)
    pool_t = ModalityGatedPooling(hidden_dim).to(device)

    optimizer = get_pooling_optimizer(
        pool_v, pool_t, training_args
    )

    train_teacher_stage1(
        teacher,
        pool_v,
        pool_t,
        train_dataset,
        collator,
        optimizer,
        training_args,
        device,
    )

    # -----------------------------
    # Save checkpoint
    # -----------------------------
    if dist.get_rank() == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        torch.save(
            {
                "pool_v": pool_v.state_dict(),
                "pool_t": pool_t.state_dict(),
                "hidden_dim": hidden_dim,
            },
            os.path.join(training_args.output_dir, "teacher_stage1.pt"),
        )

        print_rank("Saved teacher_stage1.pt")

if __name__ == "__main__":
    main()
