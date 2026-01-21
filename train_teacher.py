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


def get_pooling_optimizer(teacher_pool_v_qry,
    teacher_pool_t_qry,
    teacher_pool_v_pos,
    teacher_pool_t_pos, training_args):
    params = list(teacher_pool_v_qry.parameters()) + list(teacher_pool_t_qry.parameters()) + list(teacher_pool_v_pos.parameters()) + list(teacher_pool_t_pos.parameters())

    return AdamW(
        params,
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

def train_teacher_stage1(
    teacher_model,
    teacher_pool_v_qry,
    teacher_pool_t_qry,
    teacher_pool_v_pos,
    teacher_pool_t_pos,
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

    teacher_model.eval()        # 🔒 teacher frozen
    teacher_pool_v_qry.train()
    teacher_pool_t_qry.train()
    teacher_pool_v_pos.train()
    teacher_pool_t_pos.train()


    for epoch in range(training_args.num_train_epochs):
        sampler.set_epoch(epoch)
        print_rank(f"[Teacher Stage-1 | Frozen] Epoch {epoch + 1}")

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # ----------------------------------
            # Teacher forward (NO GRAD)
            # ----------------------------------
            
            teacher_qry_input = batch['teacher_inputs']['qry']
            teacher_pos_input = batch['teacher_inputs']['pos']
            num_text_qry_tokens = ((teacher_qry_input['input_ids'] < 151643) | (teacher_qry_input['input_ids'] > 151656)).sum(dim=1)
            num_text_pos_tokens = ((teacher_pos_input['input_ids'] < 151643) | (teacher_pos_input['input_ids'] > 151656)).sum(dim=1)
            
            batch_size = training_args.per_device_train_batch_size
            with torch.no_grad():
                teacher_qry_output = teacher_model.encode_input(teacher_qry_input)
                teacher_pos_output = teacher_model.encode_input(teacher_pos_input)
                teacher_qry_reps, teacher_qry_image_features, teacher_qry_attention, teacher_qry_hidden_states = teacher_qry_output
                teacher_pos_reps, teacher_pos_image_features, teacher_pos_attention, teacher_pos_hidden_states = teacher_pos_output
            cur_idx_qry_img = 0
            cur_idx_pos_img = 0
            H_T_v_qry = []
            H_T_t_qry = []
            
            H_T_v_pos = []
            H_T_t_pos = []
            
            for i in range(batch_size):
                # print(f"Sample {i}: num_text_qry_tokens {num_text_qry_tokens[i]}, num_text_pos_tokens {num_text_pos_tokens[i]}")
                # print(f"Sample {i} input_ids ids of teacher {teacher_qry_input['input_ids'][i]}, pos {teacher_pos_input['input_ids'][i]}")
                # print(f"Sample {i} input_ids ids of student {student_qry_input['input_ids'][i]}, pos {student_pos_input['input_ids'][i]}")
                if teacher_qry_image_features is not None:
                    if cur_idx_qry_img < len(teacher_qry_image_features):
                        if teacher_qry_image_features[cur_idx_qry_img] is not None:
                            
                            num_tokens_vision_qry_tea = teacher_qry_image_features[cur_idx_qry_img].size(0)
                            # print(f"Sample qry {i}: num_tokens_vision_qry_stu {num_tokens_vision_qry_stu}, num_tokens_vision_qry_tea {num_tokens_vision_qry_tea}")
                            num_text_token_qry_tea = num_text_qry_tokens[i]
                            
                            teacher_qry_vision_hidden_state = teacher_qry_hidden_states[-1][i][-(num_tokens_vision_qry_tea + num_text_token_qry_tea):-(num_text_token_qry_tea), :]
                        
                            teacher_qry_text_hidden_state = teacher_qry_hidden_states[-1][i][-num_text_token_qry_tea:, :]

                            H_T_v_qry.append(teacher_qry_vision_hidden_state)
                            H_T_t_qry.append(teacher_qry_text_hidden_state)
                if teacher_pos_image_features is not None:
                    if cur_idx_pos_img < len(teacher_pos_image_features):
                        if teacher_pos_image_features[cur_idx_pos_img] is not None:
                            num_tokens_vision_pos_tea = teacher_pos_image_features[cur_idx_pos_img].size(0)
                            # print(f"Sample pos {i}: num_tokens_vision_pos_stu {num_tokens_vision_pos_stu}, num_tokens_vision_pos_tea {num_tokens_vision_pos_tea}")
                            num_text_token_pos_tea = num_text_pos_tokens[i]
                            teacher_pos_vision_hidden_state = teacher_pos_hidden_states[-1][i][-(num_tokens_vision_pos_tea + num_text_token_pos_tea):-(num_text_token_pos_tea), :]
                            
                            teacher_pos_text_hidden_state = teacher_pos_hidden_states[-1][i][-num_text_token_pos_tea:, :]

                           
                            H_T_v_pos.append(teacher_pos_vision_hidden_state)
                            
                            H_T_t_pos.append(teacher_pos_text_hidden_state) 

                H_T_v_qry = torch.cat(H_T_v_qry, dim=0)
                H_T_t_qry = torch.cat(H_T_t_qry, dim=0)
                
                H_T_v_pos = torch.cat(H_T_v_pos, dim=0)
                H_T_t_pos = torch.cat(H_T_t_pos, dim=0)
                
            with torch.no_grad():
                
                z_T_v_qry, g_T_v_qry = teacher_pool_v_qry(H_T_v_qry)
                z_T_t_qry, g_T_t_qry = teacher_pool_t_qry(H_T_t_qry)

                z_T_v_pos, g_T_v_pos = teacher_pool_v_pos(H_T_v_pos)
                z_T_t_pos, g_T_t_pos = teacher_pool_t_pos(H_T_t_pos)

            


            # ----------------------------------
            # Contrastive loss
            # ----------------------------------
            loss_con_qry = contrastive_loss(
                z_T_v_qry,
                z_T_t_qry,
                temperature=training_args.temperature,
            )

            loss_con_pos = contrastive_loss(
                z_T_v_pos,
                z_T_t_pos,
                temperature=training_args.temperature,
            )
            loss_con = 0.5 * (loss_con_qry + loss_con_pos)
            # ----------------------------------
            # Final loss (pooling only)
            # ----------------------------------
            loss_task = 0
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
    teacher_pool_v_qry = ModalityGatedPooling(hidden_dim).to(device)
    teacher_pool_t_qry = ModalityGatedPooling(hidden_dim).to(device)
    teacher_pool_v_pos = ModalityGatedPooling(hidden_dim).to(device)
    teacher_pool_t_pos = ModalityGatedPooling(hidden_dim).to(device)

    optimizer = get_pooling_optimizer(
        teacher_pool_v_qry,
        teacher_pool_t_qry,
        teacher_pool_v_pos,
        teacher_pool_t_pos, training_args
        )

    train_teacher_stage1(
        teacher,
        teacher_pool_v_qry,
        teacher_pool_t_qry,
        teacher_pool_v_pos,
        teacher_pool_t_pos,
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
                "vision_pool_qry": teacher_pool_v_qry.state_dict(),
                "text_pool_qry": teacher_pool_t_qry.state_dict(),
                "vision_pool_pos": teacher_pool_v_pos.state_dict(),
                "text_pool_pos": teacher_pool_t_pos.state_dict(),
                "hidden_dim": hidden_dim,
            },
            os.path.join(training_args.output_dir, "teacher_stage1.pt"),
        )

        print_rank("Saved teacher_stage1.pt")

if __name__ == "__main__":
    main()
