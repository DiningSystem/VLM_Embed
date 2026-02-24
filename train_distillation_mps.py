"""
Train distillation với MPS 2-epoch:
  Epoch 1: Chỉ train RedundancyEstimator (mps_recon_loss).
  Epoch 2: Đóng băng RedundancyEstimator, tắt mps_recon_loss; train synergy/magnitude/orthogonality + contrastive.

Chạy: bash scripts/train_distill_mps.sh
Hoặc: deepspeed --num_gpus N train_distillation_mps.py ... --kd_loss_type mps_loss --num_train_epochs 2 ...
"""

import json
from src.distiller import Distiller, DistillationCollator, DistillationDataset
from src.arguments import DataArguments, MTEBArguments, TrainingArguments, ModelArguments
from src.utils import print_rank, print_master
from src.criterions import build_criterion
import time
import os
import sys
from tqdm import tqdm
import math
from datetime import timedelta

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW

import deepspeed
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, HfArgumentParser
from deepspeed.runtime.zero import GatheredParameters

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def get_optimizer_params(model, training_args):
    while hasattr(model, "module"):
        model = model.module
    target_model = model.encoder if hasattr(model, "encoder") else model
    trainable_params = [p for _, p in target_model.named_parameters() if p.requires_grad]
    print_master(f"Total trainable params: {sum(p.numel() for p in trainable_params)}")
    return [{"params": trainable_params}]


def get_optimizer(model, training_args):
    optimizer_grouped_parameters = get_optimizer_params(model, training_args)
    return AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training_args.weight_decay,
    )


def prepare_dataset(data_args, model_args):
    return DistillationDataset(data_args, model_args)


def to_device(obj, device):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_device(v, device) for v in obj)
    if hasattr(obj, "to") and callable(obj.to):
        return obj.to(device)
    return obj


def finetune_mps(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    distiller: Distiller,
    train_dataset: DistillationDataset,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    collator: DistillationCollator,
    criterion: nn.Module,
    device=None,
):
    """Finetune với MPS 2-epoch: đầu mỗi epoch gọi set_epoch và set_mps_phase."""
    print_rank("Start MPS 2-epoch finetuning...")
    start_time = time.time()

    is_distributed = dist.is_initialized()
    dp_world_size = dist.get_world_size() if is_distributed else 1
    dp_rank = dist.get_rank() if is_distributed else 0

    sampler = DistributedSampler(
        train_dataset, shuffle=True, drop_last=True, rank=dp_rank, num_replicas=dp_world_size
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=collator,
        sampler=sampler,
    )

    ds_config = {}
    ds_config_path = getattr(training_args, "deepspeed_config", None)
    if ds_config_path:
        if isinstance(ds_config_path, dict):
            ds_config = ds_config_path
        elif isinstance(ds_config_path, str) and os.path.exists(ds_config_path):
            with open(ds_config_path, "r") as f:
                ds_config = json.load(f)
        else:
            print_rank(f"Warning: deepspeed config path {ds_config_path} not found.")
            ds_config = {}

    ds_config["gradient_accumulation_steps"] = training_args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = training_args.per_device_train_batch_size
    ds_config["gradient_clipping"] = training_args.max_grad_norm
    ds_config["train_batch_size"] = (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * (dist.get_world_size() if is_distributed else 1)
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=distiller,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        mpu=None,
        config_params=ds_config,
    )
    print_rank(f"model_engine.module is distiller: {model_engine.module is distiller}")
    total_trainable = sum(p.numel() for p in model_engine.parameters() if p.requires_grad)
    print_rank(f"Total trainable parameters: {total_trainable}")
    for n, p in model_engine.named_parameters():
        if p.requires_grad:
            try:
                p.data = p.data.to(dtype=torch.bfloat16)
            except Exception as e:
                print_rank(f"Warning: cannot cast {n} to bfloat16: {e}")
    print_rank(f"model device: {next(model_engine.parameters()).device}")
    model_engine.train()

    use_wandb = False
    if _HAS_WANDB and dist.get_rank() == 0:
        report_to = getattr(training_args, "report_to", None)
        if report_to and ("wandb" in (report_to if isinstance(report_to, (list, tuple)) else [report_to])):
            use_wandb = True
            wandb.init(
                project=getattr(training_args, "wandb_project", "vlm_embed_mps"),
                name=getattr(training_args, "run_name", None) or training_args.output_dir,
                config={
                    "learning_rate": training_args.learning_rate,
                    "per_device_train_batch_size": training_args.per_device_train_batch_size,
                    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                    "num_train_epochs": training_args.num_train_epochs,
                    "weight_decay": training_args.weight_decay,
                    "lr_scheduler_type": training_args.lr_scheduler_type,
                    "warmup_ratio": training_args.warmup_ratio,
                    "kd_loss_type": getattr(training_args, "kd_loss_type", ""),
                    "output_dir": training_args.output_dir,
                },
            )
            if model_args.model_name:
                wandb.config.update({"model_name": model_args.model_name})
            if getattr(model_args, "teacher_model_name", None):
                wandb.config.update({"teacher_model_name": model_args.teacher_model_name})

    logging_output = {
        "epoch": 0,
        "global_step": 0,
        "loss": [],
        "contrastive_loss": [],
        "kd_loss": [],
    }
    step = 0

    for epoch in range(training_args.num_train_epochs):
        logging_output["epoch"] = epoch + 1
        print_rank(f"Start iteration of epoch {epoch + 1} (MPS phase {epoch})")

        # MPS 2-epoch: epoch 0 = chỉ RedundancyEstimator, epoch 1 = đóng băng nó, train còn lại
        if hasattr(criterion, "set_epoch"):
            criterion.set_epoch(epoch)
        if hasattr(model_engine.module, "set_mps_phase"):
            model_engine.module.set_mps_phase(epoch)

        end_epoch = False
        epoch_step = 0
        epoch_loss, epoch_contrastive_loss, epoch_kd_loss = 0.0, 0.0, 0.0
        losses, contrastive_losses, kd_losses = [], [], []
        model_engine.train()

        if is_distributed and isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        train_iter = iter(train_dataloader)
        grad_accum = int(ds_config.get("gradient_accumulation_steps", 1))
        total_steps = math.ceil(len(train_dataloader) / grad_accum)
        print_rank(f"[INFO] Batches per epoch: {len(train_dataloader)}, GradAccum: {grad_accum}, Steps: {total_steps}")

        progress_bar = tqdm(
            total=total_steps,
            desc=f"Epoch {epoch+1}",
            disable=(getattr(model_engine, "global_rank", 0) != 0),
        )

        while True:
            global_batch = []
            for _ in range(grad_accum):
                try:
                    batch = next(train_iter)
                    batch = to_device(batch, device)
                    global_batch.append(batch)
                except StopIteration:
                    end_epoch = True
                    break
            if end_epoch:
                break

            for batch in global_batch:
                loss_dict = model_engine(criterion, batch)
                loss = loss_dict["loss"]
                model_engine.backward(loss)

                contrastive_loss = loss_dict.get("contrastive_loss", torch.tensor(0.0, device=loss.device))
                kd_loss = loss_dict.get("mps_total_kd_loss", loss_dict.get("kd_loss", torch.tensor(0.0, device=loss.device)))
                recon_loss = loss_dict.get("mps_recon_loss", torch.tensor(0.0, device=loss.device))

                losses.append(loss.detach().item() * training_args.gradient_accumulation_steps)
                contrastive_losses.append(contrastive_loss.detach().item())
                kd_losses.append(kd_loss.detach().item())

                model_engine.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                step += 1

            if dist.get_rank() == 0 and step % max(1, training_args.logging_steps) == 0:
                try:
                    current_lr = optimizer.param_groups[0]["lr"]
                except Exception:
                    current_lr = None
                n = len(losses)
                batch_loss = sum(losses) / n if n else 0.0
                batch_contrastive = sum(contrastive_losses) / n if n else 0.0
                batch_kd = sum(kd_losses) / n if n else 0.0
                epoch_loss += sum(losses)
                epoch_contrastive_loss += sum(contrastive_losses)
                epoch_kd_loss += sum(kd_losses)
                progress_bar.set_postfix({
                    "loss": f"{batch_loss:.4f}",
                    "contrastive": f"{batch_contrastive:.4f}",
                    "kd": f"{batch_kd:.4f}",
                    "recon": f"{recon_loss.item():.4f}" if torch.is_tensor(recon_loss) else "0",
                    "lr": f"{current_lr:.6f}" if current_lr is not None else "N/A",
                })
                progress_bar.update(1)
                if use_wandb:
                    log_dict = {
                        "train/loss": batch_loss,
                        "train/contrastive_loss": batch_contrastive,
                        "train/kd_loss": batch_kd,
                        "train/global_step": step,
                        "train/epoch": epoch + 1,
                    }
                    if torch.is_tensor(recon_loss):
                        log_dict["train/mps_recon_loss"] = recon_loss.item()
                    if current_lr is not None:
                        log_dict["train/lr"] = current_lr
                    wandb.log(log_dict)
            epoch_step += 1

        if dist.get_rank() == 0:
            avg_epoch_loss = epoch_loss / max(1, epoch_step)
            avg_contrastive = epoch_contrastive_loss / max(1, epoch_step)
            avg_kd = epoch_kd_loss / max(1, epoch_step)
            print_rank(
                f"Epoch {epoch + 1} done. Avg Loss: {avg_epoch_loss:.4f} | "
                f"Contrastive: {avg_contrastive:.4f} | KD: {avg_kd:.4f}"
            )
            if use_wandb:
                wandb.log({
                    "epoch/avg_loss": avg_epoch_loss,
                    "epoch/avg_contrastive_loss": avg_contrastive,
                    "epoch/avg_kd_loss": avg_kd,
                    "epoch/epoch": epoch + 1,
                })
            if training_args.save_strategy == "epoch":
                ckpt_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch{epoch + 1}")
                os.makedirs(ckpt_dir, exist_ok=True)
                with GatheredParameters(model_engine.module.student.parameters(), modifier_rank=0):
                    model_engine.module.student.encoder.save_pretrained(ckpt_dir)
                try:
                    config = AutoConfig.from_pretrained(model_args.model_name) if model_args.model_name else None
                    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name) if model_args.model_name else None
                    if config is not None:
                        config.save_pretrained(ckpt_dir)
                    if tokenizer is not None:
                        tokenizer.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Warning saving config/tokenizer: {e}")
                try:
                    processor = AutoProcessor.from_pretrained(model_args.model_name) if model_args.model_name else None
                    if processor is not None:
                        processor.save_pretrained(ckpt_dir)
                except Exception as e:
                    print_rank(f"Warning saving processor: {e}")
        dist.barrier()
        print_rank(f"Epoch {epoch + 1} finished.")

    total_time = time.time() - start_time
    print_rank(f"Training completed in {total_time/3600:.2f} hours")

    if dist.get_rank() == 0 and training_args.save_strategy == "epoch":
        final_ckpt_dir = os.path.join(training_args.output_dir, "checkpoint-final")
        os.makedirs(final_ckpt_dir, exist_ok=True)
        with GatheredParameters(model_engine.module.student.parameters(), modifier_rank=0):
            model_engine.module.student.encoder.save_pretrained(final_ckpt_dir)
        print_rank(f"Final model saved at {final_ckpt_dir}")
        if model_args.model_name:
            try:
                AutoConfig.from_pretrained(model_args.model_name).save_pretrained(final_ckpt_dir)
                AutoTokenizer.from_pretrained(model_args.model_name).save_pretrained(final_ckpt_dir)
                AutoProcessor.from_pretrained(model_args.model_name).save_pretrained(final_ckpt_dir)
            except Exception as e:
                print_rank(f"Warning saving final config/tokenizer/processor: {e}")
    if use_wandb:
        try:
            wandb.finish()
        except Exception as e:
            print_rank(f"Warning: wandb.finish() failed: {e}")
    dist.barrier()
    return logging_output


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    assert getattr(training_args, "kd_loss_type", None) == "mps_loss", (
        "train_distillation_mps.py requires --kd_loss_type mps_loss"
    )
    if training_args.num_train_epochs != 2:
        print_rank(f"Warning: MPS 2-epoch expects num_train_epochs=2, got {training_args.num_train_epochs}")

    torch.backends.cudnn.enabled = False
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    deepspeed.init_distributed(timeout=timedelta(minutes=1))

    train_dataset = prepare_dataset(data_args, model_args)
    print_rank(f"Number of training samples: {len(train_dataset)}")

    distiller = Distiller(model_args, training_args)
    print_rank(f"Student params: {sum(p.numel() for p in distiller.student.parameters())}")
    print_rank(f"Teacher params: {sum(p.numel() for p in distiller.teacher.parameters())}")

    collator = DistillationCollator(
        student_processor=distiller.get_student_processor(),
        teacher_processor=distiller.get_teacher_processor(),
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
    optimizer = get_optimizer(distiller.student, training_args)
    if model_args.projector_config_path is not None:
        optimizer = distiller.add_optimizer_param_group(optimizer)

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    batch_per_step = max(
        1,
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * world_size,
    )
    steps_per_epoch = max(1, len(train_dataset) // batch_per_step)
    total_steps = steps_per_epoch * training_args.num_train_epochs

    if training_args.lr_scheduler_type == "linear":
        from transformers import get_linear_schedule_with_warmup
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(training_args.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
    elif training_args.lr_scheduler_type == "cosine":
        from transformers import get_cosine_schedule_with_warmup
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(training_args.warmup_ratio * total_steps),
            num_training_steps=total_steps,
        )
    else:
        from transformers import get_constant_schedule
        lr_scheduler = get_constant_schedule(optimizer)

    criterion = build_criterion(training_args)

    logging_output = finetune_mps(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        distiller=distiller,
        train_dataset=train_dataset,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        collator=collator,
        criterion=criterion,
        device=device,
    )
    print_rank("MPS training completed successfully!")
    return logging_output


if __name__ == "__main__":
    main()
