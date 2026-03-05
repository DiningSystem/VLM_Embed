#!/bin/bash

# Number of GPUs on this node
NUM_GPUS_PER_NODE=1

# Training entrypoint
TRAIN_SCRIPT="train_distill_ddp.py"

# Choose KD alignment space for EOS attention KL:
# - student: project teacher pooled reps to student space
# - teacher: project student pooled reps to teacher space
EOS_PROJECTION_SPACE="student"
TEMPERATURE=0.02

SUBSETS=(
  "ImageNet_1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
)

torchrun --nproc_per_node=$NUM_GPUS_PER_NODE \
    $TRAIN_SCRIPT \
    --model_name "apple/FastVLM-0.5B" \
    --teacher_model_name "raghavlite/B3_Qwen2_2B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/cls_05eos_attention_kl_intra_cosine_${EOS_PROJECTION_SPACE}" \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "constant" \
    --warmup_ratio 0.03 \
    --temperature "${TEMPERATURE}" \
    --kd_weight 0.5 \
    --kd_loss_type "eos_attention_kl_intra_cosine_loss" \
    --eos_projection_space "${EOS_PROJECTION_SPACE}" \
    --image_resolution "low" \
    --ddp_find_unused_parameters True \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-5 \
    --report_to None
