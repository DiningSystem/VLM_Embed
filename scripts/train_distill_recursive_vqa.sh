#!/bin/bash

NUM_GPUS_PER_NODE=1
TRAIN_SCRIPT="train_distill_ddp.py"

torchrun --standalone \
    --nproc_per_node=$NUM_GPUS_PER_NODE $TRAIN_SCRIPT \
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
    --subset_name "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 1.0 \
    --output_dir "training/recursive_distill_vqa" \
    --per_device_train_batch_size 10 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --num_train_epochs 2 \
    --bf16 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --save_strategy "epoch" \
    --seed 42 \
    --weight_decay 0.01 \
    --normalize True \
    --teacher_normalize True \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --kd_weight 0.3 \
    --kd_loss_type "recursive_distillation_loss" \
    --recursive_num_steps 6 \
    --recursive_backprop_steps 1 \
    --recursive_mean_weight 1.0 \
    --recursive_cov_weight 0.1 \
    --recursive_contrastive_weight 1.0 \
    --recursive_attn_weight 1.0 \
    --recursive_enable_kv_cache True \
    --recursive_kv_cache_size 32 \
    --image_resolution "mid" \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-5
