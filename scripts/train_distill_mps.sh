#!/bin/bash
# MPS 2-epoch training:
#   Epoch 1: Chỉ train RedundancyEstimator (mps_recon_loss).
#   Epoch 2: Đóng băng RedundancyEstimator, tắt mps_recon_loss; train synergy/magnitude/orthogonality + contrastive.

export WANDB_API_KEY="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"

NUM_GPUS=1
TRAIN_SCRIPT="train_distillation_mps.py"
DS_CONFIG="config/ds_config_stage2.json"

deepspeed --num_gpus=$NUM_GPUS $TRAIN_SCRIPT \
    --model_name apple/FastVLM-1.5B \
    --teacher_model_name "raghavlite/B3_Qwen2_7B" \
    --lora True \
    --teacher_lora True \
    --lora_r 64 \
    --lora_target_modules "qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj" \
    --student_hidden_dim 1536 \
    --teacher_hidden_dim 3584 \
    --teacher_lora_r 8 \
    --teacher_pooling "eos" \
    --teacher_backbone "qwen2_vl" \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "MSCOCO" \
    --dataset_split "original" \
    --image_dir "vlm2vec_train/MMEB-train" \
    --percent_data 0.3 \
    --output_dir "training/mps_2epoch" \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --deepspeed_config $DS_CONFIG \
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
    --warmup_ratio 0.04 \
    --kd_weight 0.3 \
    --kd_loss_type "mps_loss" \
    --mps_recon_weight 0.5 \
    --mps_recon_loss_type "mse" \
    --mps_synergy_weight 1.0 \
    --mps_magnitude_weight 0.1 \
    --mps_orthogonality_weight 0.5 \
    --mps_freeze_teacher_estimator True \
    --image_resolution "low" \
    --projector_config_path "./config/projector_config.json" \
    --projector_lr 5e-5 \
    --report_to wandb


# #!/bin/bash

# TRAIN_SCRIPT="train_distillation_mps.py"

# python $TRAIN_SCRIPT \
#     --model_name apple/FastVLM-1.5B \
#     --teacher_model_name "raghavlite/B3_Qwen2_7B" \
#     --lora True \
#     --teacher_lora True \
#     --lora_r 64 \
#     --lora_target_modules "qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj" \
#     --student_hidden_dim 1536 \
#     --teacher_hidden_dim 3584 \
#     --teacher_lora_r 8 \
#     --teacher_pooling "eos" \
#     --teacher_backbone "qwen2_vl" \
#     --model_backbone "llava_qwen2" \
#     --pooling "eos" \
#     --dataset_name "TIGER-Lab/MMEB-train" \
#     --subset_name "MSCOCO" \
#     --dataset_split "original" \
#     --image_dir "vlm2vec_train/MMEB-train" \
#     --percent_data 0.3 \
#     --output_dir "training/mps_2epoch" \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 1 \
#     --learning_rate 1e-4 \
#     --num_train_epochs 2 \
#     --save_total_limit 5 \
#     --logging_steps 1 \
#     --save_strategy "epoch" \
#     --seed 42 \
#     --weight_decay 0.01 \
#     --normalize True \
#     --teacher_normalize True \
#     --lr_scheduler_type "cosine" \
#     --warmup_ratio 0.04 \
#     --kd_weight 0.3 \
#     --kd_loss_type "mps_loss" \
#     --mps_recon_weight 0.5 \
#     --mps_recon_loss_type "mse" \
#     --mps_synergy_weight 1.0 \
#     --mps_magnitude_weight 0.1 \
#     --mps_orthogonality_weight 0.5 \
#     --mps_freeze_teacher_estimator True \
#     --image_resolution "low" \
#     --projector_config_path "./config/projector_config.json" \
#     --projector_lr 5e-5
