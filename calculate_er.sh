#!/bin/bash

# SUBSETS=(
#   "VOC2007"
#   "OK-VQA"
# )

SUBSETS=(
  #"ImageNet_1K"
   "OK-VQA" #"A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA"
)

MODEL="./training/FastVLM-0.5B_base_ImageNet1K_ER=0.8/checkpoint-final"
# MODEL="raghavlite/B3_Qwen2_2B"

# =========================================================================
# Dùng torchrun để khởi chạy
# =========================================================================
python calculate_er.py \
    --model_name $MODEL \
    --lora True \
    --lora_r 64 \
    --lora_alpha 64 \
    --model_backbone "llava_qwen2" \
    --pooling "eos" \
    --dataset_name "TIGER-Lab/MMEB-train" \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split "original" \
    --image_dir "./vlm2vec_train/MMEB-train" \
    --encode_output_path "./ER_outputs/teacher_er_record/" \
    --per_device_train_batch_size 1 \
    --bf16 \
    --seed 42 \
    --image_resolution "low" \
    --normalize True \