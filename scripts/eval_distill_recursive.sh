#!/bin/bash
set -euo pipefail

# Eval pipeline for recursive distillation checkpoints.
# IMPORTANT:
# - Recursive behavior in this repo is implemented inside the training KD loss.
# - Inference/evaluation uses the saved student checkpoint with the normal forward pass.
# - Therefore, eval_mmeb.py remains the correct evaluation entrypoint.

SUBSETS=(
  "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
)

DEFAULT_TRAIN_DIR="training/recursive_distill_cls"

resolve_model_path() {
  local input_path="$1"

  if [[ -n "${input_path}" ]]; then
    echo "${input_path}"
    return
  fi

  if [[ -d "${DEFAULT_TRAIN_DIR}/checkpoint-final" ]]; then
    echo "${DEFAULT_TRAIN_DIR}/checkpoint-final"
    return
  fi

  local latest_ckpt
  latest_ckpt=$(find "${DEFAULT_TRAIN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)

  if [[ -z "${latest_ckpt}" ]]; then
    echo "No checkpoint found in ${DEFAULT_TRAIN_DIR}. Pass MODEL_PATH explicitly." >&2
    exit 1
  fi

  echo "${latest_ckpt}"
}

MODEL_PATH=$(resolve_model_path "${1:-}")
OUTPUT_PATH=${2:-"MMEB-eval_outputs/recursive_distill_cls"}
RECURSIVE_EVAL_STEPS=${3:-2}

echo "Using model checkpoint: ${MODEL_PATH}"
echo "Eval output path: ${OUTPUT_PATH}"
echo "Recursive eval steps: ${RECURSIVE_EVAL_STEPS}"
echo "Running eval_mmeb_recursive.py with recursive forward-pass emulation."

python eval_mmeb_recursive.py \
  --model_name "${MODEL_PATH}" \
  --encode_output_path "${OUTPUT_PATH}" \
  --lora True \
  --lora_r 64 \
  --lora_alpha 64 \
  --pooling eos \
  --model_backbone llava_qwen2 \
  --normalize True \
  --bf16 \
  --dataset_name TIGER-Lab/MMEB-eval \
  --subset_name "${SUBSETS[@]}" \
  --dataset_split test \
  --per_device_eval_batch_size 1 \
  --seed 42 \
  --recursive_eval_steps "${RECURSIVE_EVAL_STEPS}" \
  --image_dir eval_images/ \
  --tgt_prefix_mod \
  --image_resolution low \
  --load_pretrained_lora True \
  --report_to none
