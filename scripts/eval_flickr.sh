#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/eval_flickr.sh <model_name_or_path> <output_dir> [extra eval args ...]
# Example:
#   bash scripts/eval_flickr.sh Qwen/Qwen2-VL-2B-Instruct ./outputs/flickr_eval --per_device_eval_batch_size 8
MODEL=./training/results_0305/grounding_05eos_03er_combined_student/checkpoint-final
OUTPUT_DIR=./MMEB-eval_outputs/FLICKR_05eos_03er_combined_student


python eval_flickr.py \
  --model_name "$MODEL_NAME" \
  --encode_output_path "$OUTPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --do_eval true \
  --bf16 true \
  --per_device_eval_batch_size 8 \
  --dataloader_num_workers 4 \
  --max_len 4096 \
  "$@"
