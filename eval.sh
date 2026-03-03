SUBSETS=(
   "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
   # "ImageNet-1K"
  # "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
)

MODEL=./training/cls_07eos_03er_combined_student/checkpoint-final
python eval_mmeb.py \
    --model_name $MODEL \
    --encode_output_path ./MMEB-eval_outputs/cls_07eos_03er_combined_student/ \
    --lora True --lora_r 8 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 16 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --load_pretrained_lora True \
    --report_to none
