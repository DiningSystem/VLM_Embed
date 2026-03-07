SUBSETS=(
  # "ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
   # "ImageNet-1K"
   "OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
#"ImageNet-1K" "N24News" "HatefulMemes"
)

MODEL=./training//vqa_ov_05eos_03er_combined_student/checkpoint-final
python eval_mmeb.py \
    --model_name $MODEL \
    --encode_output_path ./MMEB-eval_outputs//vqa_ov_05eos_03er_combined_student1/ \
    --lora True --lora_r 8 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_onevision \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 1 \
    --seed 210 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --image_resolution "low" \
    --load_pretrained_lora True \
    --report_to none
