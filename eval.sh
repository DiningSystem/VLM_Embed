SUBSETS=(
   #"ImageNet-1K" "N24News" "HatefulMemes" "VOC2007" "SUN397"
   # "ImageNet-1K"
   #"OK-VQA" "A-OKVQA" "DocVQA" "InfographicsVQA" "ChartQA" "Visual7W"
#"ImageNet-1K" "N24News" "HatefulMemes"
#"VOC2007"
#"MSCOCO" "RefCOCO" "RefCOCO-Matching" "Visual7W-Pointing"
"VisDial" "CIRR" "VisualNews_i2t" "VisualNews_t2i" "MSCOCO_i2t" "MSCOCO_t2i" "NIGHTS" "WebQA" # "OVEN" "FashionIQ" "EDIS" "Wiki-SS-NQ"
)

MODEL=./training/results_0605/retrieval1epoch_05eos_03er_combined_student/checkpoint-final
#MODEL=./training/results_0605/qwen7B_05eos_03er_combined_student/checkpoint-final
python eval_mmeb.py \
    --model_name $MODEL \
    --encode_output_path ./MMEB-eval_outputs/combine_eos_er_ret3/ \
    --lora True --lora_r 8 --lora_alpha 64 \
    --pooling eos \
    --model_backbone llava_qwen2 \
    --normalize True \
    --bf16 \
    --dataset_name TIGER-Lab/MMEB-eval \
    --subset_name "${SUBSETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size 15 \
    --seed 110 \
    --image_dir eval_images/ \
    --tgt_prefix_mod \
    --image_resolution "mid" \
    --load_pretrained_lora True \
    --report_to none
