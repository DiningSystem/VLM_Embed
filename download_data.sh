mkdir -p vlm2vec_train/MMEB-train/images

# classification datasets
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/HatefulMemes.zip
unzip HatefulMemes.zip -d ./vlm2vec_train/MMEB-train/images/
rm HatefulMemes.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/ImageNet_1K.zip
unzip ImageNet_1K.zip -d ./vlm2vec_train/MMEB-train/images/
rm ImageNet_1K.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/VOC2007.zip
unzip VOC2007.zip -d ./vlm2vec_train/MMEB-train/images/
rm VOC2007.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/SUN397.zip
unzip SUN397.zip -d ./vlm2vec_train/MMEB-train/images/
rm SUN397.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/VOC2007.zip
unzip VOC2007.zip -d ./vlm2vec_train/MMEB-train/images/
rm VOC2007.zip

# vqa datasets
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/OK-VQA.zip
unzip OK-VQA.zip -d ./vlm2vec_train/MMEB-train/images/
rm OK-VQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/A-OKVQA.zip
unzip A-OKVQA.zip -d ./vlm2vec_train/MMEB-train/images/
rm A-OKVQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/DocVQA.zip
unzip DocVQA.zip -d ./vlm2vec_train/MMEB-train/images/
rm DocVQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/InfographicsVQA.zip
unzip InfographicsVQA.zip -d ./vlm2vec_train/MMEB-train/images/
rm InfographicsVQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/ChartQA.zip
unzip ChartQA.zip -d ./vlm2vec_train/MMEB-train/images/
rm ChartQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/Visual7W.zip
unzip Visual7W.zip -d ./vlm2vec_train/MMEB-train/images/
rm Visual7W.zip