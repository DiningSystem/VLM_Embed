#!/bin/bash

# ==== CONFIG ====
HF_TOKEN="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"
REPO_ID="DiningSystem/vlm-weights"
FILE_OR_DIR="./training"
ZIP_NAME="training.zip"

# ==== LOGIN ====
huggingface-cli login --token "$HF_TOKEN"

# ==== ZIP FILE ====
echo "Zipping $FILE_OR_DIR ..."
zip -r "$ZIP_NAME" "$FILE_OR_DIR"

# ==== UPLOAD ====
echo "Uploading to Hugging Face repo: $REPO_ID ..."
huggingface-cli upload "$REPO_ID" "$ZIP_NAME" "$ZIP_NAME"

echo "Done!"