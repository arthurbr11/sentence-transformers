#!/bin/bash
set -e

# Your Hugging Face username
USER="arthurbresnu"

# Path where your trained models are stored
BASE_DIR="/fsx/arthur_bresnu/projects/sentence-transformers/outputs"

# Loop over all subdirectories inside outputs/
for MODEL_DIR in "$BASE_DIR"/*/; do
    # Get folder name only (without path)
    MODEL_NAME=$(basename "$MODEL_DIR")
    REPO_ID="$USER/$MODEL_NAME"

    echo ">>> Uploading $MODEL_DIR to $REPO_ID"

    # Create repo on the hub (ignore error if it already exists)
    hf repo create "$MODEL_NAME" --type model || true

    # Upload the model folder
    hf upload-large-folder "$REPO_ID" "$MODEL_DIR" --repo-type=model
done
