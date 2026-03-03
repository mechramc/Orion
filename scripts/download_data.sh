#!/bin/bash
# Download pretokenized TinyStories dataset for Stories110M training.
# Source: HuggingFace (Karpathy's pretokenized shards)
# Size: ~41MB (~20M tokens)

set -euo pipefail

DATA_DIR="$(dirname "$0")/../data"
mkdir -p "$DATA_DIR"

URL="https://huggingface.co/datasets/karpathy/llama2c/resolve/main/tinystories_data00.bin"
DEST="$DATA_DIR/tinystories_data00.bin"

if [ -f "$DEST" ]; then
    echo "Data already exists at $DEST"
    exit 0
fi

echo "Downloading TinyStories pretokenized data..."
curl -L -o "$DEST" "$URL"
echo "Downloaded to $DEST ($(du -h "$DEST" | cut -f1))"
