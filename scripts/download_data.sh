#!/bin/bash
# Fetch TinyStories and pretokenize it for Stories110M training.
#
# The previous version of this script pulled a prebuilt shard from
# huggingface.co/datasets/karpathy/llama2c. That dataset repo no longer exists (HF answers 401),
# and because the curl had no --fail the script wrote the 29-byte error body to
# data/tinystories_data00.bin and exited 0 — after which its own "already exists" guard treated
# the corpse as a valid cached dataset. So this now downloads the raw text and tokenizes it
# locally with the Llama2 sentencepiece model already committed at
# tokenizer/data/llama2_tokenizer.model, and fails loudly instead of quietly.
#
# Output: data/tinystories_data00.bin — flat uint16 token ids, the format
# kernels/training/data_loader.m mmaps.
#
# Requires: python3 with sentencepiece + numpy (pip install sentencepiece numpy)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT/data"
RAW="$DATA_DIR/TinyStories-valid.txt"
DEST="$DATA_DIR/tinystories_data00.bin"
TOKENIZER="$ROOT/tokenizer/data/llama2_tokenizer.model"
RAW_URL="https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"
PYTHON="${PYTHON:-python3}"

# A valid dataset is many megabytes. Anything smaller is a failed download, not a cache hit.
MIN_BYTES=1000000

mkdir -p "$DATA_DIR"

file_bytes() { [ -f "$1" ] && wc -c < "$1" | tr -d ' ' || echo 0; }

if [ "$(file_bytes "$DEST")" -ge "$MIN_BYTES" ]; then
    echo "Data already exists at $DEST ($(du -h "$DEST" | cut -f1))"
    exit 0
fi

if [ -f "$DEST" ]; then
    echo "Removing undersized $DEST ($(file_bytes "$DEST") bytes) — previous download failed."
    rm -f "$DEST"
fi

if [ ! -f "$TOKENIZER" ]; then
    echo "ERROR: tokenizer model missing at $TOKENIZER" >&2
    exit 1
fi

if [ "$(file_bytes "$RAW")" -lt "$MIN_BYTES" ]; then
    echo "Downloading raw TinyStories text (~18MB)..."
    rm -f "$RAW"
    if ! curl -L --fail -o "$RAW" "$RAW_URL"; then
        rm -f "$RAW"
        echo "ERROR: download failed from $RAW_URL" >&2
        exit 1
    fi
fi

echo "Tokenizing with $TOKENIZER..."
if ! "$PYTHON" "$ROOT/scripts/tokenize_tinystories.py" --input "$RAW" --output "$DEST"; then
    rm -f "$DEST"
    echo "ERROR: tokenization failed. Install deps with: pip install sentencepiece numpy" >&2
    exit 1
fi

if [ "$(file_bytes "$DEST")" -lt "$MIN_BYTES" ]; then
    rm -f "$DEST"
    echo "ERROR: produced dataset was undersized; removed." >&2
    exit 1
fi

echo "Downloaded and tokenized to $DEST ($(du -h "$DEST" | cut -f1))"
