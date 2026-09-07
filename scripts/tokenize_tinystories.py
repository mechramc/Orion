#!/usr/bin/env python3
"""Tokenize raw TinyStories text into the pretokenized uint16 format the trainer reads.

kernels/training/data_loader.m mmaps the output as a flat array of uint16 token ids with no
header and no magic — it reads seq_len tokens as input and the same window shifted by one as
target, wrapping at EOF. This script produces exactly that, using the Llama2 sentencepiece model
already committed at tokenizer/data/llama2_tokenizer.model (vocab 32000, so every id fits in a
uint16).

Documents are separated by <|endoftext|> in the source text. Each document is encoded with BOS
prepended and no EOS, matching Karpathy's llama2.c pretokenization convention, then concatenated.

This exists because the pretokenized shard the old download script pointed at
(huggingface.co/datasets/karpathy/llama2c) no longer exists. Tokenizing from the raw text with
the repo's own tokenizer is the reproducible substitute.

Usage:
    python scripts/tokenize_tinystories.py \
        --input data/TinyStories-valid.txt \
        --output data/tinystories_data00.bin

Requires: pip install sentencepiece numpy
"""

import argparse
import os
import sys

import numpy as np
import sentencepiece as spm

DEFAULT_TOKENIZER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tokenizer", "data", "llama2_tokenizer.model"
)

DOC_SEPARATOR = "<|endoftext|>"


def tokenize(input_path: str, output_path: str, tokenizer_path: str, max_tokens: int) -> int:
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    vocab_size = sp.get_piece_size()
    if vocab_size > 65536:
        sys.exit(f"vocab {vocab_size} does not fit in uint16")

    with open(input_path, "r", encoding="utf-8") as f:
        raw = f.read()

    docs = [d.strip() for d in raw.split(DOC_SEPARATOR)]
    docs = [d for d in docs if d]

    all_tokens = []
    total = 0
    for i, doc in enumerate(docs):
        ids = sp.encode(doc, add_bos=True, add_eos=False)
        all_tokens.append(np.array(ids, dtype=np.uint16))
        total += len(ids)
        if max_tokens and total >= max_tokens:
            print(f"  reached --max-tokens {max_tokens} after {i + 1} documents")
            break
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{len(docs)} documents, {total} tokens")

    tokens = np.concatenate(all_tokens)
    if max_tokens:
        tokens = tokens[:max_tokens]

    # Guard the embedding lookup: an id >= vocab is an out-of-bounds read in the ANE embedding
    # table, which shows up as a crash a long way from here.
    if tokens.max() >= vocab_size:
        sys.exit(f"token id {tokens.max()} >= vocab {vocab_size}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tokens.tofile(output_path)
    return len(tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Raw TinyStories text file")
    parser.add_argument("--output", required=True, help="Output .bin path (flat uint16)")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        help="Llama2 sentencepiece model (default: the one in tokenizer/data/)")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Stop after roughly this many tokens (0 = whole file)")
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer):
        sys.exit(f"tokenizer model not found: {args.tokenizer}")
    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}")

    print(f"Tokenizing {args.input} with {args.tokenizer}")
    n = tokenize(args.input, args.output, args.tokenizer, args.max_tokens)
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone: {n} tokens, {size_mb:.1f} MB uint16")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
