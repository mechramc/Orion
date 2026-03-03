#!/usr/bin/env python3
"""Convert Llama2/Stories weights to Orion BLOBFILE format.

Supports:
  - Karpathy's stories110M.bin format (custom binary)
  - HuggingFace Llama2 checkpoints

Usage:
    python hf_to_blobs_llama.py --checkpoint ../../model/weights/stories110M.bin \
                                --output ../../model/blobs/stories110m/
"""

import argparse
import os
import struct
import numpy as np

# TODO(M3): Implement conversion
# 1. Parse Karpathy's binary format or HuggingFace checkpoint
# 2. Extract per-layer weights: Wq, Wk, Wv, Wo, W1, W2, W3, rms_norm
# 3. Convert to fp16 BLOBFILE format


def main():
    parser = argparse.ArgumentParser(description='Convert Llama2/Stories weights to BLOBFILE format')
    parser.add_argument('--checkpoint', required=True, help='Input checkpoint path')
    parser.add_argument('--output', required=True, help='Output directory for blob files')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # TODO(M3): Implement
    print(f"Would convert {args.checkpoint} to {args.output}")
    print("Not yet implemented")


if __name__ == '__main__':
    main()
