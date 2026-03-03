#!/usr/bin/env python3
"""T027: Convert Karpathy/Llama2 Stories weights to Orion BLOBFILE format.

Supports Karpathy's llama2.c binary format (stories110M.bin).
Binary layout: 7 x int32 header, then flat float32 weights.

Usage:
    python hf_to_blobs_llama.py --checkpoint ../../model/weights/stories110M.bin \
                                --output ../../model/blobs/stories110m/
"""

import argparse
import os
import struct
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from hf_to_blobs_gpt2 import convert_tensor_to_blob


def read_llama2c_checkpoint(checkpoint_path: str):
    """Read Karpathy's llama2.c binary checkpoint format.

    Header: 7 x int32 = {dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len}
    Weights: flat float32 arrays in this order:
      - token_embedding_table [vocab_size, dim]
      - rms_att_weight [n_layers, dim]
      - wq [n_layers, dim, dim]
      - wk [n_layers, dim, dim]  (could be dim * n_kv_heads * head_dim)
      - wv [n_layers, dim, dim]
      - wo [n_layers, dim, dim]
      - rms_ffn_weight [n_layers, dim]
      - w1 [n_layers, hidden_dim, dim]
      - w2 [n_layers, dim, hidden_dim]
      - w3 [n_layers, hidden_dim, dim]
      - rms_final_weight [dim]
      - (optional) freq_cis_real, freq_cis_imag for RoPE
    """
    with open(checkpoint_path, 'rb') as f:
        # Read header
        header = struct.unpack('7i', f.read(28))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header

        # Handle negative vocab_size (indicates shared weights)
        shared_weights = vocab_size > 0
        vocab_size = abs(vocab_size)

        head_dim = dim // n_heads
        kv_dim = (dim // n_heads) * n_kv_heads  # For GQA

        print(f"Config: dim={dim}, hidden_dim={hidden_dim}, n_layers={n_layers}, "
              f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, vocab_size={vocab_size}, "
              f"seq_len={seq_len}")
        print(f"  head_dim={head_dim}, kv_dim={kv_dim}, shared_weights={shared_weights}")

        # Read all weights as flat float32
        weights_data = np.frombuffer(f.read(), dtype=np.float32)

    # Parse weights in order
    offset = 0

    def read_weight(shape):
        nonlocal offset
        size = int(np.prod(shape))
        w = weights_data[offset:offset + size].reshape(shape).copy()
        offset += size
        return w

    weights = {}

    # Token embeddings
    weights['embed'] = read_weight((vocab_size, dim))

    # Per-layer RMS attention norm
    weights['rms_att'] = read_weight((n_layers, dim))

    # QKV + output projections
    weights['wq'] = read_weight((n_layers, dim, dim))
    weights['wk'] = read_weight((n_layers, kv_dim, dim))
    weights['wv'] = read_weight((n_layers, kv_dim, dim))
    weights['wo'] = read_weight((n_layers, dim, dim))

    # Per-layer RMS FFN norm
    weights['rms_ffn'] = read_weight((n_layers, dim))

    # FFN weights (SwiGLU: gate, down, up)
    weights['w1'] = read_weight((n_layers, hidden_dim, dim))  # gate
    weights['w2'] = read_weight((n_layers, dim, hidden_dim))  # down
    weights['w3'] = read_weight((n_layers, hidden_dim, dim))  # up

    # Final RMS norm
    weights['rms_final'] = read_weight((dim,))

    config = {
        'dim': dim, 'hidden_dim': hidden_dim, 'n_layers': n_layers,
        'n_heads': n_heads, 'n_kv_heads': n_kv_heads,
        'vocab_size': vocab_size, 'seq_len': seq_len,
        'head_dim': head_dim, 'kv_dim': kv_dim,
        'shared_weights': shared_weights,
    }

    print(f"  Read {offset} / {len(weights_data)} floats "
          f"({offset * 4 / 1024 / 1024:.1f} MB)")

    return config, weights


def convert_stories(checkpoint_path: str, output_dir: str):
    """Convert Stories110M checkpoint to BLOBFILE blobs."""
    config, weights = read_llama2c_checkpoint(checkpoint_path)
    n_layers = config['n_layers']

    os.makedirs(output_dir, exist_ok=True)
    total_bytes = 0
    file_count = 0

    def save(name, tensor):
        nonlocal total_bytes, file_count
        path = os.path.join(output_dir, name)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        sz = convert_tensor_to_blob(tensor, path)
        total_bytes += sz
        file_count += 1
        print(f"  {name}: {tensor.shape} → {sz} bytes fp16")

    # Token embeddings
    save("embed.bin", weights['embed'])

    # Per-layer weights
    for i in range(n_layers):
        ldir = f"layer{i}"
        os.makedirs(os.path.join(output_dir, ldir), exist_ok=True)

        save(f"{ldir}/rms_att.bin", weights['rms_att'][i])
        save(f"{ldir}/wq.bin", weights['wq'][i])
        save(f"{ldir}/wk.bin", weights['wk'][i])
        save(f"{ldir}/wv.bin", weights['wv'][i])
        save(f"{ldir}/wo.bin", weights['wo'][i])
        save(f"{ldir}/rms_ffn.bin", weights['rms_ffn'][i])
        save(f"{ldir}/w1.bin", weights['w1'][i])
        save(f"{ldir}/w2.bin", weights['w2'][i])
        save(f"{ldir}/w3.bin", weights['w3'][i])

    # Final RMS norm
    save("rms_final.bin", weights['rms_final'])

    print(f"\nDone: {file_count} files, {total_bytes / 1024 / 1024:.1f} MB total (fp16)")
    print(f"Output: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert Llama2/Stories weights to BLOBFILE format')
    parser.add_argument('--checkpoint', required=True, help='Input checkpoint path (.bin)')
    parser.add_argument('--output', required=True, help='Output directory for blob files')
    args = parser.parse_args()

    convert_stories(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
