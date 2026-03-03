#!/usr/bin/env python3
"""Convert HuggingFace GPT-2 weights to Orion BLOBFILE format.

T024: BLOBFILE writer (make_blob_header + convert_tensor_to_blob)
T025: GPT-2 weight converter (full HF → per-layer blobs)

BLOBFILE format (matching ANEgpt convention):
  - 128-byte header (magic, data size, data offset)
  - fp16 weight data

Usage:
    python hf_to_blobs_gpt2.py --output ../../model/blobs/gpt2_124m/

Requires: pip install torch transformers
"""

import argparse
import os
import struct
import numpy as np


def make_blob_header(data_size: int) -> bytes:
    """Create 128-byte BLOBFILE header matching ANEgpt format.

    Layout:
      [0]    = 0x01 (global magic 0)
      [4]    = 0x02 (global magic 4)
      [64:68]= 0xDEADBEEF (chunk magic, little-endian)
      [68]   = 0x01 (version)
      [72:76]= data_size (uint32, LE)
      [80:84]= 128 (data offset from file start, uint32, LE)
    """
    header = bytearray(128)
    header[0] = 0x01
    header[4] = 0x02
    header[64:68] = bytes([0xEF, 0xBE, 0xAD, 0xDE])
    header[68] = 0x01
    struct.pack_into('<I', header, 72, data_size)
    struct.pack_into('<I', header, 80, 128)
    return bytes(header)


def convert_tensor_to_blob(tensor_fp32: np.ndarray, output_path: str):
    """Convert a float32 numpy tensor to BLOBFILE format (fp16)."""
    tensor_fp16 = tensor_fp32.astype(np.float16)
    data = tensor_fp16.tobytes()
    header = make_blob_header(len(data))
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(header)
        f.write(data)
    return len(data)


def convert_tensor_transposed(tensor_fp32: np.ndarray, output_path: str):
    """Convert with transpose (for row-major to col-major weight layout).
    ANE conv weights are [out_channels, in_channels, 1, 1]."""
    return convert_tensor_to_blob(tensor_fp32.T.copy(), output_path)


# GPT-2 parameter name mapping:
# HF name → (blob_name, needs_transpose)
# h.{i}.attn.c_attn.weight → layer{i}/wqkv.bin (transposed, needs split)
# h.{i}.attn.c_attn.bias → layer{i}/bqkv.bin (split into q,k,v)
# h.{i}.attn.c_proj.weight → layer{i}/wo.bin
# h.{i}.attn.c_proj.bias → layer{i}/bo.bin
# h.{i}.ln_1.weight → layer{i}/ln1_g.bin
# h.{i}.ln_1.bias → layer{i}/ln1_b.bin
# h.{i}.mlp.c_fc.weight → layer{i}/wfc.bin
# h.{i}.mlp.c_fc.bias → layer{i}/bfc.bin
# h.{i}.mlp.c_proj.weight → layer{i}/wproj.bin
# h.{i}.mlp.c_proj.bias → layer{i}/bproj.bin
# h.{i}.ln_2.weight → layer{i}/ln2_g.bin
# h.{i}.ln_2.bias → layer{i}/ln2_b.bin
# wte.weight → wte.bin
# wpe.weight → wpe.bin
# ln_f.weight → ln_f_g.bin
# ln_f.bias → ln_f_b.bin


def convert_gpt2(model_name: str, output_dir: str):
    """Convert GPT-2 model from HuggingFace to BLOBFILE blobs."""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        print("Error: pip install torch transformers")
        return False

    print(f"Loading {model_name}...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    state = model.state_dict()

    config = model.config
    n_layer = config.n_layer
    n_head = config.n_head
    d_model = config.n_embd
    head_dim = d_model // n_head

    print(f"Config: n_layer={n_layer}, n_head={n_head}, d_model={d_model}")
    os.makedirs(output_dir, exist_ok=True)

    total_bytes = 0
    file_count = 0

    def save(name, tensor, transpose=False):
        nonlocal total_bytes, file_count
        path = os.path.join(output_dir, name)
        arr = tensor.detach().cpu().float().numpy()
        if transpose:
            sz = convert_tensor_transposed(arr, path)
        else:
            sz = convert_tensor_to_blob(arr, path)
        total_bytes += sz
        file_count += 1
        print(f"  {name}: {arr.shape} → {sz} bytes fp16")

    # Token + positional embeddings
    save("wte.bin", state["transformer.wte.weight"])  # [vocab, d_model]
    save("wpe.bin", state["transformer.wpe.weight"])  # [max_pos, d_model]

    # Per-layer weights
    for i in range(n_layer):
        prefix = f"transformer.h.{i}"
        ldir = f"layer{i}"
        os.makedirs(os.path.join(output_dir, ldir), exist_ok=True)

        # Layer norm 1
        save(f"{ldir}/ln1_g.bin", state[f"{prefix}.ln_1.weight"])
        save(f"{ldir}/ln1_b.bin", state[f"{prefix}.ln_1.bias"])

        # Attention: c_attn is [d_model, 3*d_model] (fused QKV)
        # GPT-2 uses Conv1D, so weight is [in, out] — need transpose for conv [out, in, 1, 1]
        qkv_w = state[f"{prefix}.attn.c_attn.weight"]  # [d_model, 3*d_model]
        qkv_b = state[f"{prefix}.attn.c_attn.bias"]      # [3*d_model]

        # Split into Q, K, V
        wq, wk, wv = qkv_w.split(d_model, dim=1)
        bq, bk, bv = qkv_b.split(d_model, dim=0)

        save(f"{ldir}/wq.bin", wq, transpose=True)  # [d_model, d_model] → [d_model, d_model, 1, 1]
        save(f"{ldir}/wk.bin", wk, transpose=True)
        save(f"{ldir}/wv.bin", wv, transpose=True)
        save(f"{ldir}/bq.bin", bq)
        save(f"{ldir}/bk.bin", bk)
        save(f"{ldir}/bv.bin", bv)

        # Output projection
        save(f"{ldir}/wo.bin", state[f"{prefix}.attn.c_proj.weight"], transpose=True)
        save(f"{ldir}/bo.bin", state[f"{prefix}.attn.c_proj.bias"])

        # Layer norm 2
        save(f"{ldir}/ln2_g.bin", state[f"{prefix}.ln_2.weight"])
        save(f"{ldir}/ln2_b.bin", state[f"{prefix}.ln_2.bias"])

        # MLP (FFN)
        save(f"{ldir}/wfc.bin", state[f"{prefix}.mlp.c_fc.weight"], transpose=True)
        save(f"{ldir}/bfc.bin", state[f"{prefix}.mlp.c_fc.bias"])
        save(f"{ldir}/wproj.bin", state[f"{prefix}.mlp.c_proj.weight"], transpose=True)
        save(f"{ldir}/bproj.bin", state[f"{prefix}.mlp.c_proj.bias"])

    # Final layer norm
    save("ln_f_g.bin", state["transformer.ln_f.weight"])
    save("ln_f_b.bin", state["transformer.ln_f.bias"])

    print(f"\nDone: {file_count} files, {total_bytes / 1024 / 1024:.1f} MB total (fp16)")
    print(f"Output: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert GPT-2 weights to BLOBFILE format')
    parser.add_argument('--model', default='gpt2', help='HuggingFace model name (default: gpt2)')
    parser.add_argument('--output', required=True, help='Output directory for blob files')
    args = parser.parse_args()

    convert_gpt2(args.model, args.output)


if __name__ == '__main__':
    main()
