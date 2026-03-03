#ifndef ORION_GPT2_PREFILL_ATTN_H
#define ORION_GPT2_PREFILL_ATTN_H

#import "../../core/mil_builder.h"
#import "../../core/ane_runtime.h"

// T047: MIL generator for GPT-2 attention prefill kernel.
//
// Generates a MIL program for one transformer layer's attention block:
//   Input: hidden state fp32 [1, d_model, 1, seq]
//   Ops:   Cast→fp16 → LN1 → Q,K,V projections → Causal Attention → Output proj → Residual → Cast→fp32
//   Outputs: hidden state [1, d_model, 1, seq], K [1, d_model, 1, seq], V [1, d_model, 1, seq]
//
// K,V outputs are the linear projections (before head-splitting), used to populate KV cache.
//
// Weight blob paths referenced in generated MIL (caller must provide in weight dict):
//   @model_path/layer{i}/ln1_g.bin, ln1_b.bin    — LayerNorm gamma, beta
//   @model_path/layer{i}/wq.bin, bq.bin          — Q projection weight [d,d,1,1], bias [1,d,1,1]
//   @model_path/layer{i}/wk.bin, bk.bin          — K projection
//   @model_path/layer{i}/wv.bin, bv.bin          — V projection
//   @model_path/layer{i}/wo.bin, bo.bin          — Output projection
//   @model_path/masks/causal_{seq}.bin           — Causal mask [1,1,seq,seq]

/// Generate complete MIL program for GPT-2 attention prefill (one layer).
/// @param layer_idx Layer index (0-11 for GPT-2 124M)
/// @param seq_len   Bucket sequence length (must be one of {32,64,128,256,512,1024})
/// @param cfg       Model configuration
/// @return Complete MIL program text (3 outputs: hidden, k_cache, v_cache)
NSString* orion_milgen_gpt2_prefill_attn(int layer_idx, int seq_len,
                                          const OrionModelConfig* cfg);

/// Create a BLOBFILE-format causal mask for attention.
/// mask[i][j] = 0 if j <= i (can attend), -1e4 if j > i (masked).
/// @param seq_len Sequence length
/// @return NSData containing full BLOBFILE (128-byte header + fp16 data [1,1,seq,seq])
NSData* orion_make_causal_mask_blob(int seq_len);

/// Get the MIL blob path for the causal mask of a given sequence length.
/// @param seq_len Sequence length
/// @return Path string like "@model_path/masks/causal_32.bin"
NSString* orion_causal_mask_path(int seq_len);

#endif // ORION_GPT2_PREFILL_ATTN_H
