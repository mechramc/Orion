#ifndef ORION_GPT2_PREFILL_ATTN_H
#define ORION_GPT2_PREFILL_ATTN_H

#import "../../core/mil_builder.h"
#import "../../core/ane_runtime.h"

// MIL generator for GPT-2 attention prefill kernel.
// Generates a single MIL program per layer that computes:
//   LayerNorm → Q,K,V projections → Causal Attention → Output projection
//
// Causal attention is decomposed (ANE ignores masks):
//   Q@K^T / sqrt(d) → explicit causal mask → softmax → @V
//
// Outputs: hidden states + K,V for KV cache

// TODO(M2): Implement MIL generation for GPT-2 attention prefill
NSString* orion_milgen_gpt2_prefill_attn(int layer_idx, int seq_len,
                                          const OrionModelConfig* cfg);

#endif // ORION_GPT2_PREFILL_ATTN_H
