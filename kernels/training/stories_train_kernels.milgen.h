#ifndef ORION_STORIES_TRAIN_KERNELS_H
#define ORION_STORIES_TRAIN_KERNELS_H

#import "../../core/mil_builder.h"
#import "../../core/ane_runtime.h"

// MIL generators for Stories110M training kernels.
// 6 kernel types per layer (from ANEgpt):
//
// Forward:
//   fwdAttn:  RMSNorm → QKV → SDPA → Wo (taps: Q, K, V, scores)
//   fwdFFN:   RMSNorm → SwiGLU (W1, W3, SiLU, W2)
//
// Backward (dx path on ANE, dW on CPU):
//   ffnBwd:   W2^T → SiLU_bwd → W1^T, W3^T
//   sdpaBwd1: Wo^T → SDPA bwd part 1 (dV, attn probs, dp)
//   sdpaBwd2: softmax grad → dQ, dK (weight-free)
//   qkvBwd:   Wq^T + Wk^T + Wv^T → dx

// TODO(M3): Implement MIL generators for each kernel type

NSString* orion_milgen_fwd_attn(int layer_idx, const OrionModelConfig* cfg);
NSString* orion_milgen_fwd_ffn(int layer_idx, const OrionModelConfig* cfg);
NSString* orion_milgen_ffn_bwd(int layer_idx, const OrionModelConfig* cfg);
NSString* orion_milgen_sdpa_bwd1(int layer_idx, const OrionModelConfig* cfg);
NSString* orion_milgen_sdpa_bwd2(int layer_idx, const OrionModelConfig* cfg);
NSString* orion_milgen_qkv_bwd(int layer_idx, const OrionModelConfig* cfg);

#endif // ORION_STORIES_TRAIN_KERNELS_H
