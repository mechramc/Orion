#ifndef ORION_STORIES_TRAIN_KERNELS_H
#define ORION_STORIES_TRAIN_KERNELS_H

#import "../../core/mil_builder.h"
#import "../../core/ane_runtime.h"

// MIL generators for Stories110M training kernels.
// 6 kernel types per layer (from ANEgpt):
//
// Forward (T064, T065):
//   fwdAttn:  RMSNorm → QKV → SDPA → Wo (taps: Q, K, V, attn, xn)
//   fwdFFN:   RMSNorm → SwiGLU (W1, W3, SiLU, W2) (taps: h1, h3, gate, xn)
//
// Backward (T066-T069, dx path on ANE, dW on CPU):
//   ffnBwd:   W2^T → SiLU_bwd → W1^T, W3^T
//   sdpaBwd1: Wo^T → SDPA bwd part 1 (dV, attn probs, dp)
//   sdpaBwd2: softmax grad → dQ, dK (weight-free)
//   qkvBwd:   Wq^T + Wk^T + Wv^T → dx

/// T064: Forward attention with multi-output taps.
/// Input:  fp16 [1, d_model, 1, seq]
/// Outputs (6): fp16 [1,d,1,seq] each — wo_out, q_out, k_out, v_out, attn_out, rms1_out
NSString* orion_milgen_fwd_attn(int layer_idx, const OrionModelConfig* cfg);

/// T065: Forward FFN with multi-output taps.
/// Input:  fp16 [1, d_model, 1, seq]
/// Outputs (5): fp16 — w2_out[d], w1_out[h], w3_out[h], gate[h], rms2_out[d]
NSString* orion_milgen_fwd_ffn(int layer_idx, const OrionModelConfig* cfg);

/// T066: FFN backward (SiLU chain rule).
/// Input:  fp16 [1, d + 2*hidden, 1, seq] (concat dffn, h1, h3)
/// Outputs (3): fp16 — dx[d], dh1[h], dh3[h]
NSString* orion_milgen_ffn_bwd(int layer_idx, const OrionModelConfig* cfg);

/// T067: SDPA backward part 1 (Wo^T + attention recompute).
/// Input:  fp16 [1, 4*d, 1, seq] (concat qf, kf, vf, dx2f)
/// Outputs (3): fp16 — dvf[d], pf[nh*s], dpf[nh*s]
NSString* orion_milgen_sdpa_bwd1(int layer_idx, const OrionModelConfig* cfg);

/// T068: SDPA backward part 2 (softmax grad → dQ, dK, weight-free).
/// Input:  fp16 [1, 2*score_ch + 2*d, 1, seq] (concat pf, dpf, qf, kf)
/// Outputs (2): fp16 — dqf[d], dkf[d]
NSString* orion_milgen_sdpa_bwd2(int layer_idx, const OrionModelConfig* cfg);

/// T069: QKV backward (transpose projections → sum).
/// Input:  fp16 [1, 3*d, 1, seq] (concat dq, dk, dv)
/// Output: fp16 [1, d, 1, seq] = dx (single output)
NSString* orion_milgen_qkv_bwd(int layer_idx, const OrionModelConfig* cfg);

#endif // ORION_STORIES_TRAIN_KERNELS_H
