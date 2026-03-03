#ifndef ORION_GPT2_DECODE_ANE_H
#define ORION_GPT2_DECODE_ANE_H

#import "../../core/mil_builder.h"
#import "../../core/ane_runtime.h"

// T100: MIL generators for single-token ANE decode (v3 architecture).
//
// ANE requires minimum IOSurface allocation of ~49KB. Tensors smaller than
// [1,C,1,16] fail at eval. Decode uses seq=16 as minimum bucket with token
// data at position 0 and zero-padding at positions 1-15.
//
// Per-layer decode uses 2 ANE kernels + CPU cross-attention:
//   1. decode_proj: x [1,d,1,S] → LN1 → Q,K_new,V_new [1,d,1,S] each (ANE)
//   2. CPU: extract Q[0],K[0],V[0] → append K,V to cache → cross-attention → output proj → residual
//   3. decode_ffn: hidden2 [1,d,1,S] → LN2 → FFN → residual → hidden_next (ANE)
//
// Final layer uses prefill_final_ln with seq=S (already works).
//
// Weight blob paths (same as prefill):
//   @model_path/layer{i}/ln1_g.bin, ln1_b.bin, wq.bin, bq.bin, wk.bin, bk.bin, wv.bin, bv.bin
//   @model_path/layer{i}/ln2_g.bin, ln2_b.bin, wfc.bin, bfc.bin, wproj.bin, bproj.bin

/// Minimum seq dimension for ANE decode (IOSurface minimum allocation constraint).
#define ORION_DECODE_SEQ 16

/// Generate MIL program for decode QKV projections (one layer).
/// Input:  x [1, d_model, 1, ORION_DECODE_SEQ] (fp32)
/// Output: q, k_new, v_new [1, d_model, 1, ORION_DECODE_SEQ] each (fp32)
/// Caller places token at seq position 0 and reads Q[0], K[0], V[0].
NSString* orion_milgen_gpt2_decode_proj(int layer_idx, const OrionModelConfig* cfg);

/// Generate MIL program for decode FFN + residual (one layer).
/// Input:  x [1, d_model, 1, ORION_DECODE_SEQ] (fp32)
/// Output: hidden [1, d_model, 1, ORION_DECODE_SEQ] (fp32) — LN2 → FFN → residual
/// Caller places token at seq position 0 and reads hidden[0].
NSString* orion_milgen_gpt2_decode_ffn(int layer_idx, const OrionModelConfig* cfg);

#endif // ORION_GPT2_DECODE_ANE_H
