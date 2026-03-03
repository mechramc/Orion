#ifndef ORION_DECODE_ANE_H
#define ORION_DECODE_ANE_H

#import "../../core/ane_runtime.h"
#import "../../model/weight_loader.h"
#import "kv_cache.h"

/// T101: ANE-accelerated single-token decode step.
///
/// Per layer (×12):
///   1. ANE decode_proj: x → LN1 → Q,K,V  (3 outputs)
///   2. CPU: extract Q[0],K[0],V[0] → append cache → cross-attention → output proj → residual
///   3. ANE decode_ffn: x → LN2 → FFN → residual → hidden  (1 output)
///   4. CPU: extract hidden[0] → x for next layer
///
/// Final: CPU layernorm + logits projection.
///
/// Programs compiled on first call and cached via orion_cache_lookup/store.
///
/// @param w         Loaded GPT-2 weights (CPU, for embedding + attention + logits)
/// @param kv        KV cache (read for past K,V; updated with new K,V)
/// @param token     Input token id for this step
/// @param blob_dir  Path to blob directory containing weight files
/// @param logits    Output logits buffer [vocab]
/// @return true on success, false on compile/eval failure
bool orion_ane_decode_step(const OrionGPT2Weights* w,
                            OrionKVCache* kv,
                            int token,
                            const char* blob_dir,
                            float* logits);

#endif // ORION_DECODE_ANE_H
