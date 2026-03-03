#ifndef ORION_PREFILL_ANE_H
#define ORION_PREFILL_ANE_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import "../../core/ane_runtime.h"
#import "../../model/weight_loader.h"
#import "../../kernels/inference/kv_cache.h"

// T051: Prompt padding + layout conversion for ANE prefill
// T052: ANE prefill runner
// T053: K,V extraction from ANE into KV cache

/// Prepare prompt embeddings for ANE: compute wte+wpe, pad to bucket, transpose.
/// CPU layout [seq, d_model] → ANE layout [d_model, bucket_size] with zero-padding.
/// @param w          Loaded GPT-2 weights (for wte, wpe)
/// @param tokens     Input token IDs
/// @param prompt_len Number of prompt tokens
/// @param bucket     Bucket size to pad to
/// @param cfg        Model configuration
/// @return IOSurface containing fp32 [1, d_model, 1, bucket] tensor. Caller must CFRelease.
IOSurfaceRef orion_prepare_ane_input(const OrionGPT2Weights* w,
                                      const int* tokens, int prompt_len,
                                      int bucket, const OrionModelConfig* cfg);

/// Run full ANE prefill: embed → 12×(attn→FFN) → final LN → CPU logits.
/// @param w          Loaded GPT-2 weights
/// @param tokens     Input token IDs
/// @param prompt_len Number of prompt tokens
/// @param cfg        Model configuration
/// @param blob_dir   Path to weight blobs directory
/// @param kv         KV cache to populate (output)
/// @param logits     Output logits buffer [vocab] for last prompt position
/// @return true on success
bool orion_ane_prefill(const OrionGPT2Weights* w,
                        const int* tokens, int prompt_len,
                        const OrionModelConfig* cfg,
                        const char* blob_dir,
                        OrionKVCache* kv,
                        float* logits);

#endif // ORION_PREFILL_ANE_H
