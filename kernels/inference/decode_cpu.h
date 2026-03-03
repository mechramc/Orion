#ifndef ORION_DECODE_CPU_H
#define ORION_DECODE_CPU_H

#import "../../core/ane_runtime.h"
#import "kv_cache.h"

/// CPU-based autoregressive decode step.
/// Runs one transformer forward pass for a single token position
/// using the KV cache for previously computed keys/values.

/// Run a single decode step on CPU.
/// @param cfg       Model configuration
/// @param kv        KV cache (read for past K,V; updated with new K,V)
/// @param weights   Pointer to all model weights (fp32)
/// @param token     Input token id for this step
/// @param logits    Output logits buffer (vocab_size floats)
void orion_decode_cpu_step_impl(
    const OrionModelConfig* cfg,
    OrionKVCache* kv,
    const void* weights,
    int token,
    float* logits
);

/// Sample a token from logits with temperature and top-p.
int orion_sample_token(const float* logits, int vocab_size,
                       float temperature, float top_p, uint64_t* rng_state);

#endif // ORION_DECODE_CPU_H
