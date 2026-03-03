#ifndef ORION_ANE_PROGRAM_CACHE_H
#define ORION_ANE_PROGRAM_CACHE_H

#import "ane_runtime.h"

/// Binding that identifies which weights to use for compilation.
typedef struct {
    const char* weights_id; // e.g., "ckpt_00012000", "base"
    int bucket;             // Sequence length bucket
} OrionWeightsBinding;

/// Get a cached program or compile a new one.
/// Cache key: (kernel_name, layer_idx, weights_id, bucket, model_config_hash)
/// @param kernel_name  Name of the kernel (e.g., "prefill_attn", "ffn_fwd")
/// @param layer_idx    Layer index (-1 for non-layer-specific kernels)
/// @param wb           Weights binding (checkpoint id + bucket)
/// @param cfg          Model configuration
/// @return Compiled program (cached or freshly compiled)
OrionProgram* orion_get_or_compile(
    const char* kernel_name,
    int layer_idx,
    const OrionWeightsBinding* wb,
    const OrionModelConfig* cfg
);

/// Evict all cached programs for a given weights_id.
void orion_cache_evict(const char* weights_id);

/// Evict all cached programs.
void orion_cache_clear(void);

/// Get current cache size (number of programs).
int orion_cache_size(void);

#endif // ORION_ANE_PROGRAM_CACHE_H
