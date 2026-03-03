#ifndef ORION_ANE_PROGRAM_CACHE_H
#define ORION_ANE_PROGRAM_CACHE_H

#import "ane_runtime.h"

/// Binding that identifies which weights to use for compilation.
typedef struct {
    const char* weights_id; // e.g., "ckpt_00012000", "base"
    int bucket;             // Sequence length bucket
} OrionWeightsBinding;

/// Look up a cached program by composite key.
/// Cache key: (kernel_name, layer_idx, weights_id, bucket)
/// @param kernel_name  Name of the kernel (e.g., "prefill_attn", "fwd_ffn")
/// @param layer_idx    Layer index (-1 for non-layer-specific kernels)
/// @param wb           Weights binding (checkpoint id + bucket)
/// @return Cached program, or NULL on miss. Caller must NOT release the returned program.
OrionProgram* orion_cache_lookup(
    const char* kernel_name,
    int layer_idx,
    const OrionWeightsBinding* wb
);

/// Store a compiled program in the cache.
/// The cache takes ownership — caller must NOT release the program after storing.
/// If a program already exists for this key, the old one is released and replaced.
/// @param kernel_name  Name of the kernel
/// @param layer_idx    Layer index (-1 for non-layer-specific kernels)
/// @param wb           Weights binding
/// @param program      Compiled program to store (cache takes ownership)
void orion_cache_store(
    const char* kernel_name,
    int layer_idx,
    const OrionWeightsBinding* wb,
    OrionProgram* program
);

/// Evict all cached programs for a given weights_id.
/// Released programs are freed via orion_release_program.
void orion_cache_evict(const char* weights_id);

/// Evict all cached programs.
void orion_cache_clear(void);

/// Get current cache size (number of programs).
int orion_cache_size(void);

#pragma mark - T086: Weights ID Abstraction

/// Generate a training weights_id: "step_00000001", "step_00000002", ...
/// Each call increments the counter. Thread-safe.
/// WARNING: Returns pointer to a static buffer — only valid until the next call.
/// Copy with strdup() if you need to keep it across calls.
const char* orion_weights_id_next_step(void);

/// Reset the training step counter (e.g., after loading a checkpoint).
void orion_weights_id_reset(int start_step);

/// Get the current training step number.
int orion_weights_id_current_step(void);

#endif // ORION_ANE_PROGRAM_CACHE_H
