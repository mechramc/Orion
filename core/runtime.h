#ifndef ORION_RUNTIME_H
#define ORION_RUNTIME_H

#import "ane_runtime.h"
#import "kernel.h"
#import "ane_program_cache.h"

/// Thin wrapper around ANE init + compile budget tracking.
/// Does NOT own inference/training loops — those remain model-specific.
typedef struct {
    bool ane_ready;
    int total_compiles;
    int compile_budget;    // Default: 119 (ANE limit per process)
} OrionRuntime;

/// Create a runtime with default compile budget (119).
OrionRuntime* orion_runtime_create(void);

/// Initialize the ANE. Must be called before eval_kernel.
/// @return true if ANE is available, false otherwise.
bool orion_runtime_init_ane(OrionRuntime* rt);

/// Compile-or-cache + eval a kernel, tracking compile budget.
/// @return true on success, false on compile/eval failure.
bool orion_runtime_eval_kernel(OrionRuntime* rt,
                               const OrionKernel* kernel,
                               int layer_idx, int bucket,
                               const OrionModelConfig* cfg,
                               const char* blob_dir,
                               const OrionWeightsBinding* wb,
                               IOSurfaceRef* inputs, int n_inputs,
                               IOSurfaceRef* outputs, int n_outputs);

/// Check if the runtime is approaching the compile limit.
/// @param remaining  If non-NULL, set to remaining compile budget.
/// @return true if exec() restart is needed (budget exhausted).
bool orion_runtime_needs_restart(const OrionRuntime* rt, int* remaining);

/// Clear the program cache and reset compile counter.
void orion_runtime_clear_cache(OrionRuntime* rt);

/// Free the runtime.
void orion_runtime_free(OrionRuntime* rt);

#endif // ORION_RUNTIME_H
