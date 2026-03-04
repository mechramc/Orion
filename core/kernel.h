#ifndef ORION_KERNEL_H
#define ORION_KERNEL_H

#import "ane_runtime.h"
#import "ane_program_cache.h"
#import <IOSurface/IOSurface.h>

/// Function pointer type for MIL text generation.
/// @param layer_idx  Layer index (-1 for non-layer-specific kernels)
/// @param bucket     Sequence length bucket (ignored by some kernels)
/// @param cfg        Model configuration
/// @return MIL program text as NSString
typedef NSString* (*OrionMILGenFn)(int layer_idx, int bucket, const OrionModelConfig* cfg);

/// Function pointer type for weight dictionary building.
/// @param layer_idx  Layer index (-1 for non-layer-specific kernels)
/// @param bucket     Sequence length bucket (ignored by some kernels)
/// @param blob_dir   Path to weight blob directory
/// @return NSDictionary for orion_compile_mil, or @{} for weight-free kernels
typedef NSDictionary* (*OrionWDictFn)(int layer_idx, int bucket, const char* blob_dir);

/// Describes an ANE kernel: MIL generation, weight building, and I/O shape.
typedef struct {
    const char* name;           // Cache key name, e.g. "prefill_attn"
    OrionMILGenFn generate_mil; // MIL text generator
    OrionWDictFn build_wdict;   // Weight dictionary builder
    int n_inputs;               // Expected number of input IOSurfaces
    int n_outputs;              // Expected number of output IOSurfaces
} OrionKernel;

/// Compile-or-cache + eval a kernel in one call.
///
/// 1. Looks up program in cache using (kernel->name, layer_idx, wb)
/// 2. On miss: generate MIL, build wdict, compile, store in cache
/// 3. Eval the program with provided IOSurfaces
///
/// @param kernel     Kernel descriptor
/// @param layer_idx  Layer index (-1 for non-layer-specific)
/// @param bucket     Sequence length bucket
/// @param cfg        Model configuration (passed to generate_mil)
/// @param blob_dir   Weight blob directory (passed to build_wdict)
/// @param wb         Weights binding for cache key
/// @param inputs     Input IOSurface array
/// @param n_inputs   Number of inputs
/// @param outputs    Output IOSurface array (pre-allocated)
/// @param n_outputs  Number of outputs
/// @return true on success, false on compile or eval failure
bool orion_kernel_eval(const OrionKernel* kernel,
                       int layer_idx, int bucket,
                       const OrionModelConfig* cfg,
                       const char* blob_dir,
                       const OrionWeightsBinding* wb,
                       IOSurfaceRef* inputs, int n_inputs,
                       IOSurfaceRef* outputs, int n_outputs);

/// Compile-or-cache a kernel without evaluating.
/// Returns the cached/compiled program, or NULL on failure.
/// Caller must NOT release the returned program (cache owns it).
OrionProgram* orion_kernel_compile(const OrionKernel* kernel,
                                   int layer_idx, int bucket,
                                   const OrionModelConfig* cfg,
                                   const char* blob_dir,
                                   const OrionWeightsBinding* wb);

#endif // ORION_KERNEL_H
