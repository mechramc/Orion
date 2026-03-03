#ifndef ORION_ANE_RUNTIME_H
#define ORION_ANE_RUNTIME_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

/// Model configuration shared across all Orion components.
typedef struct {
    int n_layer;
    int n_head;
    int d_model;
    int head_dim;
    int hidden_dim;
    int vocab;
    int max_seq;
} OrionModelConfig;

/// Opaque handle to a compiled ANE program.
typedef struct OrionProgram OrionProgram;

/// Compile MIL text with embedded weight blobs into an ANE program.
/// @param mil_text       MIL program source text
/// @param weight_blobs   Array of pointers to weight data
/// @param weight_sizes   Array of sizes for each weight blob
/// @param num_blobs      Number of weight blobs
/// @param cfg            Model configuration
/// @param program_tag    Human-readable tag for debugging/caching
/// @return Compiled program handle, or NULL on failure
OrionProgram* orion_compile_mil(
    const char* mil_text,
    const void* const* weight_blobs,
    const size_t* weight_sizes,
    int num_blobs,
    const OrionModelConfig* cfg,
    const char* program_tag
);

/// Evaluate a compiled ANE program.
/// @param prog        Compiled program handle
/// @param inputs      Array of input IOSurface tensors
/// @param num_inputs  Number of inputs
/// @param outputs     Array of output IOSurface tensors (pre-allocated)
/// @param num_outputs Number of outputs
void orion_eval(
    OrionProgram* prog,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
);

/// Release a compiled program and its resources.
void orion_release_program(OrionProgram* prog);

#endif // ORION_ANE_RUNTIME_H
