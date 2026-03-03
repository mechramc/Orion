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

/// Initialize the ANE runtime (dlopen framework, resolve classes).
/// Must be called once before any other orion_* call.
/// @return true on success, false if ANE is unavailable.
bool orion_ane_init(void);

/// Compile MIL text with weight blobs into an ANE program.
/// @param mil_text       MIL program source text (null-terminated)
/// @param weight_dict    NSDictionary mapping BLOBFILE paths to @{@"offset":@N, @"data":NSData*},
///                       or nil/@{} for weight-free programs.
/// @param program_tag    Human-readable tag for debugging (may be NULL)
/// @return Compiled program handle, or NULL on failure.
OrionProgram* orion_compile_mil(
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag
);

/// Evaluate a compiled ANE program.
/// @param prog        Compiled program handle
/// @param inputs      Array of input IOSurface tensors
/// @param num_inputs  Number of inputs
/// @param outputs     Array of output IOSurface tensors (pre-allocated)
/// @param num_outputs Number of outputs
/// @return true on success.
bool orion_eval(
    OrionProgram* prog,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
);

/// Release a compiled program and its resources (unload from ANE, clean temp dir).
void orion_release_program(OrionProgram* prog);

/// Get the number of successful compilations this process has done.
/// Useful for tracking approach to the ~119 compile limit.
int orion_compile_count(void);

#endif // ORION_ANE_RUNTIME_H
