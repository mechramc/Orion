#ifndef ORION_ANE_RUNTIME_H
#define ORION_ANE_RUNTIME_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

/// Model configuration shared across all Orion components.
#ifndef ORION_MODEL_CONFIG_DEFINED
#define ORION_MODEL_CONFIG_DEFINED
typedef struct {
    int n_layer;
    int n_head;
    int d_model;
    int head_dim;
    int hidden_dim;
    int vocab;
    int max_seq;
    int n_kv_head;
} OrionModelConfig;
#endif

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

/// Return the temp directory used by a compiled ANE program, or nil.
/// The returned NSString is owned by the program and remains valid until
/// orion_release_program() is called for that program.
NSString* orion_program_tmp_dir(OrionProgram* prog);

/// Create a new ANE program with patched weights, reusing compiled artifacts
/// from a donor program. Skips compilation entirely — only loads.
///
/// The ANE compiler's "data" file IS the BLOBFILE. By creating a new model
/// identity and copying the compiled net.plist + new BLOBFILE, we can load
/// new weights without compiling. This produces output identical to a fresh
/// compile (verified: max diff = 0.0).
///
/// @param donor        A previously compiled OrionProgram (provides net.plist)
/// @param mil_text     Same MIL text used to compile the donor
/// @param weight_dict  New weight dict (same keys, new data)
/// @param program_tag  Tag for debugging (may be NULL)
/// @return New program handle with patched weights, or NULL on failure.
///         Caller must release with orion_release_program().
///         Does NOT increment orion_compile_count().
OrionProgram* orion_program_patch_weights(
    OrionProgram* donor,
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag
);

/// Reload an existing ANE program with new weights in-place.
/// Unloads the model, updates weight files on disk, then reloads.
/// Much faster than orion_program_patch_weights since it skips
/// descriptor/model creation and MIL parsing.
///
/// @param prog         The program to reload (must be loaded)
/// @param weight_dict  New weight dict (same keys, new data)
/// @return true on success, false on failure. On failure, program is
///         in an unloaded state and should be released.
bool orion_program_reload_weights(
    OrionProgram* prog,
    NSDictionary* weight_dict
);

/// Export a loaded program's compiled runtime artifacts to a persistent directory.
/// The destination is replaced atomically enough for single-writer Silver cache usage.
/// @param prog          Loaded OrionProgram
/// @param artifact_dir  Persistent directory to populate
/// @return true on success.
bool orion_program_export_artifacts(
    OrionProgram* prog,
    const char* artifact_dir
);

/// Load an ANE program from previously exported runtime artifacts without compiling.
/// @param mil_text       Same MIL text used for the original compile
/// @param weight_dict    Same weight dict key structure used for the original compile
/// @param artifact_dir   Directory containing exported runtime artifacts
/// @param program_tag    Tag for debugging (may be NULL)
/// @return Loaded program handle, or NULL on failure.
OrionProgram* orion_program_load_artifacts(
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* artifact_dir,
    const char* program_tag
);

#endif // ORION_ANE_RUNTIME_H
