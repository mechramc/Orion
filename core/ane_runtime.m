#import "ane_runtime.h"

// TODO(M0): Implement using private ANE APIs
// - _ANEClient for device access
// - _ANECompiler for MIL compilation
// - _ANEInMemoryModelDescriptor for program management

#pragma mark - OrionProgram

struct OrionProgram {
    id aneModel;       // _ANEInMemoryModelDescriptor
    id compiledResult; // Compiled program object
    char tag[256];
};

#pragma mark - Compile

OrionProgram* orion_compile_mil(
    const char* mil_text,
    const void* const* weight_blobs,
    const size_t* weight_sizes,
    int num_blobs,
    const OrionModelConfig* cfg,
    const char* program_tag
) {
    // TODO(M0): Implement MIL compilation via _ANECompiler
    // 1. Create _ANEInMemoryModelDescriptor from MIL text
    // 2. Attach weight blobs as BLOBFILE constants
    // 3. Compile to ANE program
    // 4. Return wrapped handle
    return NULL;
}

#pragma mark - Eval

void orion_eval(
    OrionProgram* prog,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
) {
    // TODO(M0): Implement ANE evaluation
    // 1. Map IOSurface inputs to ANE tensor descriptors
    // 2. Execute compiled program
    // 3. Results written to output IOSurfaces
}

#pragma mark - Release

void orion_release_program(OrionProgram* prog) {
    if (!prog) return;
    // TODO(M0): Explicit release of ANE objects to prevent compiler exhaustion
    // prog->aneModel = nil;
    // prog->compiledResult = nil;
    free(prog);
}
