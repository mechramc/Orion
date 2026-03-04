#import "kernel.h"
#import <stdio.h>

OrionProgram* orion_kernel_compile(const OrionKernel* kernel,
                                   int layer_idx, int bucket,
                                   const OrionModelConfig* cfg,
                                   const char* blob_dir,
                                   const OrionWeightsBinding* wb) {
    // 1. Cache lookup
    OrionProgram* prog = orion_cache_lookup(kernel->name, layer_idx, wb);
    if (prog) return prog;

    // 2. Cache miss — generate MIL, build weight dict, compile
    NSString* mil = kernel->generate_mil(layer_idx, bucket, cfg);
    if (!mil) {
        fprintf(stderr, "orion_kernel_compile: MIL generation failed for %s L%d\n",
                kernel->name, layer_idx);
        return NULL;
    }

    NSDictionary* wdict = kernel->build_wdict(layer_idx, bucket, blob_dir);

    char tag[64];
    snprintf(tag, sizeof(tag), "%s_L%d", kernel->name, layer_idx);
    prog = orion_compile_mil(mil.UTF8String, wdict, tag);
    if (!prog) {
        fprintf(stderr, "orion_kernel_compile: compile failed for %s L%d\n",
                kernel->name, layer_idx);
        return NULL;
    }

    // 3. Store in cache (cache takes ownership)
    orion_cache_store(kernel->name, layer_idx, wb, prog);
    return prog;
}

bool orion_kernel_eval(const OrionKernel* kernel,
                       int layer_idx, int bucket,
                       const OrionModelConfig* cfg,
                       const char* blob_dir,
                       const OrionWeightsBinding* wb,
                       IOSurfaceRef* inputs, int n_inputs,
                       IOSurfaceRef* outputs, int n_outputs) {
    OrionProgram* prog = orion_kernel_compile(kernel, layer_idx, bucket,
                                              cfg, blob_dir, wb);
    if (!prog) return false;

    return orion_eval(prog, inputs, n_inputs, outputs, n_outputs);
}
