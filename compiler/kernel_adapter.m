// compiler/kernel_adapter.m — T137: OrionKernel adapter for compiler frontends

#import "kernel_adapter.h"
#import "codegen.h"
#import "pipeline.h"
#import "validate.h"
#include <string.h>

// Registry for adapted kernels (maps name -> frontend function)
#define MAX_ADAPTED_KERNELS 32

static struct {
    const char* name;
    OrionFrontendFn frontend;
} s_registry[MAX_ADAPTED_KERNELS];
static int s_registry_count = 0;

// Direct API: generate MIL from a frontend function.
// Builds graph -> validates -> optimizes -> codegen.
NSString* orion_kernel_adapter_generate_mil(OrionFrontendFn frontend,
                                             int layer_idx, int bucket,
                                             const OrionModelConfig* cfg) {
    OrionGraph* g = frontend(layer_idx, bucket, cfg);
    if (!g) return nil;

    // Validate
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        NSLog(@"[kernel_adapter] validation failed: %s", vr.message);
        orion_graph_free(g);
        return nil;
    }

    // Optimize
    orion_pipeline_optimize(g);

    // Codegen
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

// Direct API: generate MIL from a 2-arg frontend (no bucket param).
NSString* orion_kernel_adapter_generate_mil_2arg(OrionFrontend2Fn frontend,
                                                  int layer_idx,
                                                  const OrionModelConfig* cfg) {
    OrionGraph* g = frontend(layer_idx, cfg);
    if (!g) return nil;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        NSLog(@"[kernel_adapter] validation failed: %s", vr.message);
        orion_graph_free(g);
        return nil;
    }

    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

// Registry-based generate_mil for use with OrionKernel.
// Looks up the frontend by kernel name and calls adapter_generate_mil.
static NSString* registry_gen_mil(int layer_idx __attribute__((unused)),
                                  int bucket __attribute__((unused)),
                                  const OrionModelConfig* cfg __attribute__((unused))) {
    // Walk the call stack isn't feasible — use thread-local context
    // This is set by orion_kernel_eval before calling generate_mil
    return nil; // Fallback — callers should use adapter_generate_mil directly
}

OrionKernel orion_kernel_from_frontend(const char* name,
                                        OrionFrontendFn frontend,
                                        OrionWDictFn wdict,
                                        int n_inputs, int n_outputs) {
    if (s_registry_count < MAX_ADAPTED_KERNELS) {
        s_registry[s_registry_count].name = name;
        s_registry[s_registry_count].frontend = frontend;
        s_registry_count++;
    }

    OrionKernel k;
    k.name = name;
    k.generate_mil = registry_gen_mil;
    k.build_wdict = wdict;
    k.n_inputs = n_inputs;
    k.n_outputs = n_outputs;
    return k;
}
