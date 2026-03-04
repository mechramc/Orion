#import "runtime.h"
#import <stdlib.h>

#define ORION_DEFAULT_COMPILE_BUDGET 119

OrionRuntime* orion_runtime_create(void) {
    OrionRuntime* rt = (OrionRuntime*)calloc(1, sizeof(OrionRuntime));
    rt->compile_budget = ORION_DEFAULT_COMPILE_BUDGET;
    return rt;
}

bool orion_runtime_init_ane(OrionRuntime* rt) {
    if (rt->ane_ready) return true;
    rt->ane_ready = orion_ane_init();
    return rt->ane_ready;
}

bool orion_runtime_eval_kernel(OrionRuntime* rt,
                               const OrionKernel* kernel,
                               int layer_idx, int bucket,
                               const OrionModelConfig* cfg,
                               const char* blob_dir,
                               const OrionWeightsBinding* wb,
                               IOSurfaceRef* inputs, int n_inputs,
                               IOSurfaceRef* outputs, int n_outputs) {
    if (!rt->ane_ready) return false;

    // Track compiles: check before and after to detect cache misses
    int before = orion_compile_count();

    bool ok = orion_kernel_eval(kernel, layer_idx, bucket, cfg, blob_dir, wb,
                                inputs, n_inputs, outputs, n_outputs);

    int after = orion_compile_count();
    rt->total_compiles += (after - before);

    return ok;
}

bool orion_runtime_needs_restart(const OrionRuntime* rt, int* remaining) {
    int actual = orion_compile_count();
    int left = rt->compile_budget - actual;
    if (remaining) *remaining = left > 0 ? left : 0;
    return actual >= rt->compile_budget;
}

void orion_runtime_clear_cache(OrionRuntime* rt) {
    orion_cache_clear();
    rt->total_compiles = 0;
}

void orion_runtime_free(OrionRuntime* rt) {
    if (rt) free(rt);
}
