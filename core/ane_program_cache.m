#import "ane_program_cache.h"
#import <Foundation/Foundation.h>

// TODO(M4): Implement program cache for weight swapping
// Cache keyed by: (model_config_hash, bucket, weights_id, kernel_type, layer_idx)
// Must enforce safe release of ObjC objects to avoid compiler exhaustion

static NSMutableDictionary* _programCache = nil;

static NSString* _cacheKey(const char* kernel_name, int layer_idx,
                           const OrionWeightsBinding* wb) {
    return [NSString stringWithFormat:@"%s:%d:%s:%d",
            kernel_name, layer_idx, wb->weights_id, wb->bucket];
}

OrionProgram* orion_get_or_compile(
    const char* kernel_name,
    int layer_idx,
    const OrionWeightsBinding* wb,
    const OrionModelConfig* cfg
) {
    // TODO(M4): Implement cache lookup + compile-on-miss
    // 1. Build cache key
    // 2. Check cache
    // 3. If miss: generate MIL, load weights, compile, store
    // 4. Return program
    return NULL;
}

void orion_cache_evict(const char* weights_id) {
    // TODO(M4): Remove all entries matching weights_id
}

void orion_cache_clear(void) {
    // TODO(M4): Release all programs and clear cache
}

int orion_cache_size(void) {
    return _programCache ? (int)[_programCache count] : 0;
}
