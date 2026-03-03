#import "kv_cache.h"
#import <stdlib.h>
#import <string.h>

// TODO(M1): Implement KV cache management

OrionKVCache* orion_kv_cache_create(const OrionModelConfig* cfg) {
    OrionKVCache* cache = calloc(1, sizeof(OrionKVCache));
    cache->n_layer = cfg->n_layer;
    cache->n_head = cfg->n_head;
    cache->head_dim = cfg->head_dim;
    cache->max_seq = cfg->max_seq;
    cache->current_len = 0;

    size_t size = (size_t)cfg->n_layer * cfg->n_head * cfg->max_seq * cfg->head_dim * sizeof(float);
    cache->k_cache = calloc(1, size);
    cache->v_cache = calloc(1, size);

    return cache;
}

void orion_kv_cache_store_prefill(OrionKVCache* cache, int layer,
                                   const float* k, const float* v, int seq_len) {
    // TODO(M1): Store prefill K,V into cache for given layer
}

void orion_kv_cache_append(OrionKVCache* cache, int layer,
                            const float* k, const float* v) {
    // TODO(M1): Append single position K,V and increment current_len
}

void orion_kv_cache_free(OrionKVCache* cache) {
    if (!cache) return;
    free(cache->k_cache);
    free(cache->v_cache);
    free(cache);
}
