#import "kv_cache.h"
#import <stdlib.h>
#import <string.h>

// T037-T038: KV Cache Management
//
// Layout: [n_layer, n_head, max_seq, head_dim]
// For layer L, head H, position S:
//   offset = L * (n_head * max_seq * head_dim) + H * (max_seq * head_dim) + S * head_dim

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
    // k, v are [seq_len, n_head * head_dim] (heads interleaved)
    // Rearrange to [n_head, seq_len, head_dim] in cache
    int nh = cache->n_head;
    int hd = cache->head_dim;
    int ms = cache->max_seq;
    size_t layer_stride = (size_t)nh * ms * hd;

    for (int h = 0; h < nh; h++) {
        for (int s = 0; s < seq_len; s++) {
            size_t cache_off = layer * layer_stride + h * ms * hd + s * hd;
            size_t src_off = s * (nh * hd) + h * hd;
            memcpy(cache->k_cache + cache_off, k + src_off, hd * sizeof(float));
            memcpy(cache->v_cache + cache_off, v + src_off, hd * sizeof(float));
        }
    }
}

void orion_kv_cache_append(OrionKVCache* cache, int layer,
                            const float* k, const float* v) {
    // k, v are [n_head * head_dim] for a single position
    // Write to position current_len in each head
    int nh = cache->n_head;
    int hd = cache->head_dim;
    int ms = cache->max_seq;
    int pos = cache->current_len;
    size_t layer_stride = (size_t)nh * ms * hd;

    for (int h = 0; h < nh; h++) {
        size_t cache_off = layer * layer_stride + h * ms * hd + pos * hd;
        size_t src_off = h * hd;
        memcpy(cache->k_cache + cache_off, k + src_off, hd * sizeof(float));
        memcpy(cache->v_cache + cache_off, v + src_off, hd * sizeof(float));
    }
}

void orion_kv_cache_free(OrionKVCache* cache) {
    if (!cache) return;
    free(cache->k_cache);
    free(cache->v_cache);
    free(cache);
}
