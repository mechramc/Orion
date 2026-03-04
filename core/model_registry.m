#import "model_registry.h"
#import <string.h>

// Bucket sizes for GPT-2 prefill
static const int kGPT2RegistryBuckets[] = {32, 64, 128, 256, 512, 1024};

static const OrionModelSpec kModels[] = {
    {
        .name = "gpt2_124m",
        .config = {
            .n_layer   = 12,
            .n_head    = 12,
            .d_model   = 768,
            .head_dim  = 64,
            .hidden_dim = 3072,
            .vocab     = 50257,
            .max_seq   = 1024,
        },
        .buckets = kGPT2RegistryBuckets,
        .n_buckets = 6,
        .default_weights_dir = "model/blobs/gpt2_124m",
    },
    {
        .name = "stories110m",
        .config = {
            .n_layer   = 12,
            .n_head    = 12,
            .d_model   = 768,
            .head_dim  = 64,
            .hidden_dim = 2048,
            .vocab     = 32000,
            .max_seq   = 256,
        },
        .buckets = NULL,
        .n_buckets = 0,
        .default_weights_dir = "model/blobs/stories110m",
    },
};

static const int kModelCount = sizeof(kModels) / sizeof(kModels[0]);

const OrionModelSpec* orion_model_lookup(const char* name) {
    if (!name) return NULL;
    for (int i = 0; i < kModelCount; i++) {
        if (strcmp(kModels[i].name, name) == 0) {
            return &kModels[i];
        }
    }
    return NULL;
}

int orion_model_count(void) {
    return kModelCount;
}

const OrionModelSpec* orion_model_at(int index) {
    if (index < 0 || index >= kModelCount) return NULL;
    return &kModels[index];
}
