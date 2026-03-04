#ifndef ORION_MODEL_REGISTRY_H
#define ORION_MODEL_REGISTRY_H

#import "ane_runtime.h"

/// Describes a supported model: config, buckets, default weight path.
typedef struct {
    const char* name;              // e.g. "gpt2_124m", "stories110m"
    OrionModelConfig config;
    const int* buckets;            // NULL if model doesn't use buckets
    int n_buckets;
    const char* default_weights_dir; // e.g. "model/blobs/gpt2_124m"
} OrionModelSpec;

/// Look up a model by name. Returns NULL for unknown names.
const OrionModelSpec* orion_model_lookup(const char* name);

/// Get total number of registered models.
int orion_model_count(void);

/// Get model at index (for iteration). Returns NULL if out of range.
const OrionModelSpec* orion_model_at(int index);

#endif // ORION_MODEL_REGISTRY_H
