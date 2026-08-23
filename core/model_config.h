// core/model_config.h — canonical pure-C model configuration shared by runtime and compiler
#ifndef ORION_MODEL_CONFIG_H
#define ORION_MODEL_CONFIG_H

#include <stddef.h>

/// Model configuration shared across all Orion components.
/// Keep this definition pure C so compiler frontends and Objective-C runtime
/// code consume the exact same byte layout.
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

// The ABI is intentionally fixed because OrionModelConfig crosses the runtime /
// compiler boundary. Fail at compile time if field order or padding changes.
_Static_assert(sizeof(OrionModelConfig) == 8 * sizeof(int),
               "OrionModelConfig ABI size changed");
_Static_assert(offsetof(OrionModelConfig, n_layer) == 0 * sizeof(int),
               "OrionModelConfig.n_layer layout changed");
_Static_assert(offsetof(OrionModelConfig, n_head) == 1 * sizeof(int),
               "OrionModelConfig.n_head layout changed");
_Static_assert(offsetof(OrionModelConfig, d_model) == 2 * sizeof(int),
               "OrionModelConfig.d_model layout changed");
_Static_assert(offsetof(OrionModelConfig, head_dim) == 3 * sizeof(int),
               "OrionModelConfig.head_dim layout changed");
_Static_assert(offsetof(OrionModelConfig, hidden_dim) == 4 * sizeof(int),
               "OrionModelConfig.hidden_dim layout changed");
_Static_assert(offsetof(OrionModelConfig, vocab) == 5 * sizeof(int),
               "OrionModelConfig.vocab layout changed");
_Static_assert(offsetof(OrionModelConfig, max_seq) == 6 * sizeof(int),
               "OrionModelConfig.max_seq layout changed");
_Static_assert(offsetof(OrionModelConfig, n_kv_head) == 7 * sizeof(int),
               "OrionModelConfig.n_kv_head layout changed");

#endif // ORION_MODEL_CONFIG_H
