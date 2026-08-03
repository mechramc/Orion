// compiler/model_config.h — Forward declaration of OrionModelConfig for pure C code.
// Avoids pulling in ObjC headers from ane_runtime.h.
#ifndef ORION_COMPILER_MODEL_CONFIG_H
#define ORION_COMPILER_MODEL_CONFIG_H

// Mirror of OrionModelConfig from core/ane_runtime.h
// Keep in sync manually. Both are POD structs with identical layout.
#ifndef ORION_MODEL_CONFIG_DEFINED
#define ORION_MODEL_CONFIG_DEFINED
typedef struct {
    int n_layer;
    int n_head;
    int n_kv_head;
    int d_model;
    int head_dim;
    int hidden_dim;
    int vocab;
    int max_seq;
} OrionModelConfig;
#endif

#endif // ORION_COMPILER_MODEL_CONFIG_H
