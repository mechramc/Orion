#ifndef ORION_QWEN35_08B_H
#define ORION_QWEN35_08B_H

#import "../../core/ane_runtime.h"

// Qwen3.5-0.8B text-only runtime configuration.
// These values are taken from the Hugging Face config.text_config dump and are
// used only for stage-1 CPU-only port scaffolding.
static const OrionModelConfig kQwen35_08B = {
    .n_layer    = 24,
    .n_head     = 8,
    .d_model    = 1024,
    .head_dim   = 256,
    .hidden_dim = 3584,
    .vocab      = 248320,
    .max_seq    = 262144,
    .n_kv_head  = 2,
};

#endif // ORION_QWEN35_08B_H
