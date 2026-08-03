#ifndef ORION_QWEN35_9B_H
#define ORION_QWEN35_9B_H

#import "../../core/ane_runtime.h"

// Qwen3.5-9B text-only runtime configuration.
// Values are taken from the local export config dump and used for
// CPU-only / ANE prefill port scaffolding.
static const OrionModelConfig kQwen35_9B = {
    .n_layer    = 32,
    .n_head     = 16,
    .d_model    = 4096,
    .head_dim   = 256,
    .hidden_dim = 12288,
    .vocab      = 248320,
    .max_seq    = 262144,
    .n_kv_head  = 4,
};

#endif // ORION_QWEN35_9B_H
