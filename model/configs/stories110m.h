#ifndef ORION_STORIES110M_H
#define ORION_STORIES110M_H

#import "../../core/ane_runtime.h"

// Stories110M (Llama2-style) model configuration
// Matches ANEgpt stories_config.h
// Architecture: 12-layer transformer with RMSNorm, SwiGLU, RoPE
// Total params: ~109.53M (84.95M transformer + 24.58M embedding)

static const OrionModelConfig kStories110M = {
    .n_layer    = 12,
    .n_head     = 12,
    .d_model    = 768,
    .head_dim   = 64,    // 768 / 12
    .hidden_dim = 2048,  // SwiGLU hidden size
    .vocab      = 32000, // Llama2 BPE vocabulary
    .max_seq    = 256,
};

// Training constants (from ANEgpt)
#define STORIES_ACCUM_STEPS    10   // Gradient accumulation steps
#define STORIES_MAX_COMPILES   100  // ANE compile budget per process
#define STORIES_KERNELS_PER_LAYER 5
#define STORIES_TOTAL_WEIGHT_KERNELS (STORIES_KERNELS_PER_LAYER * 12) // = 60

// Checkpoint magic
#define ORION_CKPT_MAGIC   0x424C5A54  // "BLZT"
#define ORION_CKPT_VERSION 2

#endif // ORION_STORIES110M_H
