#ifndef ORION_GPT2_124M_H
#define ORION_GPT2_124M_H

#import "../../core/ane_runtime.h"

// GPT-2 124M model configuration
// Architecture: 12-layer transformer decoder with GELU activations
// Vocabulary: 50257 (BPE), context: 1024 tokens

static const OrionModelConfig kGPT2_124M = {
    .n_layer   = 12,
    .n_head    = 12,
    .d_model   = 768,
    .head_dim  = 64,    // 768 / 12
    .hidden_dim = 3072, // 4 * 768
    .vocab     = 50257,
    .max_seq   = 1024,
};

// Inference bucket sizes for ANE prefill
static const int kGPT2Buckets[] = {32, 64, 128, 256, 512, 1024};
static const int kGPT2NumBuckets = 6;

#endif // ORION_GPT2_124M_H
