#ifndef ORION_GPT2_FINAL_MILGEN_H
#define ORION_GPT2_FINAL_MILGEN_H

#import "../../core/mil_builder.h"
#import "../../core/ane_runtime.h"

// T049: MIL generator for GPT-2 final LayerNorm.
//
// After the 12 transformer layers, GPT-2 applies a final LayerNorm before
// projecting to logits. The logits projection (hidden @ wte^T) is done on CPU
// because the wte blob (50257×768 = 73MB fp16) exceeds ANE SRAM.
//
// Input:  hidden state fp32 [1, d_model, 1, seq]
// Output: normalized hidden state fp32 [1, d_model, 1, seq]
//
// Weight blob paths:
//   @model_path/ln_f_g.bin  — final LN gamma [d_model]
//   @model_path/ln_f_b.bin  — final LN beta [d_model]

/// Generate MIL program for final LayerNorm.
/// @param seq_len Bucket sequence length
/// @param cfg     Model configuration
/// @return Complete MIL program text (1 output: hidden)
NSString* orion_milgen_gpt2_final_ln(int seq_len, const OrionModelConfig* cfg);

#endif // ORION_GPT2_FINAL_MILGEN_H
