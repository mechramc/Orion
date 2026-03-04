#ifndef ORION_GPT2_PREFILL_FFN_H
#define ORION_GPT2_PREFILL_FFN_H

#import "../../core/mil_builder.h"
#import "../../core/ane_runtime.h"

// T048: MIL generator for GPT-2 FFN prefill kernel.
//
// Generates a MIL program for one transformer layer's FFN block:
//   Input: hidden state fp32 [1, d_model, 1, seq]
//   Ops:   Cast→fp16 → LN2 → Linear(d→4d) → GELU → Linear(4d→d) → Residual → Cast→fp32
//   Output: hidden state fp32 [1, d_model, 1, seq]
//
// Weight blob paths referenced in generated MIL:
//   @model_path/layer{i}/ln2_g.bin, ln2_b.bin    — LayerNorm gamma, beta
//   @model_path/layer{i}/wfc.bin, bfc.bin        — FC up weight [hidden,d,1,1], bias
//   @model_path/layer{i}/wproj.bin, bproj.bin    — FC down weight [d,hidden,1,1], bias

/// Generate complete MIL program for GPT-2 FFN prefill (one layer).
/// @param layer_idx Layer index (0-11 for GPT-2 124M)
/// @param seq_len   Bucket sequence length
/// @param cfg       Model configuration
/// @return Complete MIL program text (1 output: hidden)
NSString* orion_milgen_gpt2_prefill_ffn(int layer_idx, int seq_len,
                                         const OrionModelConfig* cfg);

#endif // ORION_GPT2_PREFILL_FFN_H
