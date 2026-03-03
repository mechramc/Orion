#ifndef ORION_GPT2_PREFILL_FFN_H
#define ORION_GPT2_PREFILL_FFN_H

#import "../../core/mil_builder.h"
#import "../../core/ane_runtime.h"

// MIL generator for GPT-2 FFN prefill kernel.
// Generates a single MIL program per layer that computes:
//   LayerNorm → Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model)

// TODO(M2): Implement MIL generation for GPT-2 FFN prefill
NSString* orion_milgen_gpt2_prefill_ffn(int layer_idx, int seq_len,
                                         const OrionModelConfig* cfg);

#endif // ORION_GPT2_PREFILL_FFN_H
