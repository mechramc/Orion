// compiler/frontends/gpt2_final.h — T141: GPT-2 final LayerNorm frontend
#ifndef ORION_FRONTEND_GPT2_FINAL_H
#define ORION_FRONTEND_GPT2_FINAL_H

#include "../graph.h"
#include "../model_config.h"

// Build final LayerNorm graph.
// Equivalent to orion_milgen_gpt2_final_ln.
// Input:  fp32 [1, d_model, 1, bucket]
// Output: fp32 [1, d_model, 1, bucket] "hidden"
OrionGraph* orion_frontend_gpt2_final_ln(int bucket, const OrionModelConfig* cfg);

#endif // ORION_FRONTEND_GPT2_FINAL_H
