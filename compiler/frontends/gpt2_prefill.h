// compiler/frontends/gpt2_prefill.h — T134: GPT-2 prefill frontend
#ifndef ORION_FRONTEND_GPT2_PREFILL_H
#define ORION_FRONTEND_GPT2_PREFILL_H

#include "../graph.h"
#include "../model_config.h"

// Build prefill attention graph for one layer.
// Equivalent to orion_milgen_gpt2_prefill_attn.
// Outputs: hidden (fp32), k_cache (fp32), v_cache (fp32)
OrionGraph* orion_frontend_gpt2_prefill_attn(int layer, int bucket, const OrionModelConfig* cfg);

// Build prefill FFN graph for one layer.
// Equivalent to orion_milgen_gpt2_prefill_ffn.
// Output: hidden (fp32)
OrionGraph* orion_frontend_gpt2_prefill_ffn(int layer, int bucket, const OrionModelConfig* cfg);

#endif // ORION_FRONTEND_GPT2_PREFILL_H
