// compiler/frontends/gpt2_decode.h — T135: GPT-2 decode frontend
#ifndef ORION_FRONTEND_GPT2_DECODE_H
#define ORION_FRONTEND_GPT2_DECODE_H

#include "../graph.h"
#include "../model_config.h"

#define ORION_GRAPH_DECODE_SEQ 16

// Build decode projection graph (LN1 -> QKV).
// Outputs: q32, k32, v32 (fp32, multi-output)
OrionGraph* orion_frontend_gpt2_decode_proj(int layer, const OrionModelConfig* cfg);

// Build decode FFN graph (LN2 -> FFN -> residual).
// Output: hidden (fp32)
OrionGraph* orion_frontend_gpt2_decode_ffn(int layer, const OrionModelConfig* cfg);

#endif // ORION_FRONTEND_GPT2_DECODE_H
