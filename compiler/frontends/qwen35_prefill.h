// compiler/frontends/qwen35_prefill.h — Qwen3.5 ANE prefill frontend
#ifndef ORION_FRONTEND_QWEN35_PREFILL_H
#define ORION_FRONTEND_QWEN35_PREFILL_H

#include "../graph.h"
#include "../model_config.h"

// Build a single-output Q projection graph for one full-attention layer.
// Input: hidden [1, d_model, 1, seq] fp32
// Output: q_proj [1, 2*d_model, 1, seq] fp32
OrionGraph* orion_frontend_qwen35_prefill_q_proj(int layer, int bucket, const OrionModelConfig* cfg);

// Build a single-output Q projection graph that assumes the input is already RMSNorm'd.
// Input: normed hidden [1, d_model, 1, seq] fp32
// Output: q_proj [1, 2*d_model, 1, seq] fp32
OrionGraph* orion_frontend_qwen35_prefill_q_proj_linear_only(int layer, int bucket, const OrionModelConfig* cfg);

// Build a K/V projection graph for one full-attention layer.
// Input: hidden [1, d_model, 1, seq] fp32
// Outputs: k_proj [1, n_kv_head*head_dim, 1, seq] fp32
//          v_proj [1, n_kv_head*head_dim, 1, seq] fp32
OrionGraph* orion_frontend_qwen35_prefill_kv_proj(int layer, int bucket, const OrionModelConfig* cfg);

// Build a K/V projection graph that assumes the input is already RMSNorm'd.
// Input: normed hidden [1, d_model, 1, seq] fp32
// Outputs: k_proj [1, n_kv_head*head_dim, 1, seq] fp32
//          v_proj [1, n_kv_head*head_dim, 1, seq] fp32
OrionGraph* orion_frontend_qwen35_prefill_kv_proj_linear_only(int layer, int bucket, const OrionModelConfig* cfg);

// Build a post-attention FFN graph for one Qwen3.5 layer.
// Input: hidden [1, d_model, 1, seq] fp32
// Output: hidden [1, d_model, 1, seq] fp32
OrionGraph* orion_frontend_qwen35_prefill_ffn(int layer, int bucket, const OrionModelConfig* cfg);

#endif // ORION_FRONTEND_QWEN35_PREFILL_H
