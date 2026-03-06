// lora.h — LoRA-fused compiler frontends (T156, T158)
//
// LoRA: Y = X@W_base + alpha * ((X@A) @ B)
// W_base: BLOBFILE constant (baked into compiled program)
// A, B: IOSurface inputs (hot-swappable without recompilation)
//
// ANE layout: [1, C, 1, S]
// Conv1x1 for base weight (fast, baked), matmul for LoRA path (dynamic).

#ifndef ORION_FRONTEND_LORA_H
#define ORION_FRONTEND_LORA_H

#include "../graph.h"
#include "../model_config.h"

/// LoRA configuration.
typedef struct {
    int rank;           // LoRA rank (e.g. 8, 16, 32)
    float alpha;        // LoRA scaling factor (alpha / rank)
    bool apply_q;       // Apply LoRA to Q projection
    bool apply_k;       // Apply LoRA to K projection
    bool apply_v;       // Apply LoRA to V projection
    bool apply_o;       // Apply LoRA to O (output) projection
} OrionLoRAConfig;

/// Default LoRA config: rank=16, alpha=16.0, Q+V projections.
static inline OrionLoRAConfig orion_lora_config_default(void) {
    return (OrionLoRAConfig){
        .rank = 16,
        .alpha = 16.0f,
        .apply_q = true,
        .apply_k = false,
        .apply_v = true,
        .apply_o = false,
    };
}

/// LoRA-fused linear frontend.
/// Inputs: x [1,in_dim,1,seq], lora_A [1,in_dim,1,rank], lora_B [1,out_dim,1,rank]
/// Weights: W_base BLOBFILE [out_dim, in_dim, 1, 1]
/// Output: y [1,out_dim,1,seq] = conv(x, W_base) + (alpha/rank) * matmul(matmul(x, A), B)
///
/// Graph inputs are named: "x", "lora_A", "lora_B"
/// Graph output is named: "y"
OrionGraph* orion_frontend_lora_linear(int in_dim, int out_dim, int seq,
                                        int rank, float alpha,
                                        const char* weight_path);

/// LoRA-fused attention frontend (full Q/K/V/O projections).
/// Applies LoRA to selected projections based on config.
/// Input: x [1,d,1,s]
/// LoRA IOSurface inputs for each enabled projection (lora_{q,k,v,o}_A, lora_{q,k,v,o}_B)
/// Weights: rms1, wq, wk, wv, wo, causal_mask BLOBFILEs
/// Outputs: same as fwd_attn (wo_out, q_out, k_out, v_out, attn_out, rms1_out)
OrionGraph* orion_frontend_lora_attn(int layer, const OrionModelConfig* cfg,
                                      const OrionLoRAConfig* lora);

#endif // ORION_FRONTEND_LORA_H
