// lora.c — LoRA-fused compiler frontends (T156, T158)
//
// LoRA computation on ANE:
//   Base: conv1x1(x, W_base) — fast, weight baked into program
//   LoRA: matmul path with IOSurface inputs A, B — hot-swappable
//
// ANE matmul requires [1,1,M,K] layout; ANE native is [1,C,1,S].
// We transpose [1,C,1,S] → [1,1,S,C] before matmul, then back after.
// Transposes are free on ANE (metadata reinterpretation only).

#include "lora.h"
#include "../builder.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Helper: build LoRA path for a single projection.
// x: input node [1, in_dim, 1, seq]
// lora_A: input node [1, in_dim, 1, rank]  (IOSurface)
// lora_B: input node [1, out_dim, 1, rank]  (IOSurface)
// Returns output node [1, out_dim, 1, seq]
static int build_lora_path(OrionGraph* g, int x, int lora_A, int lora_B,
                           int in_dim, int out_dim, int seq, int rank,
                           float alpha_over_rank, const char* prefix) {
    char name[128];

    // Permutation: [1,C,1,S] → [1,1,S,C] for matmul
    int pm_vals[4] = {0, 2, 3, 1};
    snprintf(name, sizeof(name), "%s_pm", prefix);
    int bsh[4] = {4, 1, 1, 1};
    int pm = orion_gb_const_int32(g, name, bsh, pm_vals, 4);

    // Permutation back: [1,1,S,C] → [1,C,1,S]
    int pm_back_vals[4] = {0, 3, 1, 2};
    snprintf(name, sizeof(name), "%s_pmb", prefix);
    int pm_back = orion_gb_const_int32(g, name, bsh, pm_back_vals, 4);

    // Transpose x: [1, in_dim, 1, seq] → [1, 1, seq, in_dim]
    int xt_shape[4] = {1, 1, seq, in_dim};
    snprintf(name, sizeof(name), "%s_xt", prefix);
    int xt = orion_gb_transpose(g, x, pm, name, pm_vals, xt_shape);

    // Transpose A: [1, in_dim, 1, rank] → [1, 1, rank, in_dim]
    int at_shape[4] = {1, 1, rank, in_dim};
    snprintf(name, sizeof(name), "%s_at", prefix);
    int at = orion_gb_transpose(g, lora_A, pm, name, pm_vals, at_shape);

    // matmul(xt, at, tx=false, ty=true): [1,1,seq,in_dim] × [1,1,in_dim,rank] → [1,1,seq,rank]
    // x matrix: [seq, in_dim], y with ty=true: [in_dim, rank] → [seq, rank]
    int h_shape[4] = {1, 1, seq, rank};
    snprintf(name, sizeof(name), "%s_h", prefix);
    int h = orion_gb_matmul(g, xt, at, false, true, name, h_shape);

    // Transpose B: [1, out_dim, 1, rank] → [1, 1, rank, out_dim]
    int bt_shape[4] = {1, 1, rank, out_dim};
    snprintf(name, sizeof(name), "%s_bt", prefix);
    int bt = orion_gb_transpose(g, lora_B, pm, name, pm_vals, bt_shape);

    // matmul(h, bt, tx=false, ty=false): [1,1,seq,rank] × [1,1,rank,out_dim]
    // → [seq,rank] × [rank,out_dim] = [seq,out_dim]
    // bt is already [1,1,rank,out_dim] from transpose, so ty=false
    int lora_shape[4] = {1, 1, seq, out_dim};
    snprintf(name, sizeof(name), "%s_mm", prefix);
    int lora_mm = orion_gb_matmul(g, h, bt, false, false, name, lora_shape);

    // Transpose back: [1, 1, seq, out_dim] → [1, out_dim, 1, seq]
    int lora_out_shape[4] = {1, out_dim, 1, seq};
    snprintf(name, sizeof(name), "%s_lo", prefix);
    int lora_out = orion_gb_transpose(g, lora_mm, pm_back, name, pm_back_vals, lora_out_shape);

    // Scale by alpha/rank
    snprintf(name, sizeof(name), "%s_sc", prefix);
    int scale = orion_gb_const_scalar(g, name, ORION_DTYPE_FP16, alpha_over_rank);
    snprintf(name, sizeof(name), "%s_scaled", prefix);
    int scaled = orion_gb_mul(g, lora_out, scale, name);

    return scaled;
}

OrionGraph* orion_frontend_lora_linear(int in_dim, int out_dim, int seq,
                                        int rank, float alpha,
                                        const char* weight_path) {
    OrionGraph* g = orion_graph_create();

    // Input x: [1, in_dim, 1, seq]
    int x_shape[4] = {1, in_dim, 1, seq};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, x_shape);

    // LoRA adapter inputs (IOSurface — hot-swappable)
    int a_shape[4] = {1, in_dim, 1, rank};
    int lora_A = orion_gb_input(g, "lora_A", ORION_DTYPE_FP16, a_shape);

    int b_shape[4] = {1, out_dim, 1, rank};
    int lora_B = orion_gb_input(g, "lora_B", ORION_DTYPE_FP16, b_shape);

    // Base weight: conv1x1 (baked into program)
    int base = orion_gb_linear(g, x, "base", in_dim, out_dim, seq, weight_path, NULL);

    // LoRA path: matmul with IOSurface inputs
    float alpha_over_rank = alpha / (float)rank;
    int lora = build_lora_path(g, x, lora_A, lora_B,
                               in_dim, out_dim, seq, rank,
                               alpha_over_rank, "lr");

    // Fused output: base + scaled_lora
    int y = orion_gb_add(g, base, lora, "y");
    orion_gb_output(g, y, "y");

    return g;
}

// Helper: add LoRA-fused linear for a single projection within attention.
// Returns the output node [1, out_dim, 1, seq].
// If lora_A/lora_B are -1 (no LoRA), just does base linear.
static int lora_linear(OrionGraph* g, int x, const char* proj_name,
                       int in_dim, int out_dim, int seq,
                       const char* weight_path,
                       int lora_A, int lora_B, int rank, float alpha) {
    char name[128];
    snprintf(name, sizeof(name), "%s", proj_name);
    int base = orion_gb_linear(g, x, proj_name, in_dim, out_dim, seq, weight_path, NULL);

    if (lora_A < 0 || lora_B < 0) return base;

    float alpha_over_rank = alpha / (float)rank;
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "lr_%s", proj_name);
    int lora = build_lora_path(g, x, lora_A, lora_B,
                               in_dim, out_dim, seq, rank,
                               alpha_over_rank, prefix);

    snprintf(name, sizeof(name), "%s_fused", proj_name);
    return orion_gb_add(g, base, lora, name);
}

OrionGraph* orion_frontend_lora_attn(int layer, const OrionModelConfig* cfg,
                                      const OrionLoRAConfig* lora) {
    int d = cfg->d_model, s = cfg->max_seq;
    int nh = cfg->n_head, hd = cfg->head_dim;
    int rank = lora->rank;
    float alpha = lora->alpha;

    OrionGraph* g = orion_graph_create();

    // Input x: [1, d, 1, s]
    int x_shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, x_shape);

    // LoRA adapter IOSurface inputs for enabled projections
    int a_shape[4] = {1, d, 1, rank};
    int b_shape[4] = {1, d, 1, rank};

    int qa = -1, qb = -1, ka = -1, kb = -1;
    int va = -1, vb = -1, oa = -1, ob = -1;

    if (lora->apply_q) {
        qa = orion_gb_input(g, "lora_q_A", ORION_DTYPE_FP16, a_shape);
        qb = orion_gb_input(g, "lora_q_B", ORION_DTYPE_FP16, b_shape);
    }
    if (lora->apply_k) {
        ka = orion_gb_input(g, "lora_k_A", ORION_DTYPE_FP16, a_shape);
        kb = orion_gb_input(g, "lora_k_B", ORION_DTYPE_FP16, b_shape);
    }
    if (lora->apply_v) {
        va = orion_gb_input(g, "lora_v_A", ORION_DTYPE_FP16, a_shape);
        vb = orion_gb_input(g, "lora_v_B", ORION_DTYPE_FP16, b_shape);
    }
    if (lora->apply_o) {
        oa = orion_gb_input(g, "lora_o_A", ORION_DTYPE_FP16, a_shape);
        ob = orion_gb_input(g, "lora_o_B", ORION_DTYPE_FP16, b_shape);
    }

    // RMSNorm
    char path[256];
    snprintf(path, sizeof(path), "@model_path/layer%d/rms1.bin", layer);
    int rms_shape[4] = {1, d, 1, 1};
    int rms_w = orion_gb_const_weight(g, "rms1_w", ORION_DTYPE_FP16, rms_shape, path, 64);
    int rms1 = orion_gb_rmsnorm(g, x, rms_w, 1e-5f, "rms1", d, s);

    // Q, K, V projections (with optional LoRA)
    snprintf(path, sizeof(path), "@model_path/layer%d/wq.bin", layer);
    int q = lora_linear(g, rms1, "wq", d, d, s, path, qa, qb, rank, alpha);

    snprintf(path, sizeof(path), "@model_path/layer%d/wk.bin", layer);
    int k = lora_linear(g, rms1, "wk", d, d, s, path, ka, kb, rank, alpha);

    snprintf(path, sizeof(path), "@model_path/layer%d/wv.bin", layer);
    int v = lora_linear(g, rms1, "wv", d, d, s, path, va, vb, rank, alpha);

    // Attention: reshape to multi-head, compute scores, apply mask, softmax, weighted sum
    int bsh[4] = {4, 1, 1, 1};
    int rsh_vals[4] = {1, nh, hd, s};
    int rsh4d = orion_gb_const_int32(g, "rsh4d", bsh, rsh_vals, 4);
    int rsh_out[4] = {1, nh, hd, s};

    int pm_vals[4] = {0, 1, 3, 2};
    int pm = orion_gb_const_int32(g, "attn_pm", bsh, pm_vals, 4);
    int tr_out[4] = {1, nh, s, hd};

    int qr = orion_gb_reshape(g, q, rsh4d, "qr", rsh_out);
    int q4 = orion_gb_transpose(g, qr, pm, "q4", pm_vals, tr_out);

    int kr = orion_gb_reshape(g, k, rsh4d, "kr", rsh_out);
    int k4 = orion_gb_transpose(g, kr, pm, "k4", pm_vals, tr_out);

    int vr = orion_gb_reshape(g, v, rsh4d, "vr", rsh_out);
    int v4 = orion_gb_transpose(g, vr, pm, "v4", pm_vals, tr_out);

    // Scores = Q @ K^T / sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)hd);
    int sc_shape[4] = {1, nh, s, s};
    int scores = orion_gb_matmul(g, q4, k4, false, true, "scores", sc_shape);
    int sc_val = orion_gb_const_scalar(g, "sc_val", ORION_DTYPE_FP16, scale);
    int scores_sc = orion_gb_mul(g, scores, sc_val, "scores_sc");

    // Causal mask
    snprintf(path, sizeof(path), "@model_path/masks/causal_%d.bin", s);
    int mask_shape[4] = {1, 1, s, s};
    int cmask = orion_gb_const_weight(g, "cmask", ORION_DTYPE_FP16, mask_shape, path, 64);
    int masked = orion_gb_add(g, scores_sc, cmask, "masked");
    int probs = orion_gb_softmax(g, masked, -1, "probs");

    // Attention output: probs @ V
    int av_shape[4] = {1, nh, s, hd};
    int attn = orion_gb_matmul(g, probs, v4, false, false, "attn", av_shape);

    // Reshape back to [1, d, 1, s]
    int pm_back_vals[4] = {0, 1, 3, 2};
    int pm_back = orion_gb_const_int32(g, "attn_pmb", bsh, pm_back_vals, 4);
    int attn_t_shape[4] = {1, nh, hd, s};
    int attn_t = orion_gb_transpose(g, attn, pm_back, "attn_t", pm_back_vals, attn_t_shape);

    int rsh_flat_vals[4] = {1, d, 1, s};
    int rsh_flat = orion_gb_const_int32(g, "rsh_flat", bsh, rsh_flat_vals, 4);
    int attn_flat = orion_gb_reshape(g, attn_t, rsh_flat, "attn_flat", x_shape);

    // Output projection Wo (with optional LoRA)
    snprintf(path, sizeof(path), "@model_path/layer%d/wo.bin", layer);
    int wo = lora_linear(g, attn_flat, "wo", d, d, s, path, oa, ob, rank, alpha);

    // Outputs (same naming as fwd_attn for compatibility)
    orion_gb_output(g, wo, "wo_out");
    orion_gb_output(g, q, "q_out");
    orion_gb_output(g, k, "k_out");
    orion_gb_output(g, v, "v_out");
    orion_gb_output(g, attn_flat, "attn_out");
    orion_gb_output(g, rms1, "rms1_out");

    return g;
}
