// compiler/frontends/qwen35_prefill.c — Qwen3.5 ANE prefill frontend

#include "qwen35_prefill.h"
#include "../builder.h"
#include "../patterns.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static int qwen35_kv_dim(const OrionModelConfig* cfg) {
    int n_kv = cfg->n_kv_head > 0 ? cfg->n_kv_head : cfg->n_head;
    return n_kv * cfg->head_dim;
}

static int qwen35_use_matmul_v_proj(void) {
    const char *source = getenv("ORION_V_PROJ_SOURCE");
    return source && strcmp(source, "matmul") == 0;
}

static int qwen35_input_rmsnorm(OrionGraph* g, int layer, int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int s = bucket;
    int in_shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_pattern_cast_to_fp16(g, x, "x16", d, s);

    char path[256];
    int ln_shape[4] = {1, d, 1, 1};
    snprintf(path, sizeof(path), "@model_path/layer%d/input_layernorm.bin", layer);
    int rms_w = orion_gb_const_weight(g, "input_ln_w", ORION_DTYPE_FP16, ln_shape, path, 64);
    return orion_gb_rmsnorm(g, x16, rms_w, 1e-6f, "input_rms", d, s);
}

static int qwen35_normed_input(OrionGraph* g, int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int s = bucket;
    int in_shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    return orion_pattern_cast_to_fp16(g, x, "x16", d, s);
}

static int qwen35_linear_matmul(OrionGraph* g,
                                int input,
                                const char* prefix,
                                int in_dim,
                                int out_dim,
                                int seq,
                                const char* weight_path) {
    char buf[256];
    int perm_shape[4] = {4, 1, 1, 1};
    int perm_vals[4] = {0, 2, 3, 1};
    int perm_back_vals[4] = {0, 3, 1, 2};

    snprintf(buf, sizeof(buf), "%s_pm", prefix);
    int pm = orion_gb_const_int32(g, buf, perm_shape, perm_vals, 4);
    snprintf(buf, sizeof(buf), "%s_pmb", prefix);
    int pm_back = orion_gb_const_int32(g, buf, perm_shape, perm_back_vals, 4);

    int xt_shape[4] = {1, 1, seq, in_dim};
    snprintf(buf, sizeof(buf), "%s_xt", prefix);
    int xt = orion_gb_transpose(g, input, pm, buf, perm_vals, xt_shape);

    int w_shape[4] = {1, 1, out_dim, in_dim};
    snprintf(buf, sizeof(buf), "%s_W", prefix);
    int w = orion_gb_const_weight(g, buf, ORION_DTYPE_FP16, w_shape, weight_path, 64);

    int mm_shape[4] = {1, 1, seq, out_dim};
    snprintf(buf, sizeof(buf), "%s_mm", prefix);
    int mm = orion_gb_matmul(g, xt, w, false, true, buf, mm_shape);

    int out_shape[4] = {1, out_dim, 1, seq};
    snprintf(buf, sizeof(buf), "%s_out", prefix);
    return orion_gb_transpose(g, mm, pm_back, buf, perm_back_vals, out_shape);
}

static int qwen35_post_attn_rmsnorm(OrionGraph* g, int input, int layer, int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int s = bucket;
    char path[256];
    int ln_shape[4] = {1, d, 1, 1};
    snprintf(path, sizeof(path), "@model_path/layer%d/post_attention_layernorm.bin", layer);
    int rms_w = orion_gb_const_weight(g, "post_attn_ln_w", ORION_DTYPE_FP16, ln_shape, path, 64);
    return orion_gb_rmsnorm(g, input, rms_w, 1e-6f, "post_attn_rms", d, s);
}

OrionGraph* orion_frontend_qwen35_prefill_q_proj(int layer, int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int s = bucket;
    OrionGraph* g = orion_graph_create();
    int rms = qwen35_input_rmsnorm(g, layer, bucket, cfg);

    char path[256];
    snprintf(path, sizeof(path), "@model_path/layer%d/self_attn_q_proj.bin", layer);
    int q_proj = orion_gb_linear(g, rms, "q_proj", d, d * 2, s, path, NULL);
    int q_proj32 = orion_pattern_cast_to_fp32(g, q_proj, "q_proj32", d * 2, s);
    orion_gb_output(g, q_proj32, "q_proj");
    return g;
}

OrionGraph* orion_frontend_qwen35_prefill_q_proj_linear_only(int layer, int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int s = bucket;
    OrionGraph* g = orion_graph_create();
    int x16 = qwen35_normed_input(g, bucket, cfg);

    char path[256];
    snprintf(path, sizeof(path), "@model_path/layer%d/self_attn_q_proj.bin", layer);
    int q_proj = orion_gb_linear(g, x16, "q_proj", d, d * 2, s, path, NULL);
    int q_proj32 = orion_pattern_cast_to_fp32(g, q_proj, "q_proj32", d * 2, s);
    orion_gb_output(g, q_proj32, "q_proj");
    return g;
}

OrionGraph* orion_frontend_qwen35_prefill_kv_proj(int layer, int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int kv = qwen35_kv_dim(cfg);
    int s = bucket;
    OrionGraph* g = orion_graph_create();
    int rms = qwen35_input_rmsnorm(g, layer, bucket, cfg);

    char k_path[256], v_path[256];
    snprintf(k_path, sizeof(k_path), "@model_path/layer%d/self_attn_k_proj.bin", layer);
    snprintf(v_path, sizeof(v_path), "@model_path/layer%d/self_attn_v_proj.bin", layer);

    int k_proj = orion_gb_linear(g, rms, "k_proj", d, kv, s, k_path, NULL);
    int v_proj = qwen35_use_matmul_v_proj()
        ? qwen35_linear_matmul(g, rms, "v_proj", d, kv, s, v_path)
        : orion_gb_linear(g, rms, "v_proj", d, kv, s, v_path, NULL);
    int k_proj32 = orion_pattern_cast_to_fp32(g, k_proj, "k_proj32", kv, s);
    int v_proj32 = orion_pattern_cast_to_fp32(g, v_proj, "v_proj32", kv, s);

    orion_gb_output(g, k_proj32, "k_proj");
    orion_gb_output(g, v_proj32, "v_proj");
    return g;
}

OrionGraph* orion_frontend_qwen35_prefill_kv_proj_linear_only(int layer, int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int kv = qwen35_kv_dim(cfg);
    int s = bucket;
    OrionGraph* g = orion_graph_create();
    int x16 = qwen35_normed_input(g, bucket, cfg);

    char k_path[256], v_path[256];
    snprintf(k_path, sizeof(k_path), "@model_path/layer%d/self_attn_k_proj.bin", layer);
    snprintf(v_path, sizeof(v_path), "@model_path/layer%d/self_attn_v_proj.bin", layer);

    int k_proj = orion_gb_linear(g, x16, "k_proj", d, kv, s, k_path, NULL);
    int v_proj = qwen35_use_matmul_v_proj()
        ? qwen35_linear_matmul(g, x16, "v_proj", d, kv, s, v_path)
        : orion_gb_linear(g, x16, "v_proj", d, kv, s, v_path, NULL);
    int k_proj32 = orion_pattern_cast_to_fp32(g, k_proj, "k_proj32", kv, s);
    int v_proj32 = orion_pattern_cast_to_fp32(g, v_proj, "v_proj32", kv, s);

    orion_gb_output(g, k_proj32, "k_proj");
    orion_gb_output(g, v_proj32, "v_proj");
    return g;
}

OrionGraph* orion_frontend_qwen35_prefill_ffn(int layer, int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int h = cfg->hidden_dim;
    int s = bucket;
    OrionGraph* g = orion_graph_create();
    int in_shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_pattern_cast_to_fp16(g, x, "x16", d, s);
    int rms = qwen35_post_attn_rmsnorm(g, x16, layer, bucket, cfg);

    char gate_w[256], up_w[256], down_w[256];
    snprintf(gate_w, sizeof(gate_w), "@model_path/layer%d/mlp_gate_proj.bin", layer);
    snprintf(up_w, sizeof(up_w), "@model_path/layer%d/mlp_up_proj.bin", layer);
    snprintf(down_w, sizeof(down_w), "@model_path/layer%d/mlp_down_proj.bin", layer);

    int ffn = orion_pattern_swiglu_ffn(g, rms, "ffn", d, h, s, gate_w, up_w, down_w);
    int resid = orion_pattern_residual(g, x16, ffn, "ffn_resid");
    int hidden = orion_pattern_cast_to_fp32(g, resid, "hidden", d, s);
    orion_gb_output(g, hidden, "hidden");
    return g;
}
