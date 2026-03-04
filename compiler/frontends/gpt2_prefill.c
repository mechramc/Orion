// compiler/frontends/gpt2_prefill.c — T134: GPT-2 prefill frontend

#include "gpt2_prefill.h"
#include "../builder.h"
#include "../patterns.h"
#include <stdio.h>

OrionGraph* orion_frontend_gpt2_prefill_attn(int layer, int bucket, const OrionModelConfig* cfg) {
    int d  = cfg->d_model;
    int nh = cfg->n_head;
    int hd = cfg->head_dim;
    int s  = bucket;
    OrionGraph* g = orion_graph_create();

    // Input: fp32 [1, d, 1, s]
    int in_shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);

    // Cast to fp16
    int x16 = orion_pattern_cast_to_fp16(g, x, "x16", d, s);

    // Weight paths
    char path[256];

    // LayerNorm 1
    int ln1_g_shape[4] = {1, d, 1, 1};
    snprintf(path, sizeof(path), "@model_path/layer%d/ln1_g.bin", layer);
    int ln1_g = orion_gb_const_weight(g, "ln1_g", ORION_DTYPE_FP16, ln1_g_shape, path, 64);
    snprintf(path, sizeof(path), "@model_path/layer%d/ln1_b.bin", layer);
    int ln1_b = orion_gb_const_weight(g, "ln1_beta", ORION_DTYPE_FP16, ln1_g_shape, path, 64);

    int ln1 = orion_gb_layernorm(g, x16, ln1_g, ln1_b, 1e-5f, "ln1", d, s);

    // Q, K, V projections
    char wpath[256], bpath[256];
    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wq.bin", layer);
    snprintf(bpath, sizeof(bpath), "@model_path/layer%d/bq.bin", layer);
    int q = orion_gb_linear(g, ln1, "q", d, d, s, wpath, bpath);

    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wk.bin", layer);
    snprintf(bpath, sizeof(bpath), "@model_path/layer%d/bk.bin", layer);
    int k = orion_gb_linear(g, ln1, "k", d, d, s, wpath, bpath);

    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wv.bin", layer);
    snprintf(bpath, sizeof(bpath), "@model_path/layer%d/bv.bin", layer);
    int v = orion_gb_linear(g, ln1, "v", d, d, s, wpath, bpath);

    // Causal attention
    char mask_path[256];
    snprintf(mask_path, sizeof(mask_path), "@model_path/masks/causal_%d.bin", s);
    int attn = orion_pattern_attention(g, q, k, v, mask_path, nh, hd, s, "attn");

    // Output projection
    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wo.bin", layer);
    snprintf(bpath, sizeof(bpath), "@model_path/layer%d/bo.bin", layer);
    int proj = orion_gb_linear(g, attn, "proj", d, d, s, wpath, bpath);

    // Residual
    int resid = orion_pattern_residual(g, x16, proj, "resid");

    // Cast outputs to fp32
    int hidden = orion_pattern_cast_to_fp32(g, resid, "hidden", d, s);
    int k_cache = orion_pattern_cast_to_fp32(g, k, "k_cache", d, s);
    int v_cache = orion_pattern_cast_to_fp32(g, v, "v_cache", d, s);

    // Mark outputs
    orion_gb_output(g, hidden, "hidden");
    orion_gb_output(g, k_cache, "k_cache");
    orion_gb_output(g, v_cache, "v_cache");

    return g;
}

OrionGraph* orion_frontend_gpt2_prefill_ffn(int layer, int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int h = cfg->hidden_dim;
    int s = bucket;
    OrionGraph* g = orion_graph_create();

    // Input
    int in_shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_pattern_cast_to_fp16(g, x, "x16", d, s);

    // LayerNorm 2
    int ln_shape[4] = {1, d, 1, 1};
    char path[256];
    snprintf(path, sizeof(path), "@model_path/layer%d/ln2_g.bin", layer);
    int ln2_g = orion_gb_const_weight(g, "ln2_g", ORION_DTYPE_FP16, ln_shape, path, 64);
    snprintf(path, sizeof(path), "@model_path/layer%d/ln2_b.bin", layer);
    int ln2_b = orion_gb_const_weight(g, "ln2_beta", ORION_DTYPE_FP16, ln_shape, path, 64);
    int ln2 = orion_gb_layernorm(g, x16, ln2_g, ln2_b, 1e-5f, "ln2", d, s);

    // FFN: fc -> gelu -> proj
    char w1[256], b1[256], w2[256], b2[256];
    snprintf(w1, sizeof(w1), "@model_path/layer%d/wfc.bin", layer);
    snprintf(b1, sizeof(b1), "@model_path/layer%d/bfc.bin", layer);
    snprintf(w2, sizeof(w2), "@model_path/layer%d/wproj.bin", layer);
    snprintf(b2, sizeof(b2), "@model_path/layer%d/bproj.bin", layer);
    int ffn = orion_pattern_ffn(g, ln2, "ffn", d, h, s, w1, b1, w2, b2, 0); // 0=GELU

    // Residual
    int resid = orion_pattern_residual(g, x16, ffn, "resid");

    // Cast to fp32
    int hidden = orion_pattern_cast_to_fp32(g, resid, "hidden", d, s);

    orion_gb_output(g, hidden, "hidden");
    return g;
}
