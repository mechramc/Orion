// compiler/frontends/gpt2_decode.c — T135: GPT-2 decode frontend

#include "gpt2_decode.h"
#include "../builder.h"
#include "../patterns.h"
#include <stdio.h>

OrionGraph* orion_frontend_gpt2_decode_proj(int layer, const OrionModelConfig* cfg) {
    int d   = cfg->d_model;
    int seq = ORION_GRAPH_DECODE_SEQ;
    OrionGraph* g = orion_graph_create();

    // Input
    int in_shape[4] = {1, d, 1, seq};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_pattern_cast_to_fp16(g, x, "x16", d, seq);

    // LayerNorm 1
    int ln_shape[4] = {1, d, 1, 1};
    char path[256];
    snprintf(path, sizeof(path), "@model_path/layer%d/ln1_g.bin", layer);
    int ln1_g = orion_gb_const_weight(g, "ln1_g", ORION_DTYPE_FP16, ln_shape, path, 64);
    snprintf(path, sizeof(path), "@model_path/layer%d/ln1_b.bin", layer);
    int ln1_b = orion_gb_const_weight(g, "ln1_beta", ORION_DTYPE_FP16, ln_shape, path, 64);
    int ln1 = orion_gb_layernorm(g, x16, ln1_g, ln1_b, 1e-5f, "ln1", d, seq);

    // Q, K, V projections
    char wpath[256], bpath[256];
    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wq.bin", layer);
    snprintf(bpath, sizeof(bpath), "@model_path/layer%d/bq.bin", layer);
    int q = orion_gb_linear(g, ln1, "q", d, d, seq, wpath, bpath);

    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wk.bin", layer);
    snprintf(bpath, sizeof(bpath), "@model_path/layer%d/bk.bin", layer);
    int k = orion_gb_linear(g, ln1, "k", d, d, seq, wpath, bpath);

    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wv.bin", layer);
    snprintf(bpath, sizeof(bpath), "@model_path/layer%d/bv.bin", layer);
    int v = orion_gb_linear(g, ln1, "v", d, d, seq, wpath, bpath);

    // Cast to fp32
    int q32 = orion_pattern_cast_to_fp32(g, q, "q32", d, seq);
    int k32 = orion_pattern_cast_to_fp32(g, k, "k32", d, seq);
    int v32 = orion_pattern_cast_to_fp32(g, v, "v32", d, seq);

    // Outputs in alphabetical order: k32, q32, v32
    orion_gb_output(g, q32, "q32");
    orion_gb_output(g, k32, "k32");
    orion_gb_output(g, v32, "v32");

    return g;
}

OrionGraph* orion_frontend_gpt2_decode_ffn(int layer, const OrionModelConfig* cfg) {
    int d   = cfg->d_model;
    int h   = cfg->hidden_dim;
    int seq = ORION_GRAPH_DECODE_SEQ;
    OrionGraph* g = orion_graph_create();

    // Input
    int in_shape[4] = {1, d, 1, seq};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_pattern_cast_to_fp16(g, x, "x16", d, seq);

    // LayerNorm 2
    int ln_shape[4] = {1, d, 1, 1};
    char path[256];
    snprintf(path, sizeof(path), "@model_path/layer%d/ln2_g.bin", layer);
    int ln2_g = orion_gb_const_weight(g, "ln2_g", ORION_DTYPE_FP16, ln_shape, path, 64);
    snprintf(path, sizeof(path), "@model_path/layer%d/ln2_b.bin", layer);
    int ln2_b = orion_gb_const_weight(g, "ln2_beta", ORION_DTYPE_FP16, ln_shape, path, 64);
    int ln2 = orion_gb_layernorm(g, x16, ln2_g, ln2_b, 1e-5f, "ln2", d, seq);

    // FFN
    char w1[256], b1[256], w2[256], b2[256];
    snprintf(w1, sizeof(w1), "@model_path/layer%d/wfc.bin", layer);
    snprintf(b1, sizeof(b1), "@model_path/layer%d/bfc.bin", layer);
    snprintf(w2, sizeof(w2), "@model_path/layer%d/wproj.bin", layer);
    snprintf(b2, sizeof(b2), "@model_path/layer%d/bproj.bin", layer);
    int ffn = orion_pattern_ffn(g, ln2, "ffn", d, h, seq, w1, b1, w2, b2, 0);

    // Residual + cast
    int resid = orion_pattern_residual(g, x16, ffn, "resid");
    int hidden = orion_pattern_cast_to_fp32(g, resid, "hidden", d, seq);

    orion_gb_output(g, hidden, "hidden");
    return g;
}
