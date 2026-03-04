// compiler/patterns.c — T119: Composite graph patterns

#include "patterns.h"
#include <stdio.h>
#include <math.h>

int orion_pattern_attention(OrionGraph* g, int q, int k, int v,
                            const char* mask_path,
                            int n_heads, int head_dim, int seq,
                            const char* prefix) {
    char buf[ORION_MAX_NAME];
    int d_model = n_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Reshape target: [1, n_heads, head_dim, seq]
    int rsh_vals[4] = {1, n_heads, head_dim, seq};
    int rsh_shape[4] = {4, 0, 0, 0};
    snprintf(buf, sizeof(buf), "%s_rsh", prefix);
    int rsh = orion_gb_const_int32(g, buf, rsh_shape, rsh_vals, 4);

    // Perm: [0,1,3,2] -> [1, n_heads, seq, head_dim]
    int pm_vals[4] = {0, 1, 3, 2};
    snprintf(buf, sizeof(buf), "%s_pm", prefix);
    int pm = orion_gb_const_int32(g, buf, rsh_shape, pm_vals, 4);

    // Q reshape + transpose
    int rsh_out[4] = {1, n_heads, head_dim, seq};
    int tr_out[4] = {1, n_heads, seq, head_dim};

    snprintf(buf, sizeof(buf), "%s_qr", prefix);
    int qr = orion_gb_reshape(g, q, rsh, buf, rsh_out);
    snprintf(buf, sizeof(buf), "%s_q", prefix);
    int q4 = orion_gb_transpose(g, qr, pm, buf, pm_vals, tr_out);

    // K reshape + transpose
    snprintf(buf, sizeof(buf), "%s_kr", prefix);
    int kr = orion_gb_reshape(g, k, rsh, buf, rsh_out);
    snprintf(buf, sizeof(buf), "%s_k", prefix);
    int k4 = orion_gb_transpose(g, kr, pm, buf, pm_vals, tr_out);

    // V reshape + transpose
    snprintf(buf, sizeof(buf), "%s_vr", prefix);
    int vr = orion_gb_reshape(g, v, rsh, buf, rsh_out);
    snprintf(buf, sizeof(buf), "%s_v", prefix);
    int v4 = orion_gb_transpose(g, vr, pm, buf, pm_vals, tr_out);

    // scores = Q @ K^T
    int sc_shape[4] = {1, n_heads, seq, seq};
    snprintf(buf, sizeof(buf), "%s_sc", prefix);
    int sc = orion_gb_matmul(g, q4, k4, false, true, buf, sc_shape);

    // scores * scale
    snprintf(buf, sizeof(buf), "%s_scv", prefix);
    int scv = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, scale);
    snprintf(buf, sizeof(buf), "%s_scs", prefix);
    int scs = orion_gb_mul(g, sc, scv, buf);

    // Causal mask
    int mask_shape[4] = {1, 1, seq, seq};
    snprintf(buf, sizeof(buf), "%s_mask", prefix);
    int mask = orion_gb_const_weight(g, buf, ORION_DTYPE_FP16, mask_shape, mask_path, 64);
    snprintf(buf, sizeof(buf), "%s_masked", prefix);
    int masked = orion_gb_add(g, scs, mask, buf);

    // Softmax
    snprintf(buf, sizeof(buf), "%s_attn", prefix);
    int attn = orion_gb_softmax(g, masked, -1, buf);

    // context = attn @ V
    int ctx_shape[4] = {1, n_heads, seq, head_dim};
    snprintf(buf, sizeof(buf), "%s_ctx", prefix);
    int ctx = orion_gb_matmul(g, attn, v4, false, false, buf, ctx_shape);

    // Transpose back: [1, nh, seq, hd] -> [1, nh, hd, seq]
    int ctxt_shape[4] = {1, n_heads, head_dim, seq};
    snprintf(buf, sizeof(buf), "%s_ctxt", prefix);
    int ctxt = orion_gb_transpose(g, ctx, pm, buf, pm_vals, ctxt_shape);

    // Reshape to [1, d_model, 1, seq]
    int osh_vals[4] = {1, d_model, 1, seq};
    snprintf(buf, sizeof(buf), "%s_osh", prefix);
    int osh = orion_gb_const_int32(g, buf, rsh_shape, osh_vals, 4);
    int out_shape[4] = {1, d_model, 1, seq};
    snprintf(buf, sizeof(buf), "%s_out", prefix);
    return orion_gb_reshape(g, ctxt, osh, buf, out_shape);
}

int orion_pattern_ffn(OrionGraph* g, int input, const char* prefix,
                      int in_dim, int hidden_dim, int seq,
                      const char* w1_path, const char* b1_path,
                      const char* w2_path, const char* b2_path,
                      int activation) {
    char buf[ORION_MAX_NAME];

    // FC up
    snprintf(buf, sizeof(buf), "%s_fc", prefix);
    int fc = orion_gb_linear(g, input, buf, in_dim, hidden_dim, seq, w1_path, b1_path);

    // Activation
    int act;
    snprintf(buf, sizeof(buf), "%s_act", prefix);
    if (activation == 1) {
        act = orion_gb_silu(g, fc, buf, hidden_dim, seq);
    } else {
        act = orion_gb_gelu(g, fc, buf, hidden_dim, seq);
    }

    // FC down
    snprintf(buf, sizeof(buf), "%s_proj", prefix);
    return orion_gb_linear(g, act, buf, hidden_dim, in_dim, seq, w2_path, b2_path);
}

int orion_pattern_swiglu_ffn(OrionGraph* g, int input, const char* prefix,
                             int in_dim, int hidden_dim, int seq,
                             const char* w1_path, const char* w3_path,
                             const char* w2_path) {
    char buf[ORION_MAX_NAME];

    // W1 and W3 parallel up-projections
    snprintf(buf, sizeof(buf), "%s_w1", prefix);
    int h1 = orion_gb_linear(g, input, buf, in_dim, hidden_dim, seq, w1_path, NULL);

    snprintf(buf, sizeof(buf), "%s_w3", prefix);
    int h3 = orion_gb_linear(g, input, buf, in_dim, hidden_dim, seq, w3_path, NULL);

    // SiLU(h1)
    snprintf(buf, sizeof(buf), "%s_silu", prefix);
    int silu = orion_gb_silu(g, h1, buf, hidden_dim, seq);

    // gate = silu(h1) * h3
    snprintf(buf, sizeof(buf), "%s_gate", prefix);
    int gate = orion_gb_mul(g, silu, h3, buf);

    // W2 down-projection
    snprintf(buf, sizeof(buf), "%s_w2", prefix);
    return orion_gb_linear(g, gate, buf, hidden_dim, in_dim, seq, w2_path, NULL);
}

int orion_pattern_residual(OrionGraph* g, int x, int sublayer_output, const char* name) {
    return orion_gb_add(g, x, sublayer_output, name);
}

int orion_pattern_cast_to_fp16(OrionGraph* g, int input, const char* name, int dim, int seq) {
    int shape[4] = {1, dim, 1, seq};
    return orion_gb_cast(g, input, ORION_DTYPE_FP16, name, shape);
}

int orion_pattern_cast_to_fp32(OrionGraph* g, int input, const char* name, int dim, int seq) {
    int shape[4] = {1, dim, 1, seq};
    return orion_gb_cast(g, input, ORION_DTYPE_FP32, name, shape);
}
