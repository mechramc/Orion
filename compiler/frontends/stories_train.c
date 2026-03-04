// compiler/frontends/stories_train.c — T136: Stories110M training frontends

#include "stories_train.h"
#include "../builder.h"
#include "../patterns.h"
#include <stdio.h>
#include <math.h>

OrionGraph* orion_frontend_fwd_attn(int layer, const OrionModelConfig* cfg) {
    int d  = cfg->d_model;
    int nh = cfg->n_head;
    int hd = cfg->head_dim;
    int s  = cfg->max_seq;
    OrionGraph* g = orion_graph_create();

    // Input: fp16 [1, d, 1, s]
    int in_shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, in_shape);

    // RMSNorm
    char path[256];
    int w_shape[4] = {1, d, 1, 1};
    snprintf(path, sizeof(path), "@model_path/layer%d/rms1.bin", layer);
    int rms_w = orion_gb_const_weight(g, "rms1_w", ORION_DTYPE_FP16, w_shape, path, 64);
    int rms1 = orion_gb_rmsnorm(g, x, rms_w, 1e-5f, "rms1", d, s);

    // QKV projections (no bias, Llama-style)
    char wpath[256];
    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wq.bin", layer);
    int q = orion_gb_linear(g, rms1, "q", d, d, s, wpath, NULL);

    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wk.bin", layer);
    int k = orion_gb_linear(g, rms1, "k", d, d, s, wpath, NULL);

    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wv.bin", layer);
    int v = orion_gb_linear(g, rms1, "v", d, d, s, wpath, NULL);

    // Causal attention
    char mask_path[256];
    snprintf(mask_path, sizeof(mask_path), "@model_path/masks/causal_%d.bin", s);
    int attn = orion_pattern_attention(g, q, k, v, mask_path, nh, hd, s, "attn");

    // Output projection
    snprintf(wpath, sizeof(wpath), "@model_path/layer%d/wo.bin", layer);
    int wo = orion_gb_linear(g, attn, "wo", d, d, s, wpath, NULL);

    // Multi-output (alphabetical: attn_out, k_out, q_out, rms1_out, v_out, wo_out)
    orion_gb_output(g, wo, "wo_out");
    orion_gb_output(g, q, "q_out");
    orion_gb_output(g, k, "k_out");
    orion_gb_output(g, v, "v_out");
    orion_gb_output(g, attn, "attn_out");
    orion_gb_output(g, rms1, "rms1_out");

    return g;
}

OrionGraph* orion_frontend_fwd_ffn(int layer, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int h = cfg->hidden_dim;
    int s = cfg->max_seq;
    OrionGraph* g = orion_graph_create();

    int in_shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, in_shape);

    // RMSNorm
    char path[256];
    int w_shape[4] = {1, d, 1, 1};
    snprintf(path, sizeof(path), "@model_path/layer%d/rms2.bin", layer);
    int rms_w = orion_gb_const_weight(g, "rms2_w", ORION_DTYPE_FP16, w_shape, path, 64);
    int rms2 = orion_gb_rmsnorm(g, x, rms_w, 1e-5f, "rms2", d, s);

    // SwiGLU FFN
    char w1[256], w3[256], w2[256];
    snprintf(w1, sizeof(w1), "@model_path/layer%d/w1.bin", layer);
    snprintf(w3, sizeof(w3), "@model_path/layer%d/w3.bin", layer);
    snprintf(w2, sizeof(w2), "@model_path/layer%d/w2.bin", layer);

    // W1 and W3 up-projections
    int h1 = orion_gb_linear(g, rms2, "w1", d, h, s, w1, NULL);
    int h3 = orion_gb_linear(g, rms2, "w3", d, h, s, w3, NULL);

    // SiLU(h1) * h3
    int silu = orion_gb_silu(g, h1, "silu", h, s);
    int gate = orion_gb_mul(g, silu, h3, "gate");

    // W2 down-projection
    int w2_out = orion_gb_linear(g, gate, "w2", h, d, s, w2, NULL);

    // Multi-output
    orion_gb_output(g, w2_out, "w2_out");
    orion_gb_output(g, h1, "w1_out");
    orion_gb_output(g, h3, "w3_out");
    orion_gb_output(g, gate, "gate");
    orion_gb_output(g, rms2, "rms2_out");

    return g;
}

OrionGraph* orion_frontend_ffn_bwd(int layer, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int h = cfg->hidden_dim;
    int s = cfg->max_seq;
    int total_in = d + 2 * h;
    OrionGraph* g = orion_graph_create();

    // Input: packed [dffn(d), h1(h), h3(h)]
    int in_shape[4] = {1, total_in, 1, s};
    int inp = orion_gb_input(g, "inp", ORION_DTYPE_FP16, in_shape);

    // Slice dffn: [0, d)
    int b0_vals[4] = {0,0,0,0}, e0_vals[4] = {1,d,1,s};
    int b0_sh[4] = {4,0,0,0};
    int b0 = orion_gb_const_int32(g, "sl_dffn_begin", b0_sh, b0_vals, 4);
    int e0 = orion_gb_const_int32(g, "sl_dffn_end", b0_sh, e0_vals, 4);
    int sl0_shape[4] = {1, d, 1, s};
    int dffn = orion_gb_slice(g, inp, b0, e0, "dffn", sl0_shape);

    // Slice h1: [d, d+h)
    int b1_vals[4] = {0,d,0,0}, e1_vals[4] = {1,d+h,1,s};
    int b1 = orion_gb_const_int32(g, "sl_h1_begin", b0_sh, b1_vals, 4);
    int e1 = orion_gb_const_int32(g, "sl_h1_end", b0_sh, e1_vals, 4);
    int sl1_shape[4] = {1, h, 1, s};
    int h1 = orion_gb_slice(g, inp, b1, e1, "h1", sl1_shape);

    // Slice h3: [d+h, d+2h)
    int b2_vals[4] = {0,d+h,0,0}, e2_vals[4] = {1,d+2*h,1,s};
    int b2 = orion_gb_const_int32(g, "sl_h3_begin", b0_sh, b2_vals, 4);
    int e2 = orion_gb_const_int32(g, "sl_h3_end", b0_sh, e2_vals, 4);
    int h3 = orion_gb_slice(g, inp, b2, e2, "h3", sl1_shape);

    // Backprop through W2: dsilu = W2^T @ dffn
    char path[256];
    snprintf(path, sizeof(path), "@model_path/layer%d/w2t.bin", layer);
    int dsilu = orion_gb_linear(g, dffn, "w2t", d, h, s, path, NULL);

    // SiLU backward: d(SiLU)/dh1 = sig * (1 + h1 * (1 - sig))
    int sig = orion_gb_sigmoid(g, h1, "sig");
    int silu_recomp = orion_gb_mul(g, h1, sig, "silu");

    int one_c = orion_gb_const_scalar(g, "one_c", ORION_DTYPE_FP16, 1.0f);
    int oms = orion_gb_sub(g, one_c, sig, "oms");
    int homs = orion_gb_mul(g, h1, oms, "homs");
    int brk = orion_gb_add(g, one_c, homs, "brk");
    int dsd = orion_gb_mul(g, sig, brk, "dsd");

    // dh1 = dsilu * h3 * dsd
    int dsh3 = orion_gb_mul(g, dsilu, h3, "dsh3");
    int dh1 = orion_gb_mul(g, dsh3, dsd, "dh1");

    // dh3 = dsilu * silu(h1)
    int dh3 = orion_gb_mul(g, dsilu, silu_recomp, "dh3");

    // Backprop through W1, W3
    snprintf(path, sizeof(path), "@model_path/layer%d/w1t.bin", layer);
    int dx1 = orion_gb_linear(g, dh1, "w1t", h, d, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/layer%d/w3t.bin", layer);
    int dx3 = orion_gb_linear(g, dh3, "w3t", h, d, s, path, NULL);

    int dx = orion_gb_add(g, dx1, dx3, "dx");

    orion_gb_output(g, dx, "dx");
    orion_gb_output(g, dh1, "dh1");
    orion_gb_output(g, dh3, "dh3");

    return g;
}

OrionGraph* orion_frontend_sdpa_bwd1(int layer, const OrionModelConfig* cfg) {
    int d  = cfg->d_model;
    int nh = cfg->n_head;
    int hd = cfg->head_dim;
    int s  = cfg->max_seq;
    int sc_ch = nh * s;
    float scale = 1.0f / sqrtf((float)hd);
    int total_in = 4 * d;
    OrionGraph* g = orion_graph_create();

    int in_shape[4] = {1, total_in, 1, s};
    int inp = orion_gb_input(g, "inp", ORION_DTYPE_FP16, in_shape);

    // Slice: qf, kf, vf, dx2f
    int bsh[4] = {4,0,0,0};
    int b0[4] = {0,0,0,0}, e0[4] = {1,d,1,s};
    int qf_b = orion_gb_const_int32(g, "sl0_b", bsh, b0, 4);
    int qf_e = orion_gb_const_int32(g, "sl0_e", bsh, e0, 4);
    int d_shape[4] = {1,d,1,s};
    int qf = orion_gb_slice(g, inp, qf_b, qf_e, "qf", d_shape);

    int b1[4] = {0,d,0,0}, e1[4] = {1,2*d,1,s};
    int kf_b = orion_gb_const_int32(g, "sl1_b", bsh, b1, 4);
    int kf_e = orion_gb_const_int32(g, "sl1_e", bsh, e1, 4);
    int kf = orion_gb_slice(g, inp, kf_b, kf_e, "kf", d_shape);

    int b2[4] = {0,2*d,0,0}, e2[4] = {1,3*d,1,s};
    int vf_b = orion_gb_const_int32(g, "sl2_b", bsh, b2, 4);
    int vf_e = orion_gb_const_int32(g, "sl2_e", bsh, e2, 4);
    int vf = orion_gb_slice(g, inp, vf_b, vf_e, "vf", d_shape);

    int b3[4] = {0,3*d,0,0}, e3[4] = {1,4*d,1,s};
    int dx_b = orion_gb_const_int32(g, "sl3_b", bsh, b3, 4);
    int dx_e = orion_gb_const_int32(g, "sl3_e", bsh, e3, 4);
    int dx2f = orion_gb_slice(g, inp, dx_b, dx_e, "dx2f", d_shape);

    // Backprop through Wo
    char path[256];
    snprintf(path, sizeof(path), "@model_path/layer%d/wot.bin", layer);
    int wot_out = orion_gb_linear(g, dx2f, "wot", d, d, s, path, NULL);

    // Reshape Q,K,V,df to multi-head: [1,d,1,s]->[1,nh,hd,s]->[1,nh,s,hd]
    int rsh4d_vals[4] = {1,nh,hd,s};
    int pm_vals[4] = {0,1,3,2};
    int rsh4d = orion_gb_const_int32(g, "rsh4d", bsh, rsh4d_vals, 4);
    int pm = orion_gb_const_int32(g, "perm_t", bsh, pm_vals, 4);

    int rsh_out[4] = {1,nh,hd,s};
    int tr_out[4] = {1,nh,s,hd};

    int qr = orion_gb_reshape(g, qf, rsh4d, "qr", rsh_out);
    int q4 = orion_gb_transpose(g, qr, pm, "q4", pm_vals, tr_out);

    int kr = orion_gb_reshape(g, kf, rsh4d, "kr", rsh_out);
    int k4 = orion_gb_transpose(g, kr, pm, "k4", pm_vals, tr_out);

    int vr = orion_gb_reshape(g, vf, rsh4d, "vr", rsh_out);
    int v4 = orion_gb_transpose(g, vr, pm, "v4", pm_vals, tr_out);

    int dfr = orion_gb_reshape(g, wot_out, rsh4d, "dfr", rsh_out);
    int da = orion_gb_transpose(g, dfr, pm, "da", pm_vals, tr_out);

    // Recompute attention forward
    int sc_shape[4] = {1,nh,s,s};
    int scores = orion_gb_matmul(g, q4, k4, false, true, "scores", sc_shape);
    int sc_val = orion_gb_const_scalar(g, "sc_val", ORION_DTYPE_FP16, scale);
    int scores_sc = orion_gb_mul(g, scores, sc_val, "scores_sc");

    // Causal mask
    char mask_path[256];
    snprintf(mask_path, sizeof(mask_path), "@model_path/masks/causal_%d.bin", s);
    int mask_shape[4] = {1,1,s,s};
    int cmask = orion_gb_const_weight(g, "cmask", ORION_DTYPE_FP16, mask_shape, mask_path, 64);
    int masked = orion_gb_add(g, scores_sc, cmask, "masked");
    int probs = orion_gb_softmax(g, masked, -1, "probs");

    // dV = probs^T @ da
    int dv_shape[4] = {1,nh,s,hd};
    int dv4 = orion_gb_matmul(g, probs, da, true, false, "dv4", dv_shape);

    // dp = da @ V^T
    int dp_shape[4] = {1,nh,s,s};
    int dp4 = orion_gb_matmul(g, da, v4, false, true, "dp4", dp_shape);

    // Flatten dV: [1,nh,s,hd]->[1,nh,hd,s]->[1,d,1,s]
    int dvt_shape[4] = {1,nh,hd,s};
    int dvt = orion_gb_transpose(g, dv4, pm, "dvt", pm_vals, dvt_shape);
    int rsh_d_vals[4] = {1,d,1,s};
    int rsh_d = orion_gb_const_int32(g, "rsh_d", bsh, rsh_d_vals, 4);
    int dvf = orion_gb_reshape(g, dvt, rsh_d, "dvf", d_shape);

    // Flatten probs: [1,nh,s,s]->[1,sc_ch,1,s]
    int rsh_sc_vals[4] = {1,sc_ch,1,s};
    int rsh_sc = orion_gb_const_int32(g, "rsh_sc", bsh, rsh_sc_vals, 4);
    int sc_flat_shape[4] = {1,sc_ch,1,s};
    int pf = orion_gb_reshape(g, probs, rsh_sc, "pf", sc_flat_shape);
    int dpf = orion_gb_reshape(g, dp4, rsh_sc, "dpf", sc_flat_shape);

    orion_gb_output(g, dvf, "dvf");
    orion_gb_output(g, pf, "pf");
    orion_gb_output(g, dpf, "dpf");

    return g;
}

OrionGraph* orion_frontend_sdpa_bwd2(int layer __attribute__((unused)), const OrionModelConfig* cfg) {
    int d  = cfg->d_model;
    int nh = cfg->n_head;
    int hd = cfg->head_dim;
    int s  = cfg->max_seq;
    int sc_ch = nh * s;
    float scale = 1.0f / sqrtf((float)hd);
    int total_in = 2 * sc_ch + 2 * d;
    OrionGraph* g = orion_graph_create();

    int in_shape[4] = {1, total_in, 1, s};
    int inp = orion_gb_input(g, "inp", ORION_DTYPE_FP16, in_shape);

    int bsh[4] = {4,0,0,0};

    // Slice pf, dpf, qf, kf
    int b0[4] = {0,0,0,0}, e0[4] = {1,sc_ch,1,s};
    int sc_shape[4] = {1,sc_ch,1,s};
    int pf_b = orion_gb_const_int32(g, "s0b", bsh, b0, 4);
    int pf_e = orion_gb_const_int32(g, "s0e", bsh, e0, 4);
    int pf = orion_gb_slice(g, inp, pf_b, pf_e, "pf", sc_shape);

    int b1[4] = {0,sc_ch,0,0}, e1[4] = {1,2*sc_ch,1,s};
    int dpf_b = orion_gb_const_int32(g, "s1b", bsh, b1, 4);
    int dpf_e = orion_gb_const_int32(g, "s1e", bsh, e1, 4);
    int dpf = orion_gb_slice(g, inp, dpf_b, dpf_e, "dpf", sc_shape);

    int d_shape[4] = {1,d,1,s};
    int b2[4] = {0,2*sc_ch,0,0}, e2[4] = {1,2*sc_ch+d,1,s};
    int qf_b = orion_gb_const_int32(g, "s2b", bsh, b2, 4);
    int qf_e = orion_gb_const_int32(g, "s2e", bsh, e2, 4);
    int qf = orion_gb_slice(g, inp, qf_b, qf_e, "qf", d_shape);

    int b3[4] = {0,2*sc_ch+d,0,0}, e3[4] = {1,total_in,1,s};
    int kf_b = orion_gb_const_int32(g, "s3b", bsh, b3, 4);
    int kf_e = orion_gb_const_int32(g, "s3e", bsh, e3, 4);
    int kf = orion_gb_slice(g, inp, kf_b, kf_e, "kf", d_shape);

    // Reshape to 4D: probs/dp [1,sc_ch,1,s]->[1,nh,s,s], Q/K [1,d,1,s]->[1,nh,s,hd]
    int rsh_sc_vals[4] = {1,nh,s,s};
    int rsh_sc = orion_gb_const_int32(g, "rsh_sc", bsh, rsh_sc_vals, 4);
    int nhss_shape[4] = {1,nh,s,s};
    int p4 = orion_gb_reshape(g, pf, rsh_sc, "p4", nhss_shape);
    int dp4 = orion_gb_reshape(g, dpf, rsh_sc, "dp4", nhss_shape);

    int rsh4d_vals[4] = {1,nh,hd,s};
    int rsh4d = orion_gb_const_int32(g, "rsh4d", bsh, rsh4d_vals, 4);
    int pm_vals[4] = {0,1,3,2};
    int pm = orion_gb_const_int32(g, "perm_t", bsh, pm_vals, 4);

    int rsh_out[4] = {1,nh,hd,s};
    int tr_out[4] = {1,nh,s,hd};
    int qr = orion_gb_reshape(g, qf, rsh4d, "qr", rsh_out);
    int q4 = orion_gb_transpose(g, qr, pm, "q4", pm_vals, tr_out);
    int kr = orion_gb_reshape(g, kf, rsh4d, "kr", rsh_out);
    int k4 = orion_gb_transpose(g, kr, pm, "k4", pm_vals, tr_out);

    // Softmax backward: ds = p * (dp - sum(p*dp, axis=-1, keepdims=true))
    int pdp = orion_gb_mul(g, p4, dp4, "pdp");

    int ax_vals[1] = {-1};
    int ax_shape[4] = {1,0,0,0};
    int ax_neg1 = orion_gb_const_int32(g, "ax_neg1", ax_shape, ax_vals, 1);
    int spdp_shape[4] = {1,nh,s,1};
    int spdp = orion_gb_reduce_sum(g, pdp, ax_neg1, true, "spdp", spdp_shape);
    int dps = orion_gb_sub(g, dp4, spdp, "dps");
    int ds_raw = orion_gb_mul(g, p4, dps, "ds_raw");

    int sc_val = orion_gb_const_scalar(g, "sc_val", ORION_DTYPE_FP16, scale);
    int ds = orion_gb_mul(g, ds_raw, sc_val, "ds");

    // dQ = ds @ K, dK = ds^T @ Q
    int dq_shape[4] = {1,nh,s,hd};
    int dq4 = orion_gb_matmul(g, ds, k4, false, false, "dq4", dq_shape);
    int dk4 = orion_gb_matmul(g, ds, q4, true, false, "dk4", dq_shape);

    // Reshape back: [1,nh,s,hd]->[1,nh,hd,s]->[1,d,1,s]
    int dqt_shape[4] = {1,nh,hd,s};
    int dqt = orion_gb_transpose(g, dq4, pm, "dqt", pm_vals, dqt_shape);
    int rsh_d_vals[4] = {1,d,1,s};
    int rsh_d = orion_gb_const_int32(g, "rsh_d", bsh, rsh_d_vals, 4);
    int dqf = orion_gb_reshape(g, dqt, rsh_d, "dqf", d_shape);

    int dkt = orion_gb_transpose(g, dk4, pm, "dkt", pm_vals, dqt_shape);
    int dkf = orion_gb_reshape(g, dkt, rsh_d, "dkf", d_shape);

    orion_gb_output(g, dqf, "dqf");
    orion_gb_output(g, dkf, "dkf");

    return g;
}

OrionGraph* orion_frontend_qkv_bwd(int layer, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int s = cfg->max_seq;
    int total_in = 3 * d;
    OrionGraph* g = orion_graph_create();

    int in_shape[4] = {1, total_in, 1, s};
    int inp = orion_gb_input(g, "inp", ORION_DTYPE_FP16, in_shape);

    int bsh[4] = {4,0,0,0};
    int d_shape[4] = {1,d,1,s};

    int b0[4] = {0,0,0,0}, e0[4] = {1,d,1,s};
    int dq_b = orion_gb_const_int32(g, "s0b", bsh, b0, 4);
    int dq_e = orion_gb_const_int32(g, "s0e", bsh, e0, 4);
    int dq = orion_gb_slice(g, inp, dq_b, dq_e, "dq", d_shape);

    int b1[4] = {0,d,0,0}, e1[4] = {1,2*d,1,s};
    int dk_b = orion_gb_const_int32(g, "s1b", bsh, b1, 4);
    int dk_e = orion_gb_const_int32(g, "s1e", bsh, e1, 4);
    int dk = orion_gb_slice(g, inp, dk_b, dk_e, "dk", d_shape);

    int b2[4] = {0,2*d,0,0}, e2[4] = {1,3*d,1,s};
    int dv_b = orion_gb_const_int32(g, "s2b", bsh, b2, 4);
    int dv_e = orion_gb_const_int32(g, "s2e", bsh, e2, 4);
    int dv = orion_gb_slice(g, inp, dv_b, dv_e, "dv", d_shape);

    char path[256];
    snprintf(path, sizeof(path), "@model_path/layer%d/wqt.bin", layer);
    int dxq = orion_gb_linear(g, dq, "wqt", d, d, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/layer%d/wkt.bin", layer);
    int dxk = orion_gb_linear(g, dk, "wkt", d, d, s, path, NULL);
    snprintf(path, sizeof(path), "@model_path/layer%d/wvt.bin", layer);
    int dxv = orion_gb_linear(g, dv, "wvt", d, d, s, path, NULL);

    int dx_qk = orion_gb_add(g, dxq, dxk, "dx_qk");
    int dx = orion_gb_add(g, dx_qk, dxv, "dx");

    orion_gb_output(g, dx, "dx");

    return g;
}
