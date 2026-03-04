#import "stories_train_kernels.milgen.h"
#import <math.h>

// T064-T069: MIL generators for Stories110M training kernels.
// All tensors are fp16 [1, C, 1, S] on ANE.
// SEQ is always 256 for training (from config max_seq).
// Multi-output kernels use orion_mil_program_multi (separate IOSurface per output).
// NOTE: concat along axis=1 is rejected by ANE compiler — use multi-output instead.

#pragma mark - Helpers


#pragma mark - T064: fwdAttn — Attention Forward with Taps

NSString* orion_milgen_fwd_attn(int layer_idx, const OrionModelConfig* cfg) {
    int d  = cfg->d_model;   // 768
    int nh = cfg->n_head;    // 12
    int hd = cfg->head_dim;  // 64
    int s  = cfg->max_seq;   // 256

    // Weight blob paths
    NSString *rms1 = [NSString stringWithFormat:@"@model_path/layer%d/rms1.bin", layer_idx];
    NSString *wq   = [NSString stringWithFormat:@"@model_path/layer%d/wq.bin", layer_idx];
    NSString *wk   = [NSString stringWithFormat:@"@model_path/layer%d/wk.bin", layer_idx];
    NSString *wv   = [NSString stringWithFormat:@"@model_path/layer%d/wv.bin", layer_idx];
    NSString *wo   = [NSString stringWithFormat:@"@model_path/layer%d/wo.bin", layer_idx];
    NSString *mask = [NSString stringWithFormat:@"@model_path/masks/causal_%d.bin", s];

    NSMutableString *body = [NSMutableString string];

    // Input is fp16 (training kernels work in fp16 throughout)
    // RMSNorm
    [body appendString:orion_mil_rmsnorm("rms1", "x", d, s, rms1.UTF8String, 1e-5f)];

    // QKV projections (no bias for Llama-style)
    [body appendString:orion_mil_linear("q", "rms1_out", d, d, s, wq.UTF8String, NULL)];
    [body appendString:orion_mil_linear("k", "rms1_out", d, d, s, wk.UTF8String, NULL)];
    [body appendString:orion_mil_linear("v", "rms1_out", d, d, s, wv.UTF8String, NULL)];

    // Causal attention (decomposed)
    [body appendString:orion_mil_causal_attention("attn", "q_out", "k_out", "v_out",
                                                   nh, hd, s, mask.UTF8String)];

    // Output projection (no bias)
    [body appendString:orion_mil_linear("wo", "attn_out", d, d, s, wo.UTF8String, NULL)];

    // Multi-output: fp16 outputs (zero-copy for backward input assembly)
    // oo=Wo output, qf=Q, kf=K, vf=V, af=attn, xn=RMSNorm
    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp16, [1,%d,1,%d]> x", d, s];

    return orion_mil_program_multi(body, @[input_decl],
        @[@"wo_out", @"q_out", @"k_out", @"v_out", @"attn_out", @"rms1_out"]);
}

#pragma mark - T065: fwdFFN — FFN Forward with Taps

NSString* orion_milgen_fwd_ffn(int layer_idx, const OrionModelConfig* cfg) {
    int d  = cfg->d_model;     // 768
    int h  = cfg->hidden_dim;  // 2048
    int s  = cfg->max_seq;     // 256

    // Weight blob paths
    NSString *rms2 = [NSString stringWithFormat:@"@model_path/layer%d/rms2.bin", layer_idx];
    NSString *w1   = [NSString stringWithFormat:@"@model_path/layer%d/w1.bin", layer_idx];
    NSString *w3   = [NSString stringWithFormat:@"@model_path/layer%d/w3.bin", layer_idx];
    NSString *w2   = [NSString stringWithFormat:@"@model_path/layer%d/w2.bin", layer_idx];

    NSMutableString *body = [NSMutableString string];

    // RMSNorm
    [body appendString:orion_mil_rmsnorm("rms2", "x", d, s, rms2.UTF8String, 1e-5f)];

    // Parallel up-projections (SwiGLU architecture)
    // h1 = W1 @ xn → [1, hidden, 1, seq]
    // h3 = W3 @ xn → [1, hidden, 1, seq]
    [body appendString:orion_mil_linear("w1", "rms2_out", d, h, s, w1.UTF8String, NULL)];
    [body appendString:orion_mil_linear("w3", "rms2_out", d, h, s, w3.UTF8String, NULL)];

    // SiLU(h1): sig = sigmoid(h1), silu = h1 * sig
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> silu_sig = sigmoid(x=w1_out)[name=string(\"silu_sig\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> silu_out = mul(x=w1_out, y=silu_sig)[name=string(\"silu_out\")];\n",
        h, s, h, s];

    // Gate: gate = silu(h1) * h3
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu_out, y=w3_out)[name=string(\"gate\")];\n",
        h, s];

    // Down-projection: y = W2 @ gate → [1, d_model, 1, seq]
    [body appendString:orion_mil_linear("w2", "gate", h, d, s, w2.UTF8String, NULL)];

    // Multi-output: fp16 outputs
    // y=W2 output, h1=W1, h3=W3, gt=gate, xn=RMSNorm
    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp16, [1,%d,1,%d]> x", d, s];

    return orion_mil_program_multi(body, @[input_decl],
        @[@"w2_out", @"w1_out", @"w3_out", @"gate", @"rms2_out"]);
}

#pragma mark - T066: ffnBwd — FFN Backward

NSString* orion_milgen_ffn_bwd(int layer_idx, const OrionModelConfig* cfg) {
    int d  = cfg->d_model;     // 768
    int h  = cfg->hidden_dim;  // 2048
    int s  = cfg->max_seq;     // 256

    // Transposed weight blob paths
    NSString *w2t = [NSString stringWithFormat:@"@model_path/layer%d/w2t.bin", layer_idx];
    NSString *w1t = [NSString stringWithFormat:@"@model_path/layer%d/w1t.bin", layer_idx];
    NSString *w3t = [NSString stringWithFormat:@"@model_path/layer%d/w3t.bin", layer_idx];

    NSMutableString *body = [NSMutableString string];

    // Input is concat: [dffn(d), h1(h), h3(h)] → total d + 2*h channels
    // Slice input to extract components
    int total_in = d + 2*h;

    // Slice dffn: channels [0, d)
    [body appendFormat:
        @"        tensor<int32, [4]> sl_dffn_begin = const()[name=string(\"sl_dffn_begin\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [4]> sl_dffn_end = const()[name=string(\"sl_dffn_end\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> dffn = slice_by_index(begin=sl_dffn_begin, end=sl_dffn_end, x=inp)[name=string(\"dffn\")];\n",
        d, s, d, s];

    // Slice h1: channels [d, d+h)
    [body appendFormat:
        @"        tensor<int32, [4]> sl_h1_begin = const()[name=string(\"sl_h1_begin\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
         "        tensor<int32, [4]> sl_h1_end = const()[name=string(\"sl_h1_end\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> h1 = slice_by_index(begin=sl_h1_begin, end=sl_h1_end, x=inp)[name=string(\"h1\")];\n",
        d, d+h, s, h, s];

    // Slice h3: channels [d+h, d+2*h)
    [body appendFormat:
        @"        tensor<int32, [4]> sl_h3_begin = const()[name=string(\"sl_h3_begin\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
         "        tensor<int32, [4]> sl_h3_end = const()[name=string(\"sl_h3_end\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> h3 = slice_by_index(begin=sl_h3_begin, end=sl_h3_end, x=inp)[name=string(\"h3\")];\n",
        d+h, d+2*h, s, h, s];

    // 1. Backprop through W2: dsilu = W2^T @ dffn
    [body appendString:orion_mil_linear("w2t", "dffn", d, h, s, w2t.UTF8String, NULL)];

    // 2. SiLU backward
    // sig = sigmoid(h1)
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sig\")];\n", h, s];
    // silu = h1 * sig (recompute)
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1, y=sig)[name=string(\"silu\")];\n", h, s];
    // oms = 1 - sig
    [body appendFormat:
        @"        fp16 one_c = const()[name=string(\"one_c\"), val=fp16(1.0)];\n"
         "        tensor<fp16, [1,%d,1,%d]> oms = sub(x=one_c, y=sig)[name=string(\"oms\")];\n", h, s];
    // homs = h1 * oms
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> homs = mul(x=h1, y=oms)[name=string(\"homs\")];\n", h, s];
    // brk = 1 + homs = 1 + h1*(1-sig)
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> brk = add(x=one_c, y=homs)[name=string(\"brk\")];\n", h, s];
    // dsd = sig * brk = sig * (1 + h1*(1-sig)) = d(SiLU)/dh1
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> dsd = mul(x=sig, y=brk)[name=string(\"dsd\")];\n", h, s];

    // 3. Separate gates
    // dh1 = dsilu * h3 * dsd (chain rule through gating + SiLU)
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> dsh3 = mul(x=w2t_out, y=h3)[name=string(\"dsh3\")];\n", h, s];
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> dh1 = mul(x=dsh3, y=dsd)[name=string(\"dh1\")];\n", h, s];
    // dh3 = dsilu * silu(h1)
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> dh3 = mul(x=w2t_out, y=silu)[name=string(\"dh3\")];\n", h, s];

    // 4. Backprop through W1, W3 to get dx
    [body appendString:orion_mil_linear("w1t", "dh1", h, d, s, w1t.UTF8String, NULL)];
    [body appendString:orion_mil_linear("w3t", "dh3", h, d, s, w3t.UTF8String, NULL)];

    // dx = dx1 + dx3
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> dx = add(x=w1t_out, y=w3t_out)[name=string(\"dx\")];\n", d, s];

    // Multi-output: fp16 outputs
    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp16, [1,%d,1,%d]> inp", total_in, s];

    return orion_mil_program_multi(body, @[input_decl],
        @[@"dx", @"dh1", @"dh3"]);
}

#pragma mark - T067: sdpaBwd1 — SDPA Backward Part 1

NSString* orion_milgen_sdpa_bwd1(int layer_idx, const OrionModelConfig* cfg) {
    int d  = cfg->d_model;   // 768
    int nh = cfg->n_head;    // 12
    int hd = cfg->head_dim;  // 64
    int s  = cfg->max_seq;   // 256
    int sc_ch = nh * s;      // 3072 (score channels when flattened)
    float scale = 1.0f / sqrtf((float)hd);

    // Transposed weight
    NSString *wot = [NSString stringWithFormat:@"@model_path/layer%d/wot.bin", layer_idx];
    NSString *mask = [NSString stringWithFormat:@"@model_path/masks/causal_%d.bin", s];

    NSMutableString *body = [NSMutableString string];

    // Input: concat [qf(d), kf(d), vf(d), dx2f(d)] = 4*d channels
    int total_in = 4 * d;

    // Slice inputs
    [body appendFormat:
        @"        tensor<int32, [4]> sl0_b = const()[name=string(\"sl0_b\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [4]> sl0_e = const()[name=string(\"sl0_e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> qf = slice_by_index(begin=sl0_b, end=sl0_e, x=inp)[name=string(\"qf\")];\n",
        d, s, d, s];

    [body appendFormat:
        @"        tensor<int32, [4]> sl1_b = const()[name=string(\"sl1_b\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
         "        tensor<int32, [4]> sl1_e = const()[name=string(\"sl1_e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> kf = slice_by_index(begin=sl1_b, end=sl1_e, x=inp)[name=string(\"kf\")];\n",
        d, 2*d, s, d, s];

    [body appendFormat:
        @"        tensor<int32, [4]> sl2_b = const()[name=string(\"sl2_b\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
         "        tensor<int32, [4]> sl2_e = const()[name=string(\"sl2_e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> vf = slice_by_index(begin=sl2_b, end=sl2_e, x=inp)[name=string(\"vf\")];\n",
        2*d, 3*d, s, d, s];

    [body appendFormat:
        @"        tensor<int32, [4]> sl3_b = const()[name=string(\"sl3_b\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
         "        tensor<int32, [4]> sl3_e = const()[name=string(\"sl3_e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> dx2f = slice_by_index(begin=sl3_b, end=sl3_e, x=inp)[name=string(\"dx2f\")];\n",
        3*d, 4*d, s, d, s];

    // 1. Backprop through Wo: df = Wo^T @ dx2f
    [body appendString:orion_mil_linear("wot", "dx2f", d, d, s, wot.UTF8String, NULL)];

    // 2. Reshape Q, K, V, df to multi-head: [1, d, 1, s] → [1, nh, hd, s] → [1, nh, s, hd]
    [body appendFormat:
        @"        tensor<int32, [4]> rsh4d = const()[name=string(\"rsh4d\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n"
         "        tensor<int32, [4]> perm_t = const()[name=string(\"perm_t\"), val=tensor<int32, [4]>([0,1,3,2])];\n",
        nh, hd, s];

    // Reshape and transpose Q
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh4d, x=qf)[name=string(\"qr\")];\n"
         "        tensor<fp16, [1,%d,%d,%d]> q4 = transpose(perm=perm_t, x=qr)[name=string(\"q4\")];\n",
        nh, hd, s, nh, s, hd];

    // K
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh4d, x=kf)[name=string(\"kr\")];\n"
         "        tensor<fp16, [1,%d,%d,%d]> k4 = transpose(perm=perm_t, x=kr)[name=string(\"k4\")];\n",
        nh, hd, s, nh, s, hd];

    // V
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> vr = reshape(shape=rsh4d, x=vf)[name=string(\"vr\")];\n"
         "        tensor<fp16, [1,%d,%d,%d]> v4 = transpose(perm=perm_t, x=vr)[name=string(\"v4\")];\n",
        nh, hd, s, nh, s, hd];

    // df (Wo^T output)
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> dfr = reshape(shape=rsh4d, x=wot_out)[name=string(\"dfr\")];\n"
         "        tensor<fp16, [1,%d,%d,%d]> da = transpose(perm=perm_t, x=dfr)[name=string(\"da\")];\n",
        nh, hd, s, nh, s, hd];

    // 3. Recompute attention forward (need probs for backward)
    [body appendFormat:@"        bool txf = const()[name=string(\"txf\"), val=bool(false)];\n"];
    [body appendFormat:@"        bool txt = const()[name=string(\"txt\"), val=bool(true)];\n"];

    // scores = Q @ K^T * scale
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> scores = matmul(transpose_x=txf, transpose_y=txt, x=q4, y=k4)[name=string(\"scores\")];\n",
        nh, s, s];
    [body appendFormat:
        @"        fp16 sc_val = const()[name=string(\"sc_val\"), val=fp16(%f)];\n"
         "        tensor<fp16, [1,%d,%d,%d]> scores_sc = mul(x=scores, y=sc_val)[name=string(\"scores_sc\")];\n",
        scale, nh, s, s];

    // Causal mask
    [body appendFormat:
        @"        tensor<fp16, [1,1,%d,%d]> cmask = const()[name=string(\"cmask\"), "
         "val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"%@\"), offset=uint64(64)))];\n",
        s, s, s, s, mask];
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> masked = add(x=scores_sc, y=cmask)[name=string(\"masked\")];\n",
        nh, s, s];

    // Softmax → probs
    [body appendFormat:
        @"        int32 sm_ax = const()[name=string(\"sm_ax\"), val=int32(-1)];\n"
         "        tensor<fp16, [1,%d,%d,%d]> probs = softmax(axis=sm_ax, x=masked)[name=string(\"probs\")];\n",
        nh, s, s];

    // 4. dV = probs^T @ da → [1, nh, s, hd]
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> dv4 = matmul(transpose_x=txt, transpose_y=txf, x=probs, y=da)[name=string(\"dv4\")];\n",
        nh, s, hd];

    // dp = da @ V^T → [1, nh, s, s]
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> dp4 = matmul(transpose_x=txf, transpose_y=txt, x=da, y=v4)[name=string(\"dp4\")];\n",
        nh, s, s];

    // 5. Flatten outputs back to [1, C, 1, S]
    // dV: [1, nh, s, hd] → transpose [1, nh, hd, s] → reshape [1, d, 1, s]
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=perm_t, x=dv4)[name=string(\"dvt\")];\n"
         "        tensor<int32, [4]> rsh_d = const()[name=string(\"rsh_d\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=rsh_d, x=dvt)[name=string(\"dvf\")];\n",
        nh, hd, s, d, s, d, s];

    // probs: [1, nh, s, s] → reshape [1, nh*s, 1, s] = [1, sc_ch, 1, s]
    [body appendFormat:
        @"        tensor<int32, [4]> rsh_sc = const()[name=string(\"rsh_sc\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> pf = reshape(shape=rsh_sc, x=probs)[name=string(\"pf\")];\n",
        sc_ch, s, sc_ch, s];

    // dp: [1, nh, s, s] → reshape [1, sc_ch, 1, s]
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> dpf = reshape(shape=rsh_sc, x=dp4)[name=string(\"dpf\")];\n",
        sc_ch, s];

    // Multi-output: fp16 outputs
    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp16, [1,%d,1,%d]> inp", total_in, s];

    return orion_mil_program_multi(body, @[input_decl],
        @[@"dvf", @"pf", @"dpf"]);
}

#pragma mark - T068: sdpaBwd2 — SDPA Backward Part 2

NSString* orion_milgen_sdpa_bwd2(int layer_idx __attribute__((unused)), const OrionModelConfig* cfg) {
    int d  = cfg->d_model;   // 768
    int nh = cfg->n_head;    // 12
    int hd = cfg->head_dim;  // 64
    int s  = cfg->max_seq;   // 256
    int sc_ch = nh * s;      // 3072
    float scale = 1.0f / sqrtf((float)hd);

    NSMutableString *body = [NSMutableString string];

    // Input: concat [pf(sc_ch), dpf(sc_ch), qf(d), kf(d)]
    int total_in = 2 * sc_ch + 2 * d;

    // Slice inputs
    [body appendFormat:
        @"        tensor<int32, [4]> s0b = const()[name=string(\"s0b\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [4]> s0e = const()[name=string(\"s0e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> pf = slice_by_index(begin=s0b, end=s0e, x=inp)[name=string(\"pf\")];\n",
        sc_ch, s, sc_ch, s];

    [body appendFormat:
        @"        tensor<int32, [4]> s1b = const()[name=string(\"s1b\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
         "        tensor<int32, [4]> s1e = const()[name=string(\"s1e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> dpf = slice_by_index(begin=s1b, end=s1e, x=inp)[name=string(\"dpf\")];\n",
        sc_ch, 2*sc_ch, s, sc_ch, s];

    [body appendFormat:
        @"        tensor<int32, [4]> s2b = const()[name=string(\"s2b\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
         "        tensor<int32, [4]> s2e = const()[name=string(\"s2e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> qf = slice_by_index(begin=s2b, end=s2e, x=inp)[name=string(\"qf\")];\n",
        2*sc_ch, 2*sc_ch + d, s, d, s];

    [body appendFormat:
        @"        tensor<int32, [4]> s3b = const()[name=string(\"s3b\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
         "        tensor<int32, [4]> s3e = const()[name=string(\"s3e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> kf = slice_by_index(begin=s3b, end=s3e, x=inp)[name=string(\"kf\")];\n",
        2*sc_ch + d, total_in, s, d, s];

    // Reshape probs and dp from [1, sc_ch, 1, s] → [1, nh, s, s]
    [body appendFormat:
        @"        tensor<int32, [4]> rsh_sc = const()[name=string(\"rsh_sc\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n"
         "        tensor<fp16, [1,%d,%d,%d]> p4 = reshape(shape=rsh_sc, x=pf)[name=string(\"p4\")];\n"
         "        tensor<fp16, [1,%d,%d,%d]> dp4 = reshape(shape=rsh_sc, x=dpf)[name=string(\"dp4\")];\n",
        nh, s, s, nh, s, s, nh, s, s];

    // Reshape Q, K from [1, d, 1, s] → [1, nh, hd, s] → [1, nh, s, hd]
    [body appendFormat:
        @"        tensor<int32, [4]> rsh4d = const()[name=string(\"rsh4d\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n"
         "        tensor<int32, [4]> perm_t = const()[name=string(\"perm_t\"), val=tensor<int32, [4]>([0,1,3,2])];\n",
        nh, hd, s];

    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh4d, x=qf)[name=string(\"qr\")];\n"
         "        tensor<fp16, [1,%d,%d,%d]> q4 = transpose(perm=perm_t, x=qr)[name=string(\"q4\")];\n",
        nh, hd, s, nh, s, hd];

    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh4d, x=kf)[name=string(\"kr\")];\n"
         "        tensor<fp16, [1,%d,%d,%d]> k4 = transpose(perm=perm_t, x=kr)[name=string(\"k4\")];\n",
        nh, hd, s, nh, s, hd];

    // Softmax backward: ds = probs * (dp - sum(probs * dp, axis=-1, keepdims=true))
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> pdp = mul(x=p4, y=dp4)[name=string(\"pdp\")];\n",
        nh, s, s];
    [body appendFormat:
        @"        tensor<int32, [1]> ax_neg1 = const()[name=string(\"ax_neg1\"), val=tensor<int32, [1]>([-1])];\n"
         "        bool kd_true = const()[name=string(\"kd_true\"), val=bool(true)];\n"
         "        tensor<fp16, [1,%d,%d,1]> spdp = reduce_sum(x=pdp, axes=ax_neg1, keep_dims=kd_true)[name=string(\"spdp\")];\n",
        nh, s];
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> dps = sub(x=dp4, y=spdp)[name=string(\"dps\")];\n",
        nh, s, s];
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> ds_raw = mul(x=p4, y=dps)[name=string(\"ds_raw\")];\n",
        nh, s, s];

    // Scale: ds = ds_raw * scale
    [body appendFormat:
        @"        fp16 sc_val = const()[name=string(\"sc_val\"), val=fp16(%f)];\n"
         "        tensor<fp16, [1,%d,%d,%d]> ds = mul(x=ds_raw, y=sc_val)[name=string(\"ds\")];\n",
        scale, nh, s, s];

    // dQ = ds @ K → [1, nh, s, hd]
    [body appendFormat:
        @"        bool txf = const()[name=string(\"txf\"), val=bool(false)];\n"
         "        tensor<fp16, [1,%d,%d,%d]> dq4 = matmul(transpose_x=txf, transpose_y=txf, x=ds, y=k4)[name=string(\"dq4\")];\n",
        nh, s, hd];

    // dK = ds^T @ Q → [1, nh, s, hd]
    [body appendFormat:
        @"        bool txt = const()[name=string(\"txt\"), val=bool(true)];\n"
         "        tensor<fp16, [1,%d,%d,%d]> dk4 = matmul(transpose_x=txt, transpose_y=txf, x=ds, y=q4)[name=string(\"dk4\")];\n",
        nh, s, hd];

    // Reshape dQ, dK back: [1, nh, s, hd] → [1, nh, hd, s] → [1, d, 1, s]
    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> dqt = transpose(perm=perm_t, x=dq4)[name=string(\"dqt\")];\n"
         "        tensor<int32, [4]> rsh_d = const()[name=string(\"rsh_d\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> dqf = reshape(shape=rsh_d, x=dqt)[name=string(\"dqf\")];\n",
        nh, hd, s, d, s, d, s];

    [body appendFormat:
        @"        tensor<fp16, [1,%d,%d,%d]> dkt = transpose(perm=perm_t, x=dk4)[name=string(\"dkt\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> dkf = reshape(shape=rsh_d, x=dkt)[name=string(\"dkf\")];\n",
        nh, hd, s, d, s];

    // Multi-output: fp16 outputs
    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp16, [1,%d,1,%d]> inp", total_in, s];

    return orion_mil_program_multi(body, @[input_decl],
        @[@"dqf", @"dkf"]);
}

#pragma mark - T069: qkvBwd — QKV Backward

NSString* orion_milgen_qkv_bwd(int layer_idx, const OrionModelConfig* cfg) {
    int d = cfg->d_model;  // 768
    int s = cfg->max_seq;  // 256

    // Transposed weights
    NSString *wqt = [NSString stringWithFormat:@"@model_path/layer%d/wqt.bin", layer_idx];
    NSString *wkt = [NSString stringWithFormat:@"@model_path/layer%d/wkt.bin", layer_idx];
    NSString *wvt = [NSString stringWithFormat:@"@model_path/layer%d/wvt.bin", layer_idx];

    NSMutableString *body = [NSMutableString string];

    // Input: concat [dq(d), dk(d), dv(d)] = 3*d channels
    int total_in = 3 * d;

    // Slice inputs
    [body appendFormat:
        @"        tensor<int32, [4]> s0b = const()[name=string(\"s0b\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "        tensor<int32, [4]> s0e = const()[name=string(\"s0e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> dq = slice_by_index(begin=s0b, end=s0e, x=inp)[name=string(\"dq\")];\n",
        d, s, d, s];

    [body appendFormat:
        @"        tensor<int32, [4]> s1b = const()[name=string(\"s1b\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
         "        tensor<int32, [4]> s1e = const()[name=string(\"s1e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> dk = slice_by_index(begin=s1b, end=s1e, x=inp)[name=string(\"dk\")];\n",
        d, 2*d, s, d, s];

    [body appendFormat:
        @"        tensor<int32, [4]> s2b = const()[name=string(\"s2b\"), val=tensor<int32, [4]>([0,%d,0,0])];\n"
         "        tensor<int32, [4]> s2e = const()[name=string(\"s2e\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n"
         "        tensor<fp16, [1,%d,1,%d]> dv = slice_by_index(begin=s2b, end=s2e, x=inp)[name=string(\"dv\")];\n",
        2*d, 3*d, s, d, s];

    // Backprop each projection
    [body appendString:orion_mil_linear("wqt", "dq", d, d, s, wqt.UTF8String, NULL)];
    [body appendString:orion_mil_linear("wkt", "dk", d, d, s, wkt.UTF8String, NULL)];
    [body appendString:orion_mil_linear("wvt", "dv", d, d, s, wvt.UTF8String, NULL)];

    // Sum: dx = dxq + dxk + dxv
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> dx_qk = add(x=wqt_out, y=wkt_out)[name=string(\"dx_qk\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> dx = add(x=dx_qk, y=wvt_out)[name=string(\"dx\")];\n",
        d, s, d, s];

    // Single fp16 output
    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp16, [1,%d,1,%d]> inp", total_in, s];

    return orion_mil_program(body, @[input_decl], @"dx");
}
