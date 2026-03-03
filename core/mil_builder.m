#import "mil_builder.h"
#import <math.h>

// T019: orion_mil_linear (conv-based)
// T020: orion_mil_layernorm, orion_mil_rmsnorm
// T021: orion_mil_gelu, orion_mil_silu
// T022: orion_mil_causal_attention (decomposed, no SDPA)
// T023: orion_mil_program (wrapper)

#pragma mark - Constants

// Shared conv parameters (used by linear layers)
#define CONV_PARAMS \
    "        string %s_pt = const()[name=string(\"%s_pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> %s_st = const()[name=string(\"%s_st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> %s_pd = const()[name=string(\"%s_pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> %s_dl = const()[name=string(\"%s_dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 %s_gr = const()[name=string(\"%s_gr\"), val=int32(1)];\n"

#pragma mark - T023: Header + Program Wrapper

NSString* orion_mil_header(void) {
    return @"program(1.3)\n"
           "[buildInfo = dict<string, string>({"
           "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
           "{\"coremlc-version\", \"3505.4.1\"}, "
           "{\"coremltools-component-milinternal\", \"\"}, "
           "{\"coremltools-version\", \"9.0\"}"
           "})]\n";
}

NSString* orion_mil_program_multi(NSString* body, NSArray<NSString*>* inputs, NSArray<NSString*>* output_vars) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:orion_mil_header()];
    [m appendString:@"{\n"];

    // Function signature
    [m appendString:@"    func main<ios18>("];
    for (NSUInteger i = 0; i < inputs.count; i++) {
        if (i > 0) [m appendString:@", "];
        [m appendString:inputs[i]];
    }
    [m appendString:@") {\n"];

    // Body
    [m appendString:body];

    // Return
    [m appendString:@"    } -> ("];
    for (NSUInteger i = 0; i < output_vars.count; i++) {
        if (i > 0) [m appendString:@", "];
        [m appendString:output_vars[i]];
    }
    [m appendString:@");\n"];
    [m appendString:@"}\n"];
    return m;
}

NSString* orion_mil_program(NSString* body, NSArray<NSString*>* inputs, NSString* output_var) {
    return orion_mil_program_multi(body, inputs, @[output_var]);
}

#pragma mark - T019: Linear (1×1 Conv)

NSString* orion_mil_linear(const char* prefix, const char* input,
                           int in_dim, int out_dim, int seq,
                           const char* weight_path, const char* bias_path) {
    NSString *p = @(prefix);
    NSString *inp = @(input);

    NSMutableString *m = [NSMutableString string];

    // Conv constants
    [m appendFormat:@"        string %@_pt = const()[name=string(\"%@_pt\"), val=string(\"valid\")];\n", p, p];
    [m appendFormat:@"        tensor<int32, [2]> %@_st = const()[name=string(\"%@_st\"), val=tensor<int32, [2]>([1,1])];\n", p, p];
    [m appendFormat:@"        tensor<int32, [4]> %@_pd = const()[name=string(\"%@_pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n", p, p];
    [m appendFormat:@"        tensor<int32, [2]> %@_dl = const()[name=string(\"%@_dl\"), val=tensor<int32, [2]>([1,1])];\n", p, p];
    [m appendFormat:@"        int32 %@_gr = const()[name=string(\"%@_gr\"), val=int32(1)];\n", p, p];

    // Weight: [out_dim, in_dim, 1, 1] for 1×1 conv
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> %@_W = const()[name=string(\"%@_W\"), "
     "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"%s\"), offset=uint64(64)))];\n",
     out_dim, in_dim, p, p, out_dim, in_dim, weight_path];

    // Conv op
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_conv = conv("
     "dilations=%@_dl, groups=%@_gr, pad=%@_pd, pad_type=%@_pt, strides=%@_st, "
     "weight=%@_W, x=%@)[name=string(\"%@_conv\")];\n",
     out_dim, seq, p, p, p, p, p, p, p, inp, p];

    if (bias_path) {
        // Bias: [1, out_dim, 1, 1]
        [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> %@_b = const()[name=string(\"%@_b\"), "
         "val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"%s\"), offset=uint64(64)))];\n",
         out_dim, p, p, out_dim, bias_path];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_out = add(x=%@_conv, y=%@_b)[name=string(\"%@_out\")];\n",
         out_dim, seq, p, p, p, p];
    } else {
        // No bias — alias conv output as the output
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_out = identity(x=%@_conv)[name=string(\"%@_out\")];\n",
         out_dim, seq, p, p, p];
    }

    return m;
}

#pragma mark - T020: LayerNorm

NSString* orion_mil_layernorm(const char* prefix, const char* input,
                              int dim, int seq,
                              const char* gamma_path, const char* beta_path, float eps) {
    NSString *p = @(prefix);
    NSString *inp = @(input);
    NSMutableString *m = [NSMutableString string];

    // mean = reduce_mean(x, axis=1)
    [m appendFormat:@"        tensor<int32, [1]> %@_ax = const()[name=string(\"%@_ax\"), val=tensor<int32, [1]>([1])];\n", p, p];
    [m appendFormat:@"        bool %@_kd = const()[name=string(\"%@_kd\"), val=bool(true)];\n", p, p];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> %@_mean = reduce_mean(x=%@, axes=%@_ax, keep_dims=%@_kd)[name=string(\"%@_mean\")];\n",
     seq, p, inp, p, p, p];

    // x_centered = x - mean
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_cent = sub(x=%@, y=%@_mean)[name=string(\"%@_cent\")];\n",
     dim, seq, p, inp, p, p];

    // var = reduce_mean(x_centered^2, axis=1)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_sq = mul(x=%@_cent, y=%@_cent)[name=string(\"%@_sq\")];\n",
     dim, seq, p, p, p, p];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> %@_var = reduce_mean(x=%@_sq, axes=%@_ax, keep_dims=%@_kd)[name=string(\"%@_var\")];\n",
     seq, p, p, p, p, p];

    // rsqrt(var + eps)
    [m appendFormat:@"        fp16 %@_eps = const()[name=string(\"%@_eps\"), val=fp16(%f)];\n", p, p, eps];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> %@_veps = add(x=%@_var, y=%@_eps)[name=string(\"%@_veps\")];\n",
     seq, p, p, p, p];
    [m appendFormat:@"        fp16 %@_nhalf = const()[name=string(\"%@_nhalf\"), val=fp16(-0.5)];\n", p, p];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> %@_rstd = pow(x=%@_veps, y=%@_nhalf)[name=string(\"%@_rstd\")];\n",
     seq, p, p, p, p];

    // normalized = x_centered * rstd
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_norm = mul(x=%@_cent, y=%@_rstd)[name=string(\"%@_norm\")];\n",
     dim, seq, p, p, p, p];

    // gamma * normalized + beta
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> %@_g = const()[name=string(\"%@_g\"), "
     "val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"%s\"), offset=uint64(64)))];\n",
     dim, p, p, dim, gamma_path];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_scaled = mul(x=%@_norm, y=%@_g)[name=string(\"%@_scaled\")];\n",
     dim, seq, p, p, p, p];

    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> %@_beta = const()[name=string(\"%@_beta\"), "
     "val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"%s\"), offset=uint64(64)))];\n",
     dim, p, p, dim, beta_path];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_out = add(x=%@_scaled, y=%@_beta)[name=string(\"%@_out\")];\n",
     dim, seq, p, p, p, p];

    return m;
}

#pragma mark - T020: RMSNorm

NSString* orion_mil_rmsnorm(const char* prefix, const char* input,
                            int dim, int seq,
                            const char* weight_path, float eps) {
    NSString *p = @(prefix);
    NSString *inp = @(input);
    NSMutableString *m = [NSMutableString string];
    float inv_dim = 1.0f / (float)dim;

    // x^2
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_sq = mul(x=%@, y=%@)[name=string(\"%@_sq\")];\n",
     dim, seq, p, inp, inp, p];

    // reduce_sum(x^2, axis=1) / dim
    [m appendFormat:@"        tensor<int32, [1]> %@_ax = const()[name=string(\"%@_ax\"), val=tensor<int32, [1]>([1])];\n", p, p];
    [m appendFormat:@"        bool %@_kd = const()[name=string(\"%@_kd\"), val=bool(true)];\n", p, p];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> %@_ss = reduce_sum(x=%@_sq, axes=%@_ax, keep_dims=%@_kd)[name=string(\"%@_ss\")];\n",
     seq, p, p, p, p, p];
    [m appendFormat:@"        fp16 %@_invd = const()[name=string(\"%@_invd\"), val=fp16(%f)];\n", p, p, inv_dim];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> %@_ms = mul(x=%@_ss, y=%@_invd)[name=string(\"%@_ms\")];\n",
     seq, p, p, p, p];

    // rsqrt(ms + eps)
    [m appendFormat:@"        fp16 %@_eps = const()[name=string(\"%@_eps\"), val=fp16(%f)];\n", p, p, eps];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> %@_mse = add(x=%@_ms, y=%@_eps)[name=string(\"%@_mse\")];\n",
     seq, p, p, p, p];
    [m appendFormat:@"        fp16 %@_nhalf = const()[name=string(\"%@_nhalf\"), val=fp16(-0.5)];\n", p, p];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> %@_rrms = pow(x=%@_mse, y=%@_nhalf)[name=string(\"%@_rrms\")];\n",
     seq, p, p, p, p];

    // x * rrms
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_xr = mul(x=%@, y=%@_rrms)[name=string(\"%@_xr\")];\n",
     dim, seq, p, inp, p, p];

    // weight * normalized
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> %@_w = const()[name=string(\"%@_w\"), "
     "val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"%s\"), offset=uint64(64)))];\n",
     dim, p, p, dim, weight_path];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_out = mul(x=%@_xr, y=%@_w)[name=string(\"%@_out\")];\n",
     dim, seq, p, p, p, p];

    return m;
}

#pragma mark - T021: GELU

NSString* orion_mil_gelu(const char* prefix, const char* input, int dim, int seq) {
    NSString *p = @(prefix);
    NSString *inp = @(input);
    NSMutableString *m = [NSMutableString string];

    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // Decomposed into primitive MIL ops for ANE compatibility.

    // x^3
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_x2 = mul(x=%@, y=%@)[name=string(\"%@_x2\")];\n",
     dim, seq, p, inp, inp, p];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_x3 = mul(x=%@_x2, y=%@)[name=string(\"%@_x3\")];\n",
     dim, seq, p, p, inp, p];

    // 0.044715 * x^3
    [m appendFormat:@"        fp16 %@_c1 = const()[name=string(\"%@_c1\"), val=fp16(0.044715)];\n", p, p];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_cx3 = mul(x=%@_x3, y=%@_c1)[name=string(\"%@_cx3\")];\n",
     dim, seq, p, p, p, p];

    // x + 0.044715 * x^3
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_inner = add(x=%@, y=%@_cx3)[name=string(\"%@_inner\")];\n",
     dim, seq, p, inp, p, p];

    // sqrt(2/pi) * (x + 0.044715*x^3)
    [m appendFormat:@"        fp16 %@_c2 = const()[name=string(\"%@_c2\"), val=fp16(0.7979)];\n", p, p]; // sqrt(2/pi)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_scaled = mul(x=%@_inner, y=%@_c2)[name=string(\"%@_scaled\")];\n",
     dim, seq, p, p, p, p];

    // tanh
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_th = tanh(x=%@_scaled)[name=string(\"%@_th\")];\n",
     dim, seq, p, p, p];

    // 1 + tanh(...)
    [m appendFormat:@"        fp16 %@_one = const()[name=string(\"%@_one\"), val=fp16(1.0)];\n", p, p];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_onep = add(x=%@_th, y=%@_one)[name=string(\"%@_onep\")];\n",
     dim, seq, p, p, p, p];

    // 0.5 * x
    [m appendFormat:@"        fp16 %@_half = const()[name=string(\"%@_half\"), val=fp16(0.5)];\n", p, p];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_hx = mul(x=%@, y=%@_half)[name=string(\"%@_hx\")];\n",
     dim, seq, p, inp, p, p];

    // 0.5 * x * (1 + tanh(...))
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_out = mul(x=%@_hx, y=%@_onep)[name=string(\"%@_out\")];\n",
     dim, seq, p, p, p, p];

    return m;
}

#pragma mark - T021: SiLU

NSString* orion_mil_silu(const char* prefix, const char* input, int dim, int seq) {
    NSString *p = @(prefix);
    NSString *inp = @(input);
    NSMutableString *m = [NSMutableString string];

    // SiLU(x) = x * sigmoid(x)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_sig = sigmoid(x=%@)[name=string(\"%@_sig\")];\n",
     dim, seq, p, inp, p];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_out = mul(x=%@, y=%@_sig)[name=string(\"%@_out\")];\n",
     dim, seq, p, inp, p, p];

    return m;
}

#pragma mark - T022: Causal Attention (Decomposed)

NSString* orion_mil_causal_attention(const char* prefix,
                                     const char* q_input, const char* k_input, const char* v_input,
                                     int n_head, int head_dim, int seq,
                                     const char* mask_path) {
    NSString *p = @(prefix);
    int d_model = n_head * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    NSMutableString *m = [NSMutableString string];

    // Reshape Q, K, V from [1, d_model, 1, seq] to [1, n_head, head_dim, seq]
    [m appendFormat:@"        tensor<int32, [4]> %@_rsh = const()[name=string(\"%@_rsh\"), "
     "val=tensor<int32, [4]>([1,%d,%d,%d])];\n", p, p, n_head, head_dim, seq];

    // Permutation: [0,1,3,2] → [1, n_head, seq, head_dim]
    [m appendFormat:@"        tensor<int32, [4]> %@_pm = const()[name=string(\"%@_pm\"), "
     "val=tensor<int32, [4]>([0,1,3,2])];\n", p, p];

    // Q: reshape then transpose
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_qr = reshape(shape=%@_rsh, x=%@)[name=string(\"%@_qr\")];\n",
     n_head, head_dim, seq, p, p, @(q_input), p];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_q = transpose(perm=%@_pm, x=%@_qr)[name=string(\"%@_q\")];\n",
     n_head, seq, head_dim, p, p, p, p];

    // K: reshape then transpose
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_kr = reshape(shape=%@_rsh, x=%@)[name=string(\"%@_kr\")];\n",
     n_head, head_dim, seq, p, p, @(k_input), p];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_k = transpose(perm=%@_pm, x=%@_kr)[name=string(\"%@_k\")];\n",
     n_head, seq, head_dim, p, p, p, p];

    // V: reshape then transpose
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_vr = reshape(shape=%@_rsh, x=%@)[name=string(\"%@_vr\")];\n",
     n_head, head_dim, seq, p, p, @(v_input), p];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_v = transpose(perm=%@_pm, x=%@_vr)[name=string(\"%@_v\")];\n",
     n_head, seq, head_dim, p, p, p, p];

    // scores = Q @ K^T  (matmul with transpose_y=true)
    [m appendFormat:@"        bool %@_txf = const()[name=string(\"%@_txf\"), val=bool(false)];\n", p, p];
    [m appendFormat:@"        bool %@_txt = const()[name=string(\"%@_txt\"), val=bool(true)];\n", p, p];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_sc = matmul(transpose_x=%@_txf, transpose_y=%@_txt, x=%@_q, y=%@_k)[name=string(\"%@_sc\")];\n",
     n_head, seq, seq, p, p, p, p, p, p];

    // scores = scores / sqrt(head_dim)
    [m appendFormat:@"        fp16 %@_scv = const()[name=string(\"%@_scv\"), val=fp16(%f)];\n", p, p, scale];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_scs = mul(x=%@_sc, y=%@_scv)[name=string(\"%@_scs\")];\n",
     n_head, seq, seq, p, p, p, p];

    // Apply causal mask (additive: -inf for future positions)
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> %@_mask = const()[name=string(\"%@_mask\"), "
     "val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"%s\"), offset=uint64(64)))];\n",
     seq, seq, p, p, seq, seq, mask_path];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_masked = add(x=%@_scs, y=%@_mask)[name=string(\"%@_masked\")];\n",
     n_head, seq, seq, p, p, p, p];

    // Softmax along last axis
    [m appendFormat:@"        int32 %@_sax = const()[name=string(\"%@_sax\"), val=int32(-1)];\n", p, p];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_attn = softmax(axis=%@_sax, x=%@_masked)[name=string(\"%@_attn\")];\n",
     n_head, seq, seq, p, p, p, p];

    // context = attn @ V
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_ctx = matmul(transpose_x=%@_txf, transpose_y=%@_txf, x=%@_attn, y=%@_v)[name=string(\"%@_ctx\")];\n",
     n_head, seq, head_dim, p, p, p, p, p, p];

    // Transpose back: [1, n_head, seq, head_dim] → [1, n_head, head_dim, seq]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> %@_ctxt = transpose(perm=%@_pm, x=%@_ctx)[name=string(\"%@_ctxt\")];\n",
     n_head, head_dim, seq, p, p, p, p];

    // Reshape back to [1, d_model, 1, seq]
    [m appendFormat:@"        tensor<int32, [4]> %@_osh = const()[name=string(\"%@_osh\"), "
     "val=tensor<int32, [4]>([1,%d,1,%d])];\n", p, p, d_model, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> %@_out = reshape(shape=%@_osh, x=%@_ctxt)[name=string(\"%@_out\")];\n",
     d_model, seq, p, p, p, p];

    return m;
}
