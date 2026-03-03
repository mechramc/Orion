#import "gpt2_decode_ane.milgen.h"

// T100: MIL generators for single-token ANE decode.
//
// decode_proj: same as prefill attention projections but seq=ORION_DECODE_SEQ.
// decode_ffn: same as prefill FFN but seq=ORION_DECODE_SEQ.
//
// ANE requires minimum ~49KB IOSurface allocation. Using seq=16 (minimum bucket)
// with token data at position 0, zero-padding at positions 1-15.

#pragma mark - Decode QKV Projections

NSString* orion_milgen_gpt2_decode_proj(int layer_idx, const OrionModelConfig* cfg) {
    int d = cfg->d_model;  // 768
    int seq = ORION_DECODE_SEQ;  // 16

    // Build blob paths for this layer
    NSString *ln1_g = [NSString stringWithFormat:@"@model_path/layer%d/ln1_g.bin", layer_idx];
    NSString *ln1_b = [NSString stringWithFormat:@"@model_path/layer%d/ln1_b.bin", layer_idx];
    NSString *wq    = [NSString stringWithFormat:@"@model_path/layer%d/wq.bin", layer_idx];
    NSString *bq    = [NSString stringWithFormat:@"@model_path/layer%d/bq.bin", layer_idx];
    NSString *wk    = [NSString stringWithFormat:@"@model_path/layer%d/wk.bin", layer_idx];
    NSString *bk    = [NSString stringWithFormat:@"@model_path/layer%d/bk.bin", layer_idx];
    NSString *wv    = [NSString stringWithFormat:@"@model_path/layer%d/wv.bin", layer_idx];
    NSString *bv    = [NSString stringWithFormat:@"@model_path/layer%d/bv.bin", layer_idx];

    NSMutableString *body = [NSMutableString string];

    // Cast input to fp16
    [body appendFormat:
        @"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16, x=x)[name=string(\"x16\")];\n",
        d, seq];

    // LayerNorm 1
    [body appendString:orion_mil_layernorm("ln1", "x16", d, seq,
                                            ln1_g.UTF8String, ln1_b.UTF8String, 1e-5f)];

    // Q, K, V projections (all [1, d_model, 1, seq])
    [body appendString:orion_mil_linear("q", "ln1_out", d, d, seq,
                                         wq.UTF8String, bq.UTF8String)];
    [body appendString:orion_mil_linear("k", "ln1_out", d, d, seq,
                                         wk.UTF8String, bk.UTF8String)];
    [body appendString:orion_mil_linear("v", "ln1_out", d, d, seq,
                                         wv.UTF8String, bv.UTF8String)];

    // Cast outputs to fp32
    [body appendFormat:
        @"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> q32 = cast(dtype=to32, x=q_out)"
         "[name=string(\"q32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> k32 = cast(dtype=to32, x=k_out)"
         "[name=string(\"k32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> v32 = cast(dtype=to32, x=v_out)"
         "[name=string(\"v32\")];\n",
        d, seq, d, seq, d, seq];

    // Wrap in multi-output program (3 outputs: q, k_new, v_new)
    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp32, [1,%d,1,%d]> x", d, seq];

    return orion_mil_program_multi(body, @[input_decl],
                                    @[@"q32", @"k32", @"v32"]);
}

#pragma mark - Decode FFN + Residual

NSString* orion_milgen_gpt2_decode_ffn(int layer_idx, const OrionModelConfig* cfg) {
    int d = cfg->d_model;     // 768
    int h = cfg->hidden_dim;  // 3072
    int seq = ORION_DECODE_SEQ;  // 16

    // Build blob paths
    NSString *ln2_g  = [NSString stringWithFormat:@"@model_path/layer%d/ln2_g.bin", layer_idx];
    NSString *ln2_b  = [NSString stringWithFormat:@"@model_path/layer%d/ln2_b.bin", layer_idx];
    NSString *wfc    = [NSString stringWithFormat:@"@model_path/layer%d/wfc.bin", layer_idx];
    NSString *bfc    = [NSString stringWithFormat:@"@model_path/layer%d/bfc.bin", layer_idx];
    NSString *wproj  = [NSString stringWithFormat:@"@model_path/layer%d/wproj.bin", layer_idx];
    NSString *bproj  = [NSString stringWithFormat:@"@model_path/layer%d/bproj.bin", layer_idx];

    NSMutableString *body = [NSMutableString string];

    // Cast input to fp16
    [body appendFormat:
        @"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16, x=x)[name=string(\"x16\")];\n",
        d, seq];

    // LayerNorm 2
    [body appendString:orion_mil_layernorm("ln2", "x16", d, seq,
                                            ln2_g.UTF8String, ln2_b.UTF8String, 1e-5f)];

    // FC up: d_model → hidden_dim (768 → 3072)
    [body appendString:orion_mil_linear("fc", "ln2_out", d, h, seq,
                                         wfc.UTF8String, bfc.UTF8String)];

    // GELU activation
    [body appendString:orion_mil_gelu("act", "fc_out", h, seq)];

    // Projection down: hidden_dim → d_model (3072 → 768)
    [body appendString:orion_mil_linear("ffnproj", "act_out", h, d, seq,
                                         wproj.UTF8String, bproj.UTF8String)];

    // Residual: hidden = x16 + ffnproj_out
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> resid = add(x=x16, y=ffnproj_out)"
         "[name=string(\"resid\")];\n", d, seq];

    // Cast output to fp32
    [body appendFormat:
        @"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> hidden = cast(dtype=to32, x=resid)"
         "[name=string(\"hidden\")];\n",
        d, seq];

    // Wrap in single-output program
    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp32, [1,%d,1,%d]> x", d, seq];

    return orion_mil_program(body, @[input_decl], @"hidden");
}
