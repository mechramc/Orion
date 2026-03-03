#import "gpt2_prefill_ffn.milgen.h"

// T048: MIL generator for GPT-2 FFN prefill kernel
//
// Composes mil_builder helpers into a complete per-layer FFN program.
// Single output: hidden state after LN2 → FC → GELU → Proj → Residual.

NSString* orion_milgen_gpt2_prefill_ffn(int layer_idx, int seq_len,
                                         const OrionModelConfig* cfg) {
    int d = cfg->d_model;     // 768
    int h = cfg->hidden_dim;  // 3072
    int s = seq_len;

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
        d, s];

    // LayerNorm 2
    [body appendString:orion_mil_layernorm("ln2", "x16", d, s,
                                            ln2_g.UTF8String, ln2_b.UTF8String, 1e-5f)];

    // FC up: d_model → hidden_dim (768 → 3072)
    [body appendString:orion_mil_linear("fc", "ln2_out", d, h, s,
                                         wfc.UTF8String, bfc.UTF8String)];

    // GELU activation
    [body appendString:orion_mil_gelu("act", "fc_out", h, s)];

    // Projection down: hidden_dim → d_model (3072 → 768)
    [body appendString:orion_mil_linear("ffnproj", "act_out", h, d, s,
                                         wproj.UTF8String, bproj.UTF8String)];

    // Residual: hidden = x16 + ffnproj_out
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> resid = add(x=x16, y=ffnproj_out)"
         "[name=string(\"resid\")];\n", d, s];

    // Cast output to fp32
    [body appendFormat:
        @"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> hidden = cast(dtype=to32, x=resid)"
         "[name=string(\"hidden\")];\n",
        d, s];

    // Wrap in single-output program
    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp32, [1,%d,1,%d]> x", d, s];

    return orion_mil_program(body, @[input_decl], @"hidden");
}
