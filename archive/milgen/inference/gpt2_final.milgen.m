#import "gpt2_final.milgen.h"

// T049: Final LayerNorm MIL kernel for GPT-2
//
// Logits projection (hidden @ wte^T) done on CPU because wte is 73MB.

NSString* orion_milgen_gpt2_final_ln(int seq_len, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int s = seq_len;

    NSMutableString *body = [NSMutableString string];

    // Cast to fp16
    [body appendFormat:
        @"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16, x=x)[name=string(\"x16\")];\n",
        d, s];

    // Final LayerNorm
    [body appendString:orion_mil_layernorm("lnf", "x16", d, s,
                                            "@model_path/ln_f_g.bin",
                                            "@model_path/ln_f_b.bin", 1e-5f)];

    // Cast to fp32
    [body appendFormat:
        @"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> hidden = cast(dtype=to32, x=lnf_out)"
         "[name=string(\"hidden\")];\n",
        d, s];

    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp32, [1,%d,1,%d]> x", d, s];

    return orion_mil_program(body, @[input_decl], @"hidden");
}
