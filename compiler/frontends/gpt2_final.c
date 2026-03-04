// compiler/frontends/gpt2_final.c — T141: GPT-2 final LayerNorm frontend

#include "gpt2_final.h"
#include "../builder.h"
#include "../patterns.h"

OrionGraph* orion_frontend_gpt2_final_ln(int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int s = bucket;
    OrionGraph* g = orion_graph_create();

    // Input: fp32 [1, d, 1, s]
    int in_shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);

    // Cast to fp16
    int x16 = orion_pattern_cast_to_fp16(g, x, "x16", d, s);

    // Final LayerNorm
    int ln_shape[4] = {1, d, 1, 1};
    int ln_g = orion_gb_const_weight(g, "lnf_g", ORION_DTYPE_FP16, ln_shape,
                                      "@model_path/ln_f_g.bin", 64);
    int ln_b = orion_gb_const_weight(g, "lnf_beta", ORION_DTYPE_FP16, ln_shape,
                                      "@model_path/ln_f_b.bin", 64);
    int lnf = orion_gb_layernorm(g, x16, ln_g, ln_b, 1e-5f, "lnf", d, s);

    // Cast to fp32
    int hidden = orion_pattern_cast_to_fp32(g, lnf, "hidden", d, s);

    orion_gb_output(g, hidden, "hidden");
    return g;
}
