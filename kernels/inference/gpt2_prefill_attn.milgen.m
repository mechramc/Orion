#import "gpt2_prefill_attn.milgen.h"
#import <math.h>

// T047: MIL generator for GPT-2 attention prefill kernel
//
// Composes mil_builder helpers into a complete per-layer attention program.
// Each layer produces 3 outputs: hidden state (after residual), K, V (for cache).

#pragma mark - Causal Mask

NSData* orion_make_causal_mask_blob(int seq_len) {
    int count = seq_len * seq_len;
    int data_bytes = count * sizeof(_Float16);
    int total = 128 + data_bytes;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    // BLOBFILE header
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = data_bytes;
    *(uint32_t *)(buf + 80) = 128;

    // Fill causal mask: 0 for j <= i, -1e4 for j > i
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            fp16[i * seq_len + j] = (j <= i) ? (_Float16)0.0f : (_Float16)(-1e4f);
        }
    }

    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

NSString* orion_causal_mask_path(int seq_len) {
    return [NSString stringWithFormat:@"@model_path/masks/causal_%d.bin", seq_len];
}

#pragma mark - Attention Prefill Kernel

NSString* orion_milgen_gpt2_prefill_attn(int layer_idx, int seq_len,
                                          const OrionModelConfig* cfg) {
    int d  = cfg->d_model;   // 768
    int nh = cfg->n_head;    // 12
    int hd = cfg->head_dim;  // 64
    int s  = seq_len;

    // Build blob paths for this layer
    NSString *ln1_g = [NSString stringWithFormat:@"@model_path/layer%d/ln1_g.bin", layer_idx];
    NSString *ln1_b = [NSString stringWithFormat:@"@model_path/layer%d/ln1_b.bin", layer_idx];
    NSString *wq    = [NSString stringWithFormat:@"@model_path/layer%d/wq.bin", layer_idx];
    NSString *bq    = [NSString stringWithFormat:@"@model_path/layer%d/bq.bin", layer_idx];
    NSString *wk    = [NSString stringWithFormat:@"@model_path/layer%d/wk.bin", layer_idx];
    NSString *bk    = [NSString stringWithFormat:@"@model_path/layer%d/bk.bin", layer_idx];
    NSString *wv    = [NSString stringWithFormat:@"@model_path/layer%d/wv.bin", layer_idx];
    NSString *bv    = [NSString stringWithFormat:@"@model_path/layer%d/bv.bin", layer_idx];
    NSString *wo    = [NSString stringWithFormat:@"@model_path/layer%d/wo.bin", layer_idx];
    NSString *bo    = [NSString stringWithFormat:@"@model_path/layer%d/bo.bin", layer_idx];
    NSString *mask  = orion_causal_mask_path(seq_len);

    NSMutableString *body = [NSMutableString string];

    // Cast input to fp16
    [body appendFormat:
        @"        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
         "        tensor<fp16, [1,%d,1,%d]> x16 = cast(dtype=to16, x=x)[name=string(\"x16\")];\n",
        d, s];

    // LayerNorm 1
    [body appendString:orion_mil_layernorm("ln1", "x16", d, s,
                                            ln1_g.UTF8String, ln1_b.UTF8String, 1e-5f)];

    // Q, K, V projections (all [1, d_model, 1, seq])
    [body appendString:orion_mil_linear("q", "ln1_out", d, d, s,
                                         wq.UTF8String, bq.UTF8String)];
    [body appendString:orion_mil_linear("k", "ln1_out", d, d, s,
                                         wk.UTF8String, bk.UTF8String)];
    [body appendString:orion_mil_linear("v", "ln1_out", d, d, s,
                                         wv.UTF8String, bv.UTF8String)];

    // Causal attention (decomposed: Q@K^T → mask → softmax → @V)
    [body appendString:orion_mil_causal_attention("attn", "q_out", "k_out", "v_out",
                                                   nh, hd, s, mask.UTF8String)];

    // Output projection
    [body appendString:orion_mil_linear("proj", "attn_out", d, d, s,
                                         wo.UTF8String, bo.UTF8String)];

    // Residual: hidden = x16 + proj_out
    [body appendFormat:
        @"        tensor<fp16, [1,%d,1,%d]> resid = add(x=x16, y=proj_out)"
         "[name=string(\"resid\")];\n", d, s];

    // Cast outputs to fp32
    [body appendFormat:
        @"        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> hidden = cast(dtype=to32, x=resid)"
         "[name=string(\"hidden\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> k_cache = cast(dtype=to32, x=k_out)"
         "[name=string(\"k_cache\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> v_cache = cast(dtype=to32, x=v_out)"
         "[name=string(\"v_cache\")];\n",
        d, s, d, s, d, s];

    // Wrap in multi-output program
    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp32, [1,%d,1,%d]> x", d, s];

    return orion_mil_program_multi(body, @[input_decl],
                                    @[@"hidden", @"k_cache", @"v_cache"]);
}
