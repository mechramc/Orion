#import "mil_builder.h"

// TODO(M0/M2): Implement MIL text generation helpers
// MIL reference: Apple's Model Intermediate Language for ANE compilation
// All tensors use fp16 [1, C, 1, S] layout

NSString* orion_mil_linear(const char* name, int in_dim, int out_dim,
                           const char* weight_blob_ref, const char* bias_blob_ref) {
    // TODO: Generate MIL linear op with BLOBFILE weight references
    return @"";
}

NSString* orion_mil_layernorm(const char* name, int dim,
                              const char* gamma_ref, const char* beta_ref, float eps) {
    // TODO: Generate MIL layer_norm op
    return @"";
}

NSString* orion_mil_rmsnorm(const char* name, int dim,
                            const char* weight_ref, float eps) {
    // TODO: Generate MIL RMS norm (for Llama/Stories model)
    return @"";
}

NSString* orion_mil_gelu(const char* name) {
    // TODO: Generate MIL GELU activation
    return @"";
}

NSString* orion_mil_silu(const char* name) {
    // TODO: Generate MIL SiLU activation (for Llama/Stories model)
    return @"";
}

NSString* orion_mil_causal_attention(const char* name, int n_head, int head_dim, int seq_len) {
    // TODO: Generate explicit causal attention (NOT using SDPA — ANE ignores masks)
    // Must decompose: Q@K^T / sqrt(d) → apply causal mask → softmax → @V
    return @"";
}

NSString* orion_mil_program(NSString* body, NSArray<NSString*>* inputs,
                            NSArray<NSString*>* outputs) {
    // TODO: Wrap body in MIL program with function signature
    return @"";
}
