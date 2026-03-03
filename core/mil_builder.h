#ifndef ORION_MIL_BUILDER_H
#define ORION_MIL_BUILDER_H

#import <Foundation/Foundation.h>
#import "ane_runtime.h"

/// Helper to generate MIL (Model Intermediate Language) text for ANE compilation.
/// MIL is Apple's intermediate representation for neural network operations.

/// Generate MIL text for a linear layer: Y = X @ W + b
/// Weights are referenced as BLOBFILE constants.
NSString* orion_mil_linear(const char* name, int in_dim, int out_dim,
                           const char* weight_blob_ref, const char* bias_blob_ref);

/// Generate MIL text for layer normalization.
NSString* orion_mil_layernorm(const char* name, int dim,
                              const char* gamma_ref, const char* beta_ref, float eps);

/// Generate MIL text for RMS normalization (Llama-style).
NSString* orion_mil_rmsnorm(const char* name, int dim,
                            const char* weight_ref, float eps);

/// Generate MIL text for GELU activation.
NSString* orion_mil_gelu(const char* name);

/// Generate MIL text for SiLU activation (Llama-style).
NSString* orion_mil_silu(const char* name);

/// Generate MIL text for causal attention (explicit decomposition, no SDPA).
/// Uses Q @ K^T / sqrt(d) with explicit causal mask, then softmax @ V.
NSString* orion_mil_causal_attention(const char* name, int n_head, int head_dim, int seq_len);

/// Wrap operations into a complete MIL program with function signature.
NSString* orion_mil_program(NSString* body, NSArray<NSString*>* inputs,
                            NSArray<NSString*>* outputs);

#endif // ORION_MIL_BUILDER_H
