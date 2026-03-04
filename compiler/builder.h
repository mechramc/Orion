// compiler/builder.h — T118: Graph builder API
// Fluent API that mirrors mil_builder but builds graph nodes instead of MIL text.
// Pure C.

#ifndef ORION_BUILDER_H
#define ORION_BUILDER_H

#include "graph.h"

// Input node — creates a graph input placeholder
int orion_gb_input(OrionGraph* g, const char* name, OrionDtype dtype, int shape[4]);

// Constant scalar
int orion_gb_const_scalar(OrionGraph* g, const char* name, OrionDtype dtype, float value);

// Constant tensor (weight reference via BLOBFILE)
int orion_gb_const_weight(OrionGraph* g, const char* name, OrionDtype dtype,
                          int shape[4], const char* blob_path, uint64_t offset);

// Constant int32 tensor (for reshape targets, axes, perms)
int orion_gb_const_int32(OrionGraph* g, const char* name, int shape[4],
                         const int* values, int n_values);

// Constant bool
int orion_gb_const_bool(OrionGraph* g, const char* name, bool value);

// 1x1 convolution (linear layer on ANE)
// bias can be -1 for no bias
int orion_gb_conv1x1(OrionGraph* g, int input, int weight, int bias, const char* name,
                     int out_channels, int seq);

// Elementwise ops
int orion_gb_add(OrionGraph* g, int a, int b, const char* name);
int orion_gb_sub(OrionGraph* g, int a, int b, const char* name);
int orion_gb_mul(OrionGraph* g, int a, int b, const char* name);

// MatMul
int orion_gb_matmul(OrionGraph* g, int a, int b, bool transpose_x, bool transpose_y,
                    const char* name, int out_shape[4]);

// Shape ops
int orion_gb_reshape(OrionGraph* g, int input, int shape_node, const char* name,
                     int out_shape[4]);
int orion_gb_transpose(OrionGraph* g, int input, int perm_node, const char* name,
                       int perm[4], int out_shape[4]);

// Cast
int orion_gb_cast(OrionGraph* g, int input, OrionDtype target_dtype,
                  const char* name, int out_shape[4]);

// Activations
int orion_gb_relu(OrionGraph* g, int input, const char* name);
int orion_gb_tanh(OrionGraph* g, int input, const char* name);
int orion_gb_sigmoid(OrionGraph* g, int input, const char* name);
int orion_gb_softmax(OrionGraph* g, int input, int axis, const char* name);

// Math
int orion_gb_exp(OrionGraph* g, int input, const char* name);
int orion_gb_pow(OrionGraph* g, int base, int exponent, const char* name);
int orion_gb_neg(OrionGraph* g, int input, const char* name);
int orion_gb_sqrt(OrionGraph* g, int input, const char* name);
int orion_gb_rsqrt(OrionGraph* g, int input, const char* name);
int orion_gb_identity(OrionGraph* g, int input, const char* name);

// Reduce ops
int orion_gb_reduce_sum(OrionGraph* g, int input, int axes, bool keep_dims,
                        const char* name, int out_shape[4]);
int orion_gb_reduce_mean(OrionGraph* g, int input, int axes, bool keep_dims,
                         const char* name, int out_shape[4]);

// Slice
int orion_gb_slice(OrionGraph* g, int input, int begin_node, int end_node,
                   const char* name, int out_shape[4]);

// Mark a node as a graph output
void orion_gb_output(OrionGraph* g, int node, const char* name);

// Composites — expand to multiple sub-ops

// LayerNorm: gamma * (x - mean) / sqrt(var + eps) + beta
// Returns the output node index
int orion_gb_layernorm(OrionGraph* g, int input, int gamma_weight, int beta_weight,
                       float eps, const char* prefix, int dim, int seq);

// GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
int orion_gb_gelu(OrionGraph* g, int input, const char* prefix, int dim, int seq);

// SiLU: x * sigmoid(x)
int orion_gb_silu(OrionGraph* g, int input, const char* prefix, int dim, int seq);

// RMSNorm: w * x * rsqrt(mean(x^2) + eps)
int orion_gb_rmsnorm(OrionGraph* g, int input, int weight, float eps,
                     const char* prefix, int dim, int seq);

// Linear layer (convenience: creates weight const + optional bias const + conv1x1)
int orion_gb_linear(OrionGraph* g, int input, const char* prefix,
                    int in_dim, int out_dim, int seq,
                    const char* weight_path, const char* bias_path);

#endif // ORION_BUILDER_H
