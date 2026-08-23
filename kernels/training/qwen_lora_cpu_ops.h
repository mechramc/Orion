#ifndef ORION_QWEN_LORA_CPU_OPS_H
#define ORION_QWEN_LORA_CPU_OPS_H

#import <Foundation/Foundation.h>

typedef struct OrionQwenStreamingCEContext OrionQwenStreamingCEContext;

typedef struct {
    int in_dim;
    int out_dim;
    int rank;
    float alpha;
    float scale;
    float *a;
    float *b;
    float *da;
    float *db;
    float *ma;
    float *va;
    float *mb;
    float *vb;
} OrionLoRAMatrix;

void orion_qwen_lora_matrix_init(OrionLoRAMatrix *mat,
                                 int in_dim,
                                 int out_dim,
                                 int rank,
                                 float alpha,
                                 unsigned int seed);

void orion_qwen_lora_matrix_zero_grad(OrionLoRAMatrix *mat);

void orion_qwen_lora_matrix_scale_grad(OrionLoRAMatrix *mat, float scale);

void orion_qwen_lora_matrix_free(OrionLoRAMatrix *mat);

void orion_qwen_lora_linear_forward(const float *x,
                                    const float *w_base,
                                    const OrionLoRAMatrix *lora,
                                    float *y_out);

void orion_qwen_lora_linear_delta_forward(const float *x,
                                          const OrionLoRAMatrix *lora,
                                          float *y_delta_out);

void orion_qwen_lora_linear_backward(const float *x,
                                     const float *w_base,
                                     OrionLoRAMatrix *lora,
                                     const float *dy,
                                     float *dx_out);

void orion_qwen_cpu_swiglu_ffn_bwd(const float *x,
                                   const float *gate_proj,
                                   const float *up_proj,
                                   const float *down_proj,
                                   int d_model,
                                   int d_ff,
                                   const float *dy_out,
                                   float *dx_out);

OrionQwenStreamingCEContext *orion_qwen_streaming_ce_context_open(const char *embed_blob_path,
                                                                  int d_model,
                                                                  int vocab);

float orion_qwen_streaming_ce_tied_embedding_ctx(OrionQwenStreamingCEContext *ctx,
                                                 const float *hidden,
                                                 int target_token,
                                                 float *d_hidden_out);

void orion_qwen_streaming_ce_context_close(OrionQwenStreamingCEContext *ctx);

/// Streaming cross-entropy against tied embeddings.
/// Uses a 1-pass numerically stable accumulation over vocab rows.
float orion_qwen_cpu_streaming_ce_tied_embedding(const char *embed_blob_path,
                                                 const float *hidden,
                                                 int d_model,
                                                 int vocab,
                                                 int target_token,
                                                 float *d_hidden_out);

double orion_qwen_lora_abs_sum(const float *x, int n);

double orion_qwen_lora_grad_abs_sum(const OrionLoRAMatrix *mat);

#endif // ORION_QWEN_LORA_CPU_OPS_H
