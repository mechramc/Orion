#ifndef ORION_STORIES_CPU_OPS_H
#define ORION_STORIES_CPU_OPS_H

#import <Accelerate/Accelerate.h>

// CPU operations for Stories110M training.
// These run on CPU because:
//   - Adam: weights are baked constants at ANE compile time
//   - dW: weight gradient accumulation requires cblas_sgemm
//   - NLL loss: requires gather (per-position token indexing, not in MIL)
//   - Classifier backward: ANE rejects 32000-input-channel convolutions

// TODO(M3): Implement using vDSP and cblas

/// RMSNorm forward on CPU (used for fallback / verification).
void orion_cpu_rmsnorm(float* out, const float* x, const float* weight,
                       int dim, float eps);

/// Cross-entropy loss and gradient.
float orion_cpu_cross_entropy(float* dlogits, const float* logits,
                              const int* targets, int vocab, int seq_len);

/// Token embedding lookup.
void orion_cpu_embedding(float* out, const float* embed_table,
                         const int* tokens, int dim, int seq_len);

/// Adam optimizer step for a single parameter tensor.
void orion_cpu_adam_step(float* param, const float* grad,
                         float* m, float* v,
                         int size, float lr, float beta1, float beta2,
                         float eps, int t);

/// dW gradient accumulation: dW += x^T @ dy (via cblas_sgemm).
void orion_cpu_dw_accum(float* dw, const float* x, const float* dy,
                        int m, int n, int k);

/// RMSNorm backward on CPU.
/// Computes dx (gradient to input) and accumulates dweight.
/// x: [dim * seq], weight: [dim], dy: [dim * seq] (incoming gradient)
/// dx: [dim * seq] (output gradient), dweight: [dim] (accumulated)
void orion_cpu_rmsnorm_bwd(float* dx, float* dweight,
                            const float* dy, const float* x,
                            const float* weight,
                            int dim, int seq_len, float eps);

/// Embedding backward: accumulate gradients into embedding table.
/// dembed[token[i]] += dy[i] for each position i.
void orion_cpu_embedding_bwd(float* dembed, const float* dy,
                              const int* tokens, int dim, int seq_len);

#endif // ORION_STORIES_CPU_OPS_H
