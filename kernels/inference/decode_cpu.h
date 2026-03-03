#ifndef ORION_DECODE_CPU_H
#define ORION_DECODE_CPU_H

#import "../../core/ane_runtime.h"
#import "../../model/weight_loader.h"
#import "kv_cache.h"

/// T032-T036: CPU-based GPT-2 inference (prefill + decode).

#pragma mark - CPU Ops (T033)

/// GPT-2 LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
/// @param x     Input vector [dim]
/// @param gamma Scale [dim]
/// @param beta  Bias [dim]
/// @param dim   Vector dimension
/// @param out   Output vector [dim] (may alias x)
void orion_cpu_layernorm(const float* x, const float* gamma, const float* beta,
                          int dim, float* out);

#pragma mark - Prefill Forward Pass (T036)

/// Run full CPU GPT-2 forward pass (prefill mode).
/// Processes all tokens in parallel and returns logits for the last position.
/// @param w         Loaded GPT-2 weights
/// @param tokens    Input token ids [seq_len]
/// @param seq_len   Number of tokens
/// @param logits    Output logits buffer [vocab] (last position only)
void orion_gpt2_forward_cpu(const OrionGPT2Weights* w,
                             const int* tokens, int seq_len,
                             float* logits);

/// Run full CPU GPT-2 forward pass returning all logits.
/// @param w         Loaded GPT-2 weights
/// @param tokens    Input token ids [seq_len]
/// @param seq_len   Number of tokens
/// @param all_logits Output logits buffer [seq_len * vocab]
void orion_gpt2_forward_cpu_all(const OrionGPT2Weights* w,
                                 const int* tokens, int seq_len,
                                 float* all_logits);

#pragma mark - Decode Step (T039)

/// Run a single decode step on CPU using KV cache.
/// @param w         Loaded GPT-2 weights
/// @param kv        KV cache (read for past K,V; updated with new K,V)
/// @param token     Input token id for this step
/// @param logits    Output logits buffer [vocab]
void orion_gpt2_decode_step(const OrionGPT2Weights* w,
                              OrionKVCache* kv,
                              int token,
                              float* logits);

#pragma mark - Sampling (T040)

/// Sample a token from logits with temperature and top-p.
int orion_sample_token(const float* logits, int vocab_size,
                       float temperature, float top_p, uint64_t* rng_state);

#endif // ORION_DECODE_CPU_H
