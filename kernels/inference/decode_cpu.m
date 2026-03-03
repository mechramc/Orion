#import "decode_cpu.h"
#import <Accelerate/Accelerate.h>
#import <math.h>
#import <stdlib.h>
#import <string.h>

// T032-T036: CPU GPT-2 Forward Pass
// Uses Accelerate (cblas_sgemm) for matrix multiplications.
//
// Weight layout: converter transposes Conv1D weights [in, out] → blob [out, in].
// CPU matmul: Y = X @ W_original = X @ W_blob^T → use CblasTrans on W_blob.

static const float LAYERNORM_EPS = 1e-5f;

#pragma mark - T033: LayerNorm

void orion_cpu_layernorm(const float* x, const float* gamma, const float* beta,
                          int dim, float* out) {
    // mean = sum(x) / dim
    float mean = 0.0f;
    vDSP_meanv(x, 1, &mean, dim);

    // var = sum((x - mean)^2) / dim
    float neg_mean = -mean;
    float* tmp = (float*)malloc(dim * sizeof(float));
    vDSP_vsadd(x, 1, &neg_mean, tmp, 1, dim);  // tmp = x - mean
    float var = 0.0f;
    vDSP_dotpr(tmp, 1, tmp, 1, &var, dim);      // var = sum(tmp^2)
    var /= dim;

    // rstd = 1 / sqrt(var + eps)
    float rstd = 1.0f / sqrtf(var + LAYERNORM_EPS);

    // out = gamma * (x - mean) * rstd + beta
    vDSP_vsmul(tmp, 1, &rstd, tmp, 1, dim);     // tmp = (x - mean) * rstd
    vDSP_vmul(tmp, 1, gamma, 1, out, 1, dim);   // out = tmp * gamma
    vDSP_vadd(out, 1, beta, 1, out, 1, dim);    // out = out + beta

    free(tmp);
}

#pragma mark - Internal: Linear (matmul + bias)

/// Y[M, N] = X[M, K] @ W_blob[N, K]^T + bias[N]
/// W_blob is stored [N, K] (transposed from Conv1D).
static void linear(const float* x, const float* w, const float* bias,
                    int M, int K, int N, float* out) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, x, K, w, K,
                0.0f, out, N);
    if (bias) {
        for (int i = 0; i < M; i++) {
            vDSP_vadd(out + i * N, 1, bias, 1, out + i * N, 1, N);
        }
    }
}

#pragma mark - Internal: GELU

/// GELU activation (tanh approximation matching GPT-2).
/// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
static void gelu(float* x, int n) {
    const float c = sqrtf(2.0f / M_PI);
    for (int i = 0; i < n; i++) {
        float v = x[i];
        float v3 = v * v * v;
        float t = c * (v + 0.044715f * v3);
        x[i] = 0.5f * v * (1.0f + tanhf(t));
    }
}

#pragma mark - T032: Token + Positional Embedding

/// Embed tokens and add positional embedding.
/// out[seq_len, d_model] = wte[tokens[i]] + wpe[i]
static void embed(const OrionGPT2Weights* w, const int* tokens,
                   int seq_len, float* out) {
    int d = w->d_model;
    for (int i = 0; i < seq_len; i++) {
        const float* tok_emb = w->wte + tokens[i] * d;
        const float* pos_emb = w->wpe + i * d;
        vDSP_vadd(tok_emb, 1, pos_emb, 1, out + i * d, 1, d);
    }
}

#pragma mark - T034: Multi-Head Attention (Prefill)

/// Full multi-head self-attention for prefill (all positions at once).
/// x[seq, d_model] → attn_out[seq, d_model]
static void attention_prefill(const OrionGPT2LayerWeights* l,
                                const float* x, int seq_len,
                                int n_head, int d_model, float* out) {
    int head_dim = d_model / n_head;

    // Q, K, V projections: [seq, d_model] @ [d_model, d_model]^T + bias
    float* q = (float*)malloc(seq_len * d_model * sizeof(float));
    float* k = (float*)malloc(seq_len * d_model * sizeof(float));
    float* v = (float*)malloc(seq_len * d_model * sizeof(float));

    linear(x, l->wq, l->bq, seq_len, d_model, d_model, q);
    linear(x, l->wk, l->bk, seq_len, d_model, d_model, k);
    linear(x, l->wv, l->bv, seq_len, d_model, d_model, v);

    // Attention per head
    float scale = 1.0f / sqrtf((float)head_dim);
    float* attn_out = (float*)calloc(seq_len * d_model, sizeof(float));
    float* scores = (float*)malloc(seq_len * seq_len * sizeof(float));
    float* softmax_buf = (float*)malloc(seq_len * sizeof(float));

    for (int h = 0; h < n_head; h++) {
        // Extract Q_h[seq, head_dim], K_h[seq, head_dim], V_h[seq, head_dim]
        // They are interleaved in [seq, d_model] as head-contiguous blocks
        // Q[pos][h * head_dim ... (h+1) * head_dim]
        float* qh = (float*)malloc(seq_len * head_dim * sizeof(float));
        float* kh = (float*)malloc(seq_len * head_dim * sizeof(float));
        float* vh = (float*)malloc(seq_len * head_dim * sizeof(float));

        for (int s = 0; s < seq_len; s++) {
            memcpy(qh + s * head_dim, q + s * d_model + h * head_dim, head_dim * sizeof(float));
            memcpy(kh + s * head_dim, k + s * d_model + h * head_dim, head_dim * sizeof(float));
            memcpy(vh + s * head_dim, v + s * d_model + h * head_dim, head_dim * sizeof(float));
        }

        // scores[seq, seq] = Q_h @ K_h^T * scale
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim,
                    scale, qh, head_dim, kh, head_dim,
                    0.0f, scores, seq_len);

        // Apply causal mask + softmax
        for (int i = 0; i < seq_len; i++) {
            // Mask future positions
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = -INFINITY;
            }

            // Softmax over row i
            float max_val = scores[i * seq_len];
            for (int j = 1; j <= i; j++) {
                if (scores[i * seq_len + j] > max_val)
                    max_val = scores[i * seq_len + j];
            }

            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                sum += scores[i * seq_len + j];
            }
            for (int j = 0; j <= i; j++) {
                scores[i * seq_len + j] /= sum;
            }
            // Future positions already -inf → exp → 0, but set explicitly
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = 0.0f;
            }
        }

        // attn_h[seq, head_dim] = scores[seq, seq] @ V_h[seq, head_dim]
        float* attn_h = (float*)malloc(seq_len * head_dim * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len,
                    1.0f, scores, seq_len, vh, head_dim,
                    0.0f, attn_h, head_dim);

        // Write back to attn_out[seq, d_model]
        for (int s = 0; s < seq_len; s++) {
            memcpy(attn_out + s * d_model + h * head_dim, attn_h + s * head_dim, head_dim * sizeof(float));
        }

        free(qh); free(kh); free(vh); free(attn_h);
    }

    // Output projection: [seq, d_model] @ [d_model, d_model]^T + bias
    linear(attn_out, l->wo, l->bo, seq_len, d_model, d_model, out);

    free(q); free(k); free(v);
    free(attn_out); free(scores); free(softmax_buf);
}

#pragma mark - T035: FFN

/// FFN: Linear → GELU → Linear
/// x[seq, d_model] → out[seq, d_model]
static void ffn(const OrionGPT2LayerWeights* l,
                 const float* x, int seq_len,
                 int d_model, int d_ff, float* out) {
    // fc = x @ wfc^T + bfc  → [seq, d_ff]
    float* fc = (float*)malloc(seq_len * d_ff * sizeof(float));
    linear(x, l->wfc, l->bfc, seq_len, d_model, d_ff, fc);

    // gelu activation in-place
    gelu(fc, seq_len * d_ff);

    // proj = fc @ wproj^T + bproj → [seq, d_model]
    linear(fc, l->wproj, l->bproj, seq_len, d_ff, d_model, out);

    free(fc);
}

#pragma mark - T036: Full Forward Pass

void orion_gpt2_forward_cpu(const OrionGPT2Weights* w,
                             const int* tokens, int seq_len,
                             float* logits) {
    int d = w->d_model;
    int vocab = w->vocab;

    // Allocate full logits and call the _all version, then copy last row
    float* all_logits = (float*)malloc(seq_len * vocab * sizeof(float));
    orion_gpt2_forward_cpu_all(w, tokens, seq_len, all_logits);

    // Copy last position logits
    memcpy(logits, all_logits + (seq_len - 1) * vocab, vocab * sizeof(float));
    free(all_logits);
}

void orion_gpt2_forward_cpu_all(const OrionGPT2Weights* w,
                                 const int* tokens, int seq_len,
                                 float* all_logits) {
    int d = w->d_model;
    int n_layer = w->n_layer;
    int n_head = 12;  // GPT-2 124M
    int d_ff = w->d_ff;
    int vocab = w->vocab;

    // x[seq, d_model] — hidden state
    float* x = (float*)malloc(seq_len * d * sizeof(float));
    float* ln_out = (float*)malloc(seq_len * d * sizeof(float));
    float* attn_out = (float*)malloc(seq_len * d * sizeof(float));
    float* ffn_out = (float*)malloc(seq_len * d * sizeof(float));

    // T032: Token + positional embedding
    embed(w, tokens, seq_len, x);

    // Transformer layers
    for (int layer = 0; layer < n_layer; layer++) {
        const OrionGPT2LayerWeights* l = &w->layers[layer];

        // Pre-attention LayerNorm
        for (int s = 0; s < seq_len; s++) {
            orion_cpu_layernorm(x + s * d, l->ln1_g, l->ln1_b, d, ln_out + s * d);
        }

        // Multi-head self-attention
        attention_prefill(l, ln_out, seq_len, n_head, d, attn_out);

        // Residual connection: x = x + attn_out
        vDSP_vadd(x, 1, attn_out, 1, x, 1, seq_len * d);

        // Pre-FFN LayerNorm
        for (int s = 0; s < seq_len; s++) {
            orion_cpu_layernorm(x + s * d, l->ln2_g, l->ln2_b, d, ln_out + s * d);
        }

        // FFN
        ffn(l, ln_out, seq_len, d, d_ff, ffn_out);

        // Residual connection: x = x + ffn_out
        vDSP_vadd(x, 1, ffn_out, 1, x, 1, seq_len * d);
    }

    // Final LayerNorm
    for (int s = 0; s < seq_len; s++) {
        orion_cpu_layernorm(x + s * d, w->ln_f_g, w->ln_f_b, d, x + s * d);
    }

    // Logits: x[seq, d_model] @ wte[vocab, d_model]^T → [seq, vocab]
    // wte is NOT transposed (stored as [vocab, d_model]), so use CblasTrans
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, vocab, d,
                1.0f, x, d, w->wte, d,
                0.0f, all_logits, vocab);

    free(x); free(ln_out); free(attn_out); free(ffn_out);
}

#pragma mark - T039: Single-Token Decode (stub for now)

void orion_gpt2_decode_step(const OrionGPT2Weights* w,
                              OrionKVCache* kv,
                              int token,
                              float* logits) {
    // TODO(T039): Single-token transformer forward pass with KV cache
    // For now, use prefill with seq_len=kv->current_len+1
    // This is correct but slow (recomputes all positions)
}

#pragma mark - T040: Sampling

int orion_sample_token(const float* logits, int vocab_size,
                       float temperature, float top_p, uint64_t* rng_state) {
    // Simple argmax for temp=0
    if (temperature <= 0.0f) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }
    // TODO(T040): Full top-p nucleus sampling
    return 0;
}
