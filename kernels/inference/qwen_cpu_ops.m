#import "qwen_cpu_ops.h"
#import <Accelerate/Accelerate.h>
#import <math.h>
#import <stdlib.h>

void orion_qwen_cpu_rmsnorm(const float* x, const float* weight, int dim, float eps, float* out) {
    float mean_sq = 0.0f;
    vDSP_measqv(x, 1, &mean_sq, dim);
    float scale = 1.0f / sqrtf(mean_sq + eps);
    vDSP_vsmul(x, 1, &scale, out, 1, dim);
    vDSP_vmul(out, 1, weight, 1, out, 1, dim);
}

static void linear_no_bias(const float* x, const float* w, int in_dim, int out_dim, float* out) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                out_dim, in_dim,
                1.0f, w, in_dim, x, 1,
                0.0f, out, 1);
}

static inline float silu(float x) {
    return x / (1.0f + expf(-x));
}

static inline float sigmoidf_safe(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float softplusf_safe(float x) {
    if (x > 20.0f) return x;
    return log1pf(expf(x));
}

static void l2norm_vec(const float* x, int dim, float eps, float* out) {
    float sumsq = 0.0f;
    for (int i = 0; i < dim; i++) {
        sumsq += x[i] * x[i];
    }
    float inv = 1.0f / sqrtf(sumsq + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = x[i] * inv;
    }
}

static void qwen_rmsnorm_gated(const float* x, const float* gate, const float* weight, int dim, float eps, float* out) {
    float mean_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        mean_sq += x[i] * x[i];
    }
    mean_sq /= (float)dim;
    float inv = 1.0f / sqrtf(mean_sq + eps);
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] * inv) * weight[i] * silu(gate[i]);
    }
}

static void apply_rope_text_inplace(float* q,
                                    float* k,
                                    int seq_len,
                                    int n_q_head,
                                    int n_kv_head,
                                    int head_dim,
                                    float rope_theta,
                                    float partial_rotary_factor) {
    int rotary_dim = (int)(head_dim * partial_rotary_factor);
    if (rotary_dim > head_dim) rotary_dim = head_dim;
    if (rotary_dim % 2 != 0) rotary_dim -= 1;
    if (rotary_dim <= 0) return;

    int half_rot = rotary_dim / 2;
    float* inv_freq = (float*)malloc((size_t)half_rot * sizeof(float));
    if (!inv_freq) return;

    for (int i = 0; i < half_rot; i++) {
        float exponent = (2.0f * (float)i) / (float)rotary_dim;
        inv_freq[i] = 1.0f / powf(rope_theta, exponent);
    }

    for (int pos = 0; pos < seq_len; pos++) {
        for (int h = 0; h < n_q_head; h++) {
            float* qh = q + pos * (n_q_head * head_dim) + h * head_dim;
            for (int i = 0; i < half_rot; i++) {
                float angle = (float)pos * inv_freq[i];
                float c = cosf(angle);
                float s = sinf(angle);
                float x0 = qh[i];
                float x1 = qh[i + half_rot];
                qh[i] = x0 * c - x1 * s;
                qh[i + half_rot] = x1 * c + x0 * s;
            }
        }
        for (int h = 0; h < n_kv_head; h++) {
            float* kh = k + pos * (n_kv_head * head_dim) + h * head_dim;
            for (int i = 0; i < half_rot; i++) {
                float angle = (float)pos * inv_freq[i];
                float c = cosf(angle);
                float s = sinf(angle);
                float x0 = kh[i];
                float x1 = kh[i + half_rot];
                kh[i] = x0 * c - x1 * s;
                kh[i + half_rot] = x1 * c + x0 * s;
            }
        }
    }

    free(inv_freq);
}

static void qwen_full_attention_from_projected_qkv(const float* q_proj_out_seq,
                                                   const float* k_proj_out_seq,
                                                   const float* v_proj_out_seq,
                                                   int seq_len,
                                                   const float* o_proj,
                                                   const float* q_norm,
                                                   const float* k_norm,
                                                   int d_model,
                                                   int n_head,
                                                   int n_kv_head,
                                                   int head_dim,
                                                   int apply_rope,
                                                   float rope_theta,
                                                   float partial_rotary_factor,
                                                   float* out_seq) {
    int q_dim = n_head * head_dim;
    int kv_dim = n_kv_head * head_dim;
    int q_per_kv = n_head / n_kv_head;
    float scale = 1.0f / sqrtf((float)head_dim);

    float* q = (float*)malloc((size_t)seq_len * q_dim * sizeof(float));
    float* gate = (float*)malloc((size_t)seq_len * q_dim * sizeof(float));
    float* k = (float*)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float* v = (float*)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float* attn_cat = (float*)calloc((size_t)seq_len * q_dim, sizeof(float));
    float* scores = (float*)malloc((size_t)seq_len * seq_len * sizeof(float));

    memcpy(k, k_proj_out_seq, (size_t)seq_len * kv_dim * sizeof(float));
    memcpy(v, v_proj_out_seq, (size_t)seq_len * kv_dim * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        memcpy(q + s * q_dim,
               q_proj_out_seq + s * (q_dim * 2),
               (size_t)q_dim * sizeof(float));
        memcpy(gate + s * q_dim,
               q_proj_out_seq + s * (q_dim * 2) + q_dim,
               (size_t)q_dim * sizeof(float));
    }

    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_head; h++) {
            orion_qwen_cpu_rmsnorm(q + s * q_dim + h * head_dim, q_norm, head_dim, 1e-6f,
                              q + s * q_dim + h * head_dim);
        }
        for (int h = 0; h < n_kv_head; h++) {
            orion_qwen_cpu_rmsnorm(k + s * kv_dim + h * head_dim, k_norm, head_dim, 1e-6f,
                              k + s * kv_dim + h * head_dim);
        }
    }

    if (apply_rope) {
        apply_rope_text_inplace(q, k, seq_len, n_head, n_kv_head, head_dim, rope_theta, partial_rotary_factor);
    }

    for (int h = 0; h < n_head; h++) {
        int kv_head = h / q_per_kv;
        float* qh = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));
        float* kh = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));
        float* vh = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));
        float* attn_h = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));

        for (int s = 0; s < seq_len; s++) {
            memcpy(qh + s * head_dim, q + s * q_dim + h * head_dim, (size_t)head_dim * sizeof(float));
            memcpy(kh + s * head_dim, k + s * kv_dim + kv_head * head_dim, (size_t)head_dim * sizeof(float));
            memcpy(vh + s * head_dim, v + s * kv_dim + kv_head * head_dim, (size_t)head_dim * sizeof(float));
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim,
                    scale, qh, head_dim, kh, head_dim,
                    0.0f, scores, seq_len);

        for (int i = 0; i < seq_len; i++) {
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = -INFINITY;
            }
            float max_val = scores[i * seq_len];
            for (int j = 1; j <= i; j++) {
                if (scores[i * seq_len + j] > max_val) {
                    max_val = scores[i * seq_len + j];
                }
            }
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                sum += scores[i * seq_len + j];
            }
            for (int j = 0; j <= i; j++) {
                scores[i * seq_len + j] /= sum;
            }
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = 0.0f;
            }
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len,
                    1.0f, scores, seq_len, vh, head_dim,
                    0.0f, attn_h, head_dim);

        for (int s = 0; s < seq_len; s++) {
            memcpy(attn_cat + s * q_dim + h * head_dim,
                   attn_h + s * head_dim,
                   (size_t)head_dim * sizeof(float));
        }

        free(qh);
        free(kh);
        free(vh);
        free(attn_h);
    }

    for (int i = 0; i < seq_len * q_dim; i++) {
        attn_cat[i] *= sigmoidf_safe(gate[i]);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, d_model, q_dim,
                1.0f, attn_cat, q_dim, o_proj, q_dim,
                0.0f, out_seq, d_model);

    free(q);
    free(gate);
    free(k);
    free(v);
    free(attn_cat);
    free(scores);
}

void orion_qwen_cpu_swiglu_ffn(const float* x,
                               const float* gate_proj,
                               const float* up_proj,
                               const float* down_proj,
                               int d_model,
                               int d_ff,
                               float* out) {
    float* gate = (float*)malloc((size_t)d_ff * sizeof(float));
    float* up = (float*)malloc((size_t)d_ff * sizeof(float));
    float* hidden = (float*)malloc((size_t)d_ff * sizeof(float));

    linear_no_bias(x, gate_proj, d_model, d_ff, gate);
    linear_no_bias(x, up_proj, d_model, d_ff, up);

    for (int i = 0; i < d_ff; i++) {
        hidden[i] = silu(gate[i]) * up[i];
    }

    linear_no_bias(hidden, down_proj, d_ff, d_model, out);

    free(gate);
    free(up);
    free(hidden);
}

void orion_qwen_cpu_full_attention_prefill_no_rope(const float* x_seq,
                                                   int seq_len,
                                                   const float* q_proj,
                                                   const float* k_proj,
                                                   const float* v_proj,
                                                   const float* o_proj,
                                                   const float* q_norm,
                                                   const float* k_norm,
                                                   int d_model,
                                                   int n_head,
                                                   int n_kv_head,
                                                   int head_dim,
                                                   float* out_seq) {
    int q_dim = n_head * head_dim;
    int kv_dim = n_kv_head * head_dim;
    int q_per_kv = n_head / n_kv_head;
    float scale = 1.0f / sqrtf((float)head_dim);

    float* q_full = (float*)malloc((size_t)seq_len * (q_dim * 2) * sizeof(float));
    float* q = (float*)malloc((size_t)seq_len * q_dim * sizeof(float));
    float* gate = (float*)malloc((size_t)seq_len * q_dim * sizeof(float));
    float* k = (float*)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float* v = (float*)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float* attn_cat = (float*)calloc((size_t)seq_len * q_dim, sizeof(float));
    float* scores = (float*)malloc((size_t)seq_len * seq_len * sizeof(float));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, q_dim * 2, d_model,
                1.0f, x_seq, d_model, q_proj, d_model,
                0.0f, q_full, q_dim * 2);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, kv_dim, d_model,
                1.0f, x_seq, d_model, k_proj, d_model,
                0.0f, k, kv_dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, kv_dim, d_model,
                1.0f, x_seq, d_model, v_proj, d_model,
                0.0f, v, kv_dim);

    for (int s = 0; s < seq_len; s++) {
        memcpy(q + s * q_dim, q_full + s * (q_dim * 2), (size_t)q_dim * sizeof(float));
        memcpy(gate + s * q_dim, q_full + s * (q_dim * 2) + q_dim, (size_t)q_dim * sizeof(float));
    }

    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_head; h++) {
            orion_qwen_cpu_rmsnorm(q + s * q_dim + h * head_dim, q_norm, head_dim, 1e-6f,
                              q + s * q_dim + h * head_dim);
        }
        for (int h = 0; h < n_kv_head; h++) {
            orion_qwen_cpu_rmsnorm(k + s * kv_dim + h * head_dim, k_norm, head_dim, 1e-6f,
                              k + s * kv_dim + h * head_dim);
        }
    }

    for (int h = 0; h < n_head; h++) {
        int kv_head = h / q_per_kv;
        float* qh = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));
        float* kh = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));
        float* vh = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));
        float* attn_h = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));

        for (int s = 0; s < seq_len; s++) {
            memcpy(qh + s * head_dim, q + s * q_dim + h * head_dim, (size_t)head_dim * sizeof(float));
            memcpy(kh + s * head_dim, k + s * kv_dim + kv_head * head_dim, (size_t)head_dim * sizeof(float));
            memcpy(vh + s * head_dim, v + s * kv_dim + kv_head * head_dim, (size_t)head_dim * sizeof(float));
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim,
                    scale, qh, head_dim, kh, head_dim,
                    0.0f, scores, seq_len);

        for (int i = 0; i < seq_len; i++) {
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = -INFINITY;
            }
            float max_val = scores[i * seq_len];
            for (int j = 1; j <= i; j++) {
                if (scores[i * seq_len + j] > max_val) {
                    max_val = scores[i * seq_len + j];
                }
            }
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                sum += scores[i * seq_len + j];
            }
            for (int j = 0; j <= i; j++) {
                scores[i * seq_len + j] /= sum;
            }
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = 0.0f;
            }
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len,
                    1.0f, scores, seq_len, vh, head_dim,
                    0.0f, attn_h, head_dim);

        for (int s = 0; s < seq_len; s++) {
            memcpy(attn_cat + s * q_dim + h * head_dim, attn_h + s * head_dim,
                   (size_t)head_dim * sizeof(float));
        }

        free(qh);
        free(kh);
        free(vh);
        free(attn_h);
    }

    for (int i = 0; i < seq_len * q_dim; i++) {
        attn_cat[i] *= sigmoidf_safe(gate[i]);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, d_model, q_dim,
                1.0f, attn_cat, q_dim, o_proj, q_dim,
                0.0f, out_seq, d_model);

    free(q_full);
    free(q);
    free(gate);
    free(k);
    free(v);
    free(attn_cat);
    free(scores);
}

void orion_qwen_cpu_full_attention_prefill_with_rope(const float* x_seq,
                                                     int seq_len,
                                                     const float* q_proj,
                                                     const float* k_proj,
                                                     const float* v_proj,
                                                     const float* o_proj,
                                                     const float* q_norm,
                                                     const float* k_norm,
                                                     int d_model,
                                                     int n_head,
                                                     int n_kv_head,
                                                     int head_dim,
                                                     float rope_theta,
                                                     float partial_rotary_factor,
                                                     float* out_seq) {
    int q_dim = n_head * head_dim;
    int kv_dim = n_kv_head * head_dim;
    int q_per_kv = n_head / n_kv_head;
    float scale = 1.0f / sqrtf((float)head_dim);

    float* q_full = (float*)malloc((size_t)seq_len * (q_dim * 2) * sizeof(float));
    float* q = (float*)malloc((size_t)seq_len * q_dim * sizeof(float));
    float* gate = (float*)malloc((size_t)seq_len * q_dim * sizeof(float));
    float* k = (float*)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float* v = (float*)malloc((size_t)seq_len * kv_dim * sizeof(float));
    float* attn_cat = (float*)calloc((size_t)seq_len * q_dim, sizeof(float));
    float* scores = (float*)malloc((size_t)seq_len * seq_len * sizeof(float));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, q_dim * 2, d_model,
                1.0f, x_seq, d_model, q_proj, d_model,
                0.0f, q_full, q_dim * 2);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, kv_dim, d_model,
                1.0f, x_seq, d_model, k_proj, d_model,
                0.0f, k, kv_dim);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, kv_dim, d_model,
                1.0f, x_seq, d_model, v_proj, d_model,
                0.0f, v, kv_dim);

    for (int s = 0; s < seq_len; s++) {
        memcpy(q + s * q_dim, q_full + s * (q_dim * 2), (size_t)q_dim * sizeof(float));
        memcpy(gate + s * q_dim, q_full + s * (q_dim * 2) + q_dim, (size_t)q_dim * sizeof(float));
    }

    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_head; h++) {
            orion_qwen_cpu_rmsnorm(q + s * q_dim + h * head_dim, q_norm, head_dim, 1e-6f,
                              q + s * q_dim + h * head_dim);
        }
        for (int h = 0; h < n_kv_head; h++) {
            orion_qwen_cpu_rmsnorm(k + s * kv_dim + h * head_dim, k_norm, head_dim, 1e-6f,
                              k + s * kv_dim + h * head_dim);
        }
    }

    apply_rope_text_inplace(q, k, seq_len, n_head, n_kv_head, head_dim, rope_theta, partial_rotary_factor);

    for (int h = 0; h < n_head; h++) {
        int kv_head = h / q_per_kv;
        float* qh = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));
        float* kh = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));
        float* vh = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));
        float* attn_h = (float*)malloc((size_t)seq_len * head_dim * sizeof(float));

        for (int s = 0; s < seq_len; s++) {
            memcpy(qh + s * head_dim, q + s * q_dim + h * head_dim, (size_t)head_dim * sizeof(float));
            memcpy(kh + s * head_dim, k + s * kv_dim + kv_head * head_dim, (size_t)head_dim * sizeof(float));
            memcpy(vh + s * head_dim, v + s * kv_dim + kv_head * head_dim, (size_t)head_dim * sizeof(float));
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim,
                    scale, qh, head_dim, kh, head_dim,
                    0.0f, scores, seq_len);

        for (int i = 0; i < seq_len; i++) {
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = -INFINITY;
            }
            float max_val = scores[i * seq_len];
            for (int j = 1; j <= i; j++) {
                if (scores[i * seq_len + j] > max_val) {
                    max_val = scores[i * seq_len + j];
                }
            }
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                scores[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                sum += scores[i * seq_len + j];
            }
            for (int j = 0; j <= i; j++) {
                scores[i * seq_len + j] /= sum;
            }
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = 0.0f;
            }
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len,
                    1.0f, scores, seq_len, vh, head_dim,
                    0.0f, attn_h, head_dim);

        for (int s = 0; s < seq_len; s++) {
            memcpy(attn_cat + s * q_dim + h * head_dim, attn_h + s * head_dim,
                   (size_t)head_dim * sizeof(float));
        }

        free(qh);
        free(kh);
        free(vh);
        free(attn_h);
    }

    for (int i = 0; i < seq_len * q_dim; i++) {
        attn_cat[i] *= sigmoidf_safe(gate[i]);
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, d_model, q_dim,
                1.0f, attn_cat, q_dim, o_proj, q_dim,
                0.0f, out_seq, d_model);

    free(q_full);
    free(q);
    free(gate);
    free(k);
    free(v);
    free(attn_cat);
    free(scores);
}

void orion_qwen_cpu_linear_attention_prep(const float* x_seq,
                                          int seq_len,
                                          const float* in_proj_qkv,
                                          const float* in_proj_z,
                                          const float* in_proj_a,
                                          const float* in_proj_b,
                                          const float* conv1d,
                                          const float* dt_bias,
                                          const float* a_log,
                                          int d_model,
                                          int num_k_heads,
                                          int num_v_heads,
                                          int head_k_dim,
                                          int head_v_dim,
                                          int conv_kernel,
                                          float* query_out,
                                          float* key_out,
                                          float* value_out,
                                          float* z_out,
                                          float* beta_out,
                                          float* g_out) {
    int key_dim = num_k_heads * head_k_dim;
    int value_dim = num_v_heads * head_v_dim;
    int conv_dim = key_dim * 2 + value_dim;

    float* mixed = (float*)malloc((size_t)seq_len * conv_dim * sizeof(float));
    float* mixed_conv = (float*)malloc((size_t)seq_len * conv_dim * sizeof(float));

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, conv_dim, d_model,
                1.0f, x_seq, d_model, in_proj_qkv, d_model,
                0.0f, mixed, conv_dim);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, value_dim, d_model,
                1.0f, x_seq, d_model, in_proj_z, d_model,
                0.0f, z_out, value_dim);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, num_v_heads, d_model,
                1.0f, x_seq, d_model, in_proj_b, d_model,
                0.0f, beta_out, num_v_heads);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, num_v_heads, d_model,
                1.0f, x_seq, d_model, in_proj_a, d_model,
                0.0f, g_out, num_v_heads);

    for (int t = 0; t < seq_len; t++) {
        for (int c = 0; c < conv_dim; c++) {
            float sum = 0.0f;
            const float* kernel = conv1d + c * conv_kernel;
            for (int k = 0; k < conv_kernel; k++) {
                int src_t = t - (conv_kernel - 1) + k;
                float x = 0.0f;
                if (src_t >= 0 && src_t < seq_len) {
                    x = mixed[src_t * conv_dim + c];
                }
                sum += kernel[k] * x;
            }
            mixed_conv[t * conv_dim + c] = silu(sum);
        }
    }

    for (int t = 0; t < seq_len; t++) {
        const float* row = mixed_conv + t * conv_dim;
        memcpy(query_out + t * key_dim, row, (size_t)key_dim * sizeof(float));
        memcpy(key_out + t * key_dim, row + key_dim, (size_t)key_dim * sizeof(float));
        memcpy(value_out + t * value_dim, row + key_dim * 2, (size_t)value_dim * sizeof(float));
    }

    for (int i = 0; i < seq_len * num_v_heads; i++) {
        beta_out[i] = sigmoidf_safe(beta_out[i]);
    }

    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < num_v_heads; h++) {
            float a = g_out[t * num_v_heads + h];
            float dt = dt_bias[h];
            float al = a_log[h];
            g_out[t * num_v_heads + h] = -expf(al) * softplusf_safe(a + dt);
        }
    }

    free(mixed);
    free(mixed_conv);
}

void orion_qwen_cpu_linear_attention_recurrent_prefill(const float* x_seq,
                                                       int seq_len,
                                                       const float* in_proj_qkv,
                                                       const float* in_proj_z,
                                                       const float* in_proj_a,
                                                       const float* in_proj_b,
                                                       const float* conv1d,
                                                       const float* dt_bias,
                                                       const float* a_log,
                                                       const float* norm_weight,
                                                       const float* out_proj,
                                                       int d_model,
                                                       int num_k_heads,
                                                       int num_v_heads,
                                                       int head_k_dim,
                                                       int head_v_dim,
                                                       int conv_kernel,
                                                       float* out_seq) {
    int key_dim = num_k_heads * head_k_dim;
    int value_dim = num_v_heads * head_v_dim;
    float scale = 1.0f / sqrtf((float)head_k_dim);

    float* query = (float*)calloc((size_t)seq_len * key_dim, sizeof(float));
    float* key = (float*)calloc((size_t)seq_len * key_dim, sizeof(float));
    float* value = (float*)calloc((size_t)seq_len * value_dim, sizeof(float));
    float* z = (float*)calloc((size_t)seq_len * value_dim, sizeof(float));
    float* beta = (float*)calloc((size_t)seq_len * num_v_heads, sizeof(float));
    float* g = (float*)calloc((size_t)seq_len * num_v_heads, sizeof(float));
    float* core = (float*)calloc((size_t)seq_len * value_dim, sizeof(float));
    float* gated = (float*)calloc((size_t)seq_len * value_dim, sizeof(float));
    float* state = (float*)calloc((size_t)num_v_heads * head_k_dim * head_v_dim, sizeof(float));
    float* q_norm = (float*)malloc((size_t)head_k_dim * sizeof(float));
    float* k_norm = (float*)malloc((size_t)head_k_dim * sizeof(float));
    float* kv_mem = (float*)malloc((size_t)head_v_dim * sizeof(float));
    float* delta = (float*)malloc((size_t)head_v_dim * sizeof(float));
    float* core_head = (float*)malloc((size_t)head_v_dim * sizeof(float));

    orion_qwen_cpu_linear_attention_prep(
        x_seq, seq_len,
        in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, dt_bias, a_log,
        d_model, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel,
        query, key, value, z, beta, g
    );

    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < num_v_heads; h++) {
            const float* q_t = query + t * key_dim + h * head_k_dim;
            const float* k_t = key + t * key_dim + h * head_k_dim;
            const float* v_t = value + t * value_dim + h * head_v_dim;
            float* state_h = state + h * head_k_dim * head_v_dim;
            float* core_t = core + t * value_dim + h * head_v_dim;
            const float* z_t = z + t * value_dim + h * head_v_dim;

            l2norm_vec(q_t, head_k_dim, 1e-6f, q_norm);
            l2norm_vec(k_t, head_k_dim, 1e-6f, k_norm);
            for (int i = 0; i < head_k_dim; i++) {
                q_norm[i] *= scale;
            }

            float decay = expf(g[t * num_v_heads + h]);
            float beta_t = beta[t * num_v_heads + h];

            for (int i = 0; i < head_k_dim * head_v_dim; i++) {
                state_h[i] *= decay;
            }

            for (int v_idx = 0; v_idx < head_v_dim; v_idx++) {
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < head_k_dim; k_idx++) {
                    sum += state_h[k_idx * head_v_dim + v_idx] * k_norm[k_idx];
                }
                kv_mem[v_idx] = sum;
                delta[v_idx] = (v_t[v_idx] - sum) * beta_t;
            }

            for (int k_idx = 0; k_idx < head_k_dim; k_idx++) {
                for (int v_idx = 0; v_idx < head_v_dim; v_idx++) {
                    state_h[k_idx * head_v_dim + v_idx] += k_norm[k_idx] * delta[v_idx];
                }
            }

            for (int v_idx = 0; v_idx < head_v_dim; v_idx++) {
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < head_k_dim; k_idx++) {
                    sum += state_h[k_idx * head_v_dim + v_idx] * q_norm[k_idx];
                }
                core_head[v_idx] = sum;
            }

            qwen_rmsnorm_gated(core_head, z_t, norm_weight, head_v_dim, 1e-6f, core_t);
        }
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, d_model, value_dim,
                1.0f, core, value_dim, out_proj, value_dim,
                0.0f, out_seq, d_model);

    free(query);
    free(key);
    free(value);
    free(z);
    free(beta);
    free(g);
    free(core);
    free(gated);
    free(state);
    free(q_norm);
    free(k_norm);
    free(kv_mem);
    free(delta);
    free(core_head);
}

void orion_qwen_cpu_full_attention_from_projections_with_rope(
    const float* q_proj_out_seq,
    const float* k_proj_out_seq,
    const float* v_proj_out_seq,
    int seq_len,
    const float* o_proj,
    const float* q_norm,
    const float* k_norm,
    int d_model,
    int n_head,
    int n_kv_head,
    int head_dim,
    float rope_theta,
    float partial_rotary_factor,
    float* out_seq) {
    qwen_full_attention_from_projected_qkv(q_proj_out_seq,
                                           k_proj_out_seq,
                                           v_proj_out_seq,
                                           seq_len,
                                           o_proj,
                                           q_norm,
                                           k_norm,
                                           d_model,
                                           n_head,
                                           n_kv_head,
                                           head_dim,
                                           1,
                                           rope_theta,
                                           partial_rotary_factor,
                                           out_seq);
}
