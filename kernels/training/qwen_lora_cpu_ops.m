#import "qwen_lora_cpu_ops.h"
#import "../inference/qwen_cpu_ops.h"
#import "../training/stories_cpu_ops.h"
#import "../../model/weight_loader.h"
#import <Accelerate/Accelerate.h>
#import <math.h>
#import <stdlib.h>
#import <string.h>

struct OrionQwenStreamingCEContext {
    OrionBlobRowReader *reader;
    int d_model;
    int vocab;
    float *row;
    float *target;
    float *expected;
};

static inline float lora_rand_unit(unsigned int *state) {
    *state = (*state * 1664525u) + 1013904223u;
    return ((float)((*state >> 8) & 0x00FFFFFF) / 16777215.0f) - 0.5f;
}

static inline float silu_fwd(float x) {
    return x / (1.0f + expf(-x));
}

static inline float silu_bwd(float x) {
    float sig = 1.0f / (1.0f + expf(-x));
    return sig + x * sig * (1.0f - sig);
}

double orion_qwen_lora_abs_sum(const float *x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)x[i]);
    return total;
}

double orion_qwen_lora_grad_abs_sum(const OrionLoRAMatrix *mat) {
    if (!mat) return 0.0;
    return orion_qwen_lora_abs_sum(mat->da, mat->rank * mat->in_dim) +
           orion_qwen_lora_abs_sum(mat->db, mat->out_dim * mat->rank);
}

void orion_qwen_lora_matrix_init(OrionLoRAMatrix *mat,
                                 int in_dim,
                                 int out_dim,
                                 int rank,
                                 float alpha,
                                 unsigned int seed) {
    memset(mat, 0, sizeof(*mat));
    mat->in_dim = in_dim;
    mat->out_dim = out_dim;
    mat->rank = rank;
    mat->alpha = alpha;
    mat->scale = alpha / (float)rank;

    size_t a_count = (size_t)rank * in_dim;
    size_t b_count = (size_t)out_dim * rank;
    mat->a = (float *)calloc(a_count, sizeof(float));
    mat->b = (float *)calloc(b_count, sizeof(float));
    mat->da = (float *)calloc(a_count, sizeof(float));
    mat->db = (float *)calloc(b_count, sizeof(float));
    mat->ma = (float *)calloc(a_count, sizeof(float));
    mat->va = (float *)calloc(a_count, sizeof(float));
    mat->mb = (float *)calloc(b_count, sizeof(float));
    mat->vb = (float *)calloc(b_count, sizeof(float));

    unsigned int state = seed ? seed : 1u;
    for (size_t i = 0; i < a_count; i++) {
        mat->a[i] = 0.01f * lora_rand_unit(&state);
    }
}

void orion_qwen_lora_matrix_zero_grad(OrionLoRAMatrix *mat) {
    if (!mat || !mat->da || !mat->db) return;
    memset(mat->da, 0, (size_t)mat->rank * mat->in_dim * sizeof(float));
    memset(mat->db, 0, (size_t)mat->out_dim * mat->rank * sizeof(float));
}

void orion_qwen_lora_matrix_scale_grad(OrionLoRAMatrix *mat, float scale) {
    if (!mat || !mat->da || !mat->db) return;
    size_t a_count = (size_t)mat->rank * mat->in_dim;
    size_t b_count = (size_t)mat->out_dim * mat->rank;
    for (size_t i = 0; i < a_count; i++) mat->da[i] *= scale;
    for (size_t i = 0; i < b_count; i++) mat->db[i] *= scale;
}

void orion_qwen_lora_matrix_free(OrionLoRAMatrix *mat) {
    if (!mat) return;
    free(mat->a); free(mat->b);
    free(mat->da); free(mat->db);
    free(mat->ma); free(mat->va);
    free(mat->mb); free(mat->vb);
    memset(mat, 0, sizeof(*mat));
}

void orion_qwen_lora_linear_forward(const float *x,
                                    const float *w_base,
                                    const OrionLoRAMatrix *lora,
                                    float *y_out) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                lora->out_dim, lora->in_dim,
                1.0f, w_base, lora->in_dim, x, 1,
                0.0f, y_out, 1);

    float *z = (float *)calloc((size_t)lora->rank, sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                lora->rank, lora->in_dim,
                1.0f, lora->a, lora->in_dim, x, 1,
                0.0f, z, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                lora->out_dim, lora->rank,
                lora->scale, lora->b, lora->rank, z, 1,
                1.0f, y_out, 1);
    free(z);
}

void orion_qwen_lora_linear_delta_forward(const float *x,
                                          const OrionLoRAMatrix *lora,
                                          float *y_delta_out) {
    memset(y_delta_out, 0, (size_t)lora->out_dim * sizeof(float));
    float *z = (float *)calloc((size_t)lora->rank, sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                lora->rank, lora->in_dim,
                1.0f, lora->a, lora->in_dim, x, 1,
                0.0f, z, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                lora->out_dim, lora->rank,
                lora->scale, lora->b, lora->rank, z, 1,
                0.0f, y_delta_out, 1);
    free(z);
}

void orion_qwen_lora_linear_backward(const float *x,
                                     const float *w_base,
                                     OrionLoRAMatrix *lora,
                                     const float *dy,
                                     float *dx_out) {
    memset(dx_out, 0, (size_t)lora->in_dim * sizeof(float));

    cblas_sgemv(CblasRowMajor, CblasTrans,
                lora->out_dim, lora->in_dim,
                1.0f, w_base, lora->in_dim, dy, 1,
                0.0f, dx_out, 1);

    float *z = (float *)calloc((size_t)lora->rank, sizeof(float));
    float *dz = (float *)calloc((size_t)lora->rank, sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                lora->rank, lora->in_dim,
                1.0f, lora->a, lora->in_dim, x, 1,
                0.0f, z, 1);

    for (int o = 0; o < lora->out_dim; o++) {
        const float dy_o = dy[o];
        const float *b_row = lora->b + (size_t)o * lora->rank;
        float *db_row = lora->db + (size_t)o * lora->rank;
        for (int r = 0; r < lora->rank; r++) {
            db_row[r] += lora->scale * dy_o * z[r];
            dz[r] += lora->scale * b_row[r] * dy_o;
        }
    }

    for (int r = 0; r < lora->rank; r++) {
        float *da_row = lora->da + (size_t)r * lora->in_dim;
        const float a_scale = dz[r];
        for (int i = 0; i < lora->in_dim; i++) {
            da_row[i] += a_scale * x[i];
            dx_out[i] += lora->a[(size_t)r * lora->in_dim + i] * a_scale;
        }
    }

    free(z);
    free(dz);
}

void orion_qwen_cpu_swiglu_ffn_bwd(const float *x,
                                   const float *gate_proj,
                                   const float *up_proj,
                                   const float *down_proj,
                                   int d_model,
                                   int d_ff,
                                   const float *dy_out,
                                   float *dx_out) {
    float *gate = (float *)calloc((size_t)d_ff, sizeof(float));
    float *up = (float *)calloc((size_t)d_ff, sizeof(float));
    float *hidden = (float *)calloc((size_t)d_ff, sizeof(float));
    float *dhidden = (float *)calloc((size_t)d_ff, sizeof(float));
    float *dgate = (float *)calloc((size_t)d_ff, sizeof(float));
    float *dup = (float *)calloc((size_t)d_ff, sizeof(float));

    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                d_ff, d_model, 1.0f, gate_proj, d_model, x, 1, 0.0f, gate, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                d_ff, d_model, 1.0f, up_proj, d_model, x, 1, 0.0f, up, 1);
    for (int i = 0; i < d_ff; i++) hidden[i] = silu_fwd(gate[i]) * up[i];

    cblas_sgemv(CblasRowMajor, CblasTrans,
                d_model, d_ff, 1.0f, down_proj, d_ff, dy_out, 1, 0.0f, dhidden, 1);

    for (int i = 0; i < d_ff; i++) {
        dgate[i] = dhidden[i] * up[i] * silu_bwd(gate[i]);
        dup[i] = dhidden[i] * silu_fwd(gate[i]);
    }

    memset(dx_out, 0, (size_t)d_model * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasTrans,
                d_ff, d_model, 1.0f, gate_proj, d_model, dgate, 1, 1.0f, dx_out, 1);
    cblas_sgemv(CblasRowMajor, CblasTrans,
                d_ff, d_model, 1.0f, up_proj, d_model, dup, 1, 1.0f, dx_out, 1);

    free(gate);
    free(up);
    free(hidden);
    free(dhidden);
    free(dgate);
    free(dup);
}

float orion_qwen_cpu_streaming_ce_tied_embedding(const char *embed_blob_path,
                                                 const float *hidden,
                                                 int d_model,
                                                 int vocab,
                                                 int target_token,
                                                 float *d_hidden_out) {
    OrionQwenStreamingCEContext *ctx = orion_qwen_streaming_ce_context_open(embed_blob_path, d_model, vocab);
    if (!ctx) {
        return NAN;
    }
    float loss = orion_qwen_streaming_ce_tied_embedding_ctx(ctx, hidden, target_token, d_hidden_out);
    orion_qwen_streaming_ce_context_close(ctx);
    return loss;
}

OrionQwenStreamingCEContext *orion_qwen_streaming_ce_context_open(const char *embed_blob_path,
                                                                  int d_model,
                                                                  int vocab) {
    if (!embed_blob_path || d_model <= 0 || vocab <= 0) return NULL;
    OrionQwenStreamingCEContext *ctx = (OrionQwenStreamingCEContext *)calloc(1, sizeof(OrionQwenStreamingCEContext));
    if (!ctx) return NULL;
    ctx->reader = orion_blob_row_reader_open(embed_blob_path, d_model);
    ctx->d_model = d_model;
    ctx->vocab = vocab;
    ctx->row = (float *)malloc((size_t)d_model * sizeof(float));
    ctx->target = (float *)malloc((size_t)d_model * sizeof(float));
    ctx->expected = (float *)calloc((size_t)d_model, sizeof(float));
    if (!ctx->reader || !ctx->row || !ctx->target || !ctx->expected) {
        orion_qwen_streaming_ce_context_close(ctx);
        return NULL;
    }
    return ctx;
}

float orion_qwen_streaming_ce_tied_embedding_ctx(OrionQwenStreamingCEContext *ctx,
                                                 const float *hidden,
                                                 int target_token,
                                                 float *d_hidden_out) {
    if (!ctx || !ctx->reader || !hidden || !d_hidden_out || target_token < 0 || target_token >= ctx->vocab) {
        return NAN;
    }

    float max_logit = -INFINITY;
    float sum_exp = 0.0f;
    float target_logit = NAN;
    int target_seen = 0;
    memset(ctx->expected, 0, (size_t)ctx->d_model * sizeof(float));
    for (int tok = 0; tok < ctx->vocab; tok++) {
        if (!orion_blob_row_reader_read_f32(ctx->reader, tok, ctx->row)) {
            return NAN;
        }
        float dot = cblas_sdot(ctx->d_model, hidden, 1, ctx->row, 1);
        if (tok == target_token) {
            memcpy(ctx->target, ctx->row, (size_t)ctx->d_model * sizeof(float));
            target_logit = dot;
            target_seen = 1;
        }

        if (dot > max_logit) {
            float scale = isfinite(max_logit) ? expf(max_logit - dot) : 0.0f;
            for (int i = 0; i < ctx->d_model; i++) {
                ctx->expected[i] = (ctx->expected[i] * scale) + ctx->row[i];
            }
            sum_exp = (sum_exp * scale) + 1.0f;
            max_logit = dot;
        } else {
            float expv = expf(dot - max_logit);
            sum_exp += expv;
            for (int i = 0; i < ctx->d_model; i++) ctx->expected[i] += expv * ctx->row[i];
        }
    }

    if (!target_seen || !isfinite(target_logit) || !(sum_exp > 0.0f)) {
        return NAN;
    }

    float inv_sum = 1.0f / sum_exp;
    for (int i = 0; i < ctx->d_model; i++) {
        d_hidden_out[i] = ctx->expected[i] * inv_sum - ctx->target[i];
    }

    float loss = -(target_logit - max_logit - logf(sum_exp));
    return loss;
}

void orion_qwen_streaming_ce_context_close(OrionQwenStreamingCEContext *ctx) {
    if (!ctx) return;
    if (ctx->reader) orion_blob_row_reader_close(ctx->reader);
    free(ctx->row);
    free(ctx->target);
    free(ctx->expected);
    free(ctx);
}
