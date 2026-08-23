#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <unistd.h>
#import "../model/weight_loader.h"
#import "../kernels/inference/qwen_cpu_ops.h"

static double abs_sum(const float *x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)x[i]);
    return total;
}

static int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

static float *load_exact(const char *blob_dir, int layer_idx, const char *suffix, int count) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/layer%d/%s", blob_dir, layer_idx, suffix);
    return orion_read_blob_f32_exact(path, count);
}

static int sampled_topk_logits(const char *blob_dir,
                               const char *lm_head_name,
                               const float *hidden,
                               int d_model,
                               int sample_vocab,
                               int *top_id,
                               float *top_logit) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/model/%s", blob_dir, lm_head_name);
    float *row = (float *)malloc((size_t)d_model * sizeof(float));
    if (!row) return 0;

    int best_id = -1;
    float best = -INFINITY;
    for (int tok = 0; tok < sample_vocab; tok++) {
        if (!orion_read_blob_row_f32(path, tok, d_model, row)) {
            free(row);
            return 0;
        }
        float dot = 0.0f;
        for (int i = 0; i < d_model; i++) dot += hidden[i] * row[i];
        if (dot > best) {
            best = dot;
            best_id = tok;
        }
    }

    free(row);
    *top_id = best_id;
    *top_logit = best;
    return 1;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <blob_dir>\n", argv[0]);
            return 2;
        }

        const char *blob_dir = argv[1];
        OrionQwen35Manifest *manifest = orion_qwen35_manifest_load(blob_dir);
        if (!manifest) {
            fprintf(stderr, "FAIL: manifest loader returned NULL\n");
            return 1;
        }

        const int seq_len = 1;
        const int d_model = manifest->d_model;
        const int d_ff = manifest->d_ff;
        const int n_head = manifest->n_head;
        const int n_kv_head = manifest->n_kv_head;
        const int head_dim = manifest->head_dim;
        const int q_dim = n_head * head_dim;
        const int kv_dim = n_kv_head * head_dim;

        float *hidden = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *normed = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *mixer_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *mlp_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *scratch = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *last_hidden = (float *)calloc((size_t)d_model, sizeof(float));
        float *final_norm = NULL;
        if (!hidden || !normed || !mixer_out || !mlp_out || !scratch || !last_hidden) {
            fprintf(stderr, "FAIL: allocation failed\n");
            goto fail;
        }

        char embed_path[2048];
        snprintf(embed_path, sizeof(embed_path), "%s/model/embed_tokens.bin", blob_dir);
        if (!orion_read_blob_row_f32(embed_path, 0, d_model, hidden)) {
            fprintf(stderr, "FAIL: failed to read token embedding row 0\n");
            goto fail;
        }

        for (int layer_idx = 0; layer_idx < manifest->n_layer; layer_idx++) {
            float *input_ln = load_exact(blob_dir, layer_idx, "input_layernorm.bin", d_model);
            float *post_ln = load_exact(blob_dir, layer_idx, "post_attention_layernorm.bin", d_model);
            float *gate_proj = load_exact(blob_dir, layer_idx, "mlp_gate_proj.bin", d_ff * d_model);
            float *up_proj = load_exact(blob_dir, layer_idx, "mlp_up_proj.bin", d_ff * d_model);
            float *down_proj = load_exact(blob_dir, layer_idx, "mlp_down_proj.bin", d_model * d_ff);
            if (!input_ln || !post_ln || !gate_proj || !up_proj || !down_proj) {
                fprintf(stderr, "FAIL: missing layer %d norm/mlp tensors\n", layer_idx);
                free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
                goto fail;
            }

            orion_qwen_cpu_rmsnorm(hidden, input_ln, d_model, 1e-6f, normed);
            memset(mixer_out, 0, (size_t)d_model * sizeof(float));

            char full_q_path[2048];
            snprintf(full_q_path, sizeof(full_q_path), "%s/layer%d/self_attn_q_proj.bin", blob_dir, layer_idx);
            if (file_exists(full_q_path)) {
                float *q_proj = load_exact(blob_dir, layer_idx, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
                float *k_proj = load_exact(blob_dir, layer_idx, "self_attn_k_proj.bin", kv_dim * d_model);
                float *v_proj = load_exact(blob_dir, layer_idx, "self_attn_v_proj.bin", kv_dim * d_model);
                float *o_proj = load_exact(blob_dir, layer_idx, "self_attn_o_proj.bin", d_model * q_dim);
                float *q_norm = load_exact(blob_dir, layer_idx, "self_attn_q_norm.bin", head_dim);
                float *k_norm = load_exact(blob_dir, layer_idx, "self_attn_k_norm.bin", head_dim);
                if (!q_proj || !k_proj || !v_proj || !o_proj || !q_norm || !k_norm) {
                    fprintf(stderr, "FAIL: missing full-attention tensors for layer %d\n", layer_idx);
                    free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
                    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
                    goto fail;
                }
                orion_qwen_cpu_full_attention_prefill_with_rope(
                    normed, seq_len, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
                    d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
                    mixer_out
                );
                free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
            } else {
                int qkv_rows = orion_blob_element_count(([NSString stringWithFormat:@"%s/layer%d/linear_attn_in_proj_qkv.bin", blob_dir, layer_idx]).UTF8String) / d_model;
                int value_dim = orion_blob_element_count(([NSString stringWithFormat:@"%s/layer%d/linear_attn_out_proj.bin", blob_dir, layer_idx]).UTF8String) / d_model;
                int num_v_heads = orion_blob_element_count(([NSString stringWithFormat:@"%s/layer%d/linear_attn_dt_bias.bin", blob_dir, layer_idx]).UTF8String);
                int head_v_dim = orion_blob_element_count(([NSString stringWithFormat:@"%s/layer%d/linear_attn_norm.bin", blob_dir, layer_idx]).UTF8String);
                int key_dim = (qkv_rows - value_dim) / 2;
                int num_k_heads = num_v_heads;
                int head_k_dim = key_dim / num_k_heads;
                int conv_kernel = orion_blob_element_count(([NSString stringWithFormat:@"%s/layer%d/linear_attn_conv1d.bin", blob_dir, layer_idx]).UTF8String) / qkv_rows;

                float *in_proj_qkv = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_qkv.bin", qkv_rows * d_model);
                float *in_proj_z = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_z.bin", value_dim * d_model);
                float *in_proj_a = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_a.bin", num_v_heads * d_model);
                float *in_proj_b = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_b.bin", num_v_heads * d_model);
                float *conv1d = load_exact(blob_dir, layer_idx, "linear_attn_conv1d.bin", qkv_rows * conv_kernel);
                float *dt_bias = load_exact(blob_dir, layer_idx, "linear_attn_dt_bias.bin", num_v_heads);
                float *a_log = load_exact(blob_dir, layer_idx, "linear_attn_a_log.bin", num_v_heads);
                float *norm_weight = load_exact(blob_dir, layer_idx, "linear_attn_norm.bin", head_v_dim);
                float *out_proj = load_exact(blob_dir, layer_idx, "linear_attn_out_proj.bin", d_model * value_dim);
                if (!in_proj_qkv || !in_proj_z || !in_proj_a || !in_proj_b || !conv1d ||
                    !dt_bias || !a_log || !norm_weight || !out_proj) {
                    fprintf(stderr, "FAIL: missing linear-attention tensors for layer %d\n", layer_idx);
                    free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
                    free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
                    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
                    goto fail;
                }
                orion_qwen_cpu_linear_attention_recurrent_prefill(
                    normed, seq_len, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
                    conv1d, dt_bias, a_log, norm_weight, out_proj,
                    d_model, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel,
                    mixer_out
                );
                free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
                free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
            }

            for (int i = 0; i < d_model; i++) hidden[i] += mixer_out[i];
            orion_qwen_cpu_rmsnorm(hidden, post_ln, d_model, 1e-6f, scratch);
            orion_qwen_cpu_swiglu_ffn(scratch, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out);
            for (int i = 0; i < d_model; i++) hidden[i] += mlp_out[i];

            free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
        }

        char final_norm_path[2048];
        snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
        final_norm = orion_read_blob_f32_exact(final_norm_path, d_model);
        if (!final_norm) {
            fprintf(stderr, "FAIL: missing final norm\n");
            goto fail;
        }
        orion_qwen_cpu_rmsnorm(hidden, final_norm, d_model, 1e-6f, last_hidden);

        const char *lm_head_name = manifest->tie_word_embeddings ? "embed_tokens.bin" : "lm_head.bin";
        int top_id = -1;
        float top_logit = -INFINITY;
        if (!sampled_topk_logits(blob_dir, lm_head_name, last_hidden, d_model, 16, &top_id, &top_logit)) {
            fprintf(stderr, "FAIL: failed sampled logits scan\n");
            goto fail;
        }

        double hidden_abs = abs_sum(hidden, d_model);
        double last_hidden_abs = abs_sum(last_hidden, d_model);
        if (hidden_abs <= 0.0 || last_hidden_abs <= 0.0 || top_id < 0 || !isfinite(top_logit)) {
            fprintf(stderr, "FAIL: expected non-zero hidden/logits path\n");
            goto fail;
        }

        printf("PASS: qwen35 9b cpu infer smoke\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  seq_len=%d\n", seq_len);
        printf("  layers_run=%d\n", manifest->n_layer);
        printf("  hidden_abs_sum=%.6f\n", hidden_abs);
        printf("  last_hidden_abs_sum=%.6f\n", last_hidden_abs);
        printf("  sampled_top_token_id=%d\n", top_id);
        printf("  sampled_top_token_logit=%.6f\n", top_logit);
        printf("  next_blocker=%s\n", "9b decode loop and ANE prefill integration");

        free(hidden);
        free(normed);
        free(mixer_out);
        free(mlp_out);
        free(scratch);
        free(last_hidden);
        free(final_norm);
        orion_qwen35_manifest_free(manifest);
        return 0;

fail:
        free(hidden);
        free(normed);
        free(mixer_out);
        free(mlp_out);
        free(scratch);
        free(last_hidden);
        free(final_norm);
        orion_qwen35_manifest_free(manifest);
        return 1;
    }
}
