#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import <string.h>
#import "../model/weight_loader.h"
#import "../kernels/inference/qwen_cpu_ops.h"

static double abs_sum(const float *x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        total += fabs((double)x[i]);
    }
    return total;
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

        const int layer_idx = 0;
        const int seq_len = 2;
        const int d_model = manifest->d_model;
        const int num_k_heads = 16;
        const int num_v_heads = 16;
        const int head_k_dim = 128;
        const int head_v_dim = 128;
        const int conv_kernel = 4;
        const int key_dim = num_k_heads * head_k_dim;
        const int value_dim = num_v_heads * head_v_dim;

        float *x_seq = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *query = (float *)calloc((size_t)seq_len * key_dim, sizeof(float));
        float *key = (float *)calloc((size_t)seq_len * key_dim, sizeof(float));
        float *value = (float *)calloc((size_t)seq_len * value_dim, sizeof(float));
        float *z = (float *)calloc((size_t)seq_len * value_dim, sizeof(float));
        float *beta = (float *)calloc((size_t)seq_len * num_v_heads, sizeof(float));
        float *g = (float *)calloc((size_t)seq_len * num_v_heads, sizeof(float));
        if (!x_seq || !query || !key || !value || !z || !beta || !g) {
            fprintf(stderr, "FAIL: allocation failed\n");
            goto fail;
        }

        char path[2048];
        snprintf(path, sizeof(path), "%s/model/embed_tokens.bin", blob_dir);
        if (!orion_read_blob_row_f32(path, 0, d_model, x_seq + 0 * d_model) ||
            !orion_read_blob_row_f32(path, 1, d_model, x_seq + 1 * d_model)) {
            fprintf(stderr, "FAIL: failed to read token embeddings\n");
            goto fail;
        }

        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_in_proj_qkv.bin", blob_dir, layer_idx);
        float *in_proj_qkv = orion_read_blob_f32_exact(path, (key_dim * 2 + value_dim) * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_in_proj_z.bin", blob_dir, layer_idx);
        float *in_proj_z = orion_read_blob_f32_exact(path, value_dim * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_in_proj_a.bin", blob_dir, layer_idx);
        float *in_proj_a = orion_read_blob_f32_exact(path, num_v_heads * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_in_proj_b.bin", blob_dir, layer_idx);
        float *in_proj_b = orion_read_blob_f32_exact(path, num_v_heads * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_conv1d.bin", blob_dir, layer_idx);
        float *conv1d = orion_read_blob_f32_exact(path, (key_dim * 2 + value_dim) * conv_kernel);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_dt_bias.bin", blob_dir, layer_idx);
        float *dt_bias = orion_read_blob_f32_exact(path, num_v_heads);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_a_log.bin", blob_dir, layer_idx);
        float *a_log = orion_read_blob_f32_exact(path, num_v_heads);

        if (!in_proj_qkv || !in_proj_z || !in_proj_a || !in_proj_b || !conv1d || !dt_bias || !a_log) {
            fprintf(stderr, "FAIL: failed to read one or more linear-attention tensors\n");
            free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
            free(conv1d); free(dt_bias); free(a_log);
            goto fail;
        }

        orion_qwen_cpu_linear_attention_prep(
            x_seq,
            seq_len,
            in_proj_qkv,
            in_proj_z,
            in_proj_a,
            in_proj_b,
            conv1d,
            dt_bias,
            a_log,
            d_model,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_kernel,
            query,
            key,
            value,
            z,
            beta,
            g
        );

        free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
        free(conv1d); free(dt_bias); free(a_log);

        double q_abs = abs_sum(query, seq_len * key_dim);
        double k_abs = abs_sum(key, seq_len * key_dim);
        double v_abs = abs_sum(value, seq_len * value_dim);
        double z_abs = abs_sum(z, seq_len * value_dim);
        double beta_abs = abs_sum(beta, seq_len * num_v_heads);
        double g_abs = abs_sum(g, seq_len * num_v_heads);

        if (q_abs <= 0.0 || k_abs <= 0.0 || v_abs <= 0.0 || z_abs <= 0.0 || beta_abs <= 0.0 || g_abs <= 0.0) {
            fprintf(stderr, "FAIL: expected non-zero linear attention prep activations\n");
            goto fail;
        }

        printf("PASS: qwen35 linear attention prep cpu smoke\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  layer_idx=%d\n", layer_idx);
        printf("  seq_len=%d\n", seq_len);
        printf("  key_dim=%d\n", key_dim);
        printf("  value_dim=%d\n", value_dim);
        printf("  query_abs_sum=%.6f\n", q_abs);
        printf("  key_abs_sum=%.6f\n", k_abs);
        printf("  value_abs_sum=%.6f\n", v_abs);
        printf("  z_abs_sum=%.6f\n", z_abs);
        printf("  beta_abs_sum=%.6f\n", beta_abs);
        printf("  g_abs_sum=%.6f\n", g_abs);
        printf("  next_blocker=%s\n", "implement gated delta recurrent core and hybrid attention dispatch");

        free(x_seq);
        free(query);
        free(key);
        free(value);
        free(z);
        free(beta);
        free(g);
        orion_qwen35_manifest_free(manifest);
        return 0;

fail:
        free(x_seq);
        free(query);
        free(key);
        free(value);
        free(z);
        free(beta);
        free(g);
        orion_qwen35_manifest_free(manifest);
        return 1;
    }
}
