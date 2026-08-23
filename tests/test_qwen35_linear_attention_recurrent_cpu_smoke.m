#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
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
        const int value_dim = num_v_heads * head_v_dim;
        const int conv_kernel = 4;

        float *x_seq = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *out_seq = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        if (!x_seq || !out_seq) {
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
        float *in_proj_qkv = orion_read_blob_f32_exact(path, (value_dim + value_dim + value_dim) * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_in_proj_z.bin", blob_dir, layer_idx);
        float *in_proj_z = orion_read_blob_f32_exact(path, value_dim * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_in_proj_a.bin", blob_dir, layer_idx);
        float *in_proj_a = orion_read_blob_f32_exact(path, num_v_heads * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_in_proj_b.bin", blob_dir, layer_idx);
        float *in_proj_b = orion_read_blob_f32_exact(path, num_v_heads * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_conv1d.bin", blob_dir, layer_idx);
        float *conv1d = orion_read_blob_f32_exact(path, (value_dim + value_dim + value_dim) * conv_kernel);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_dt_bias.bin", blob_dir, layer_idx);
        float *dt_bias = orion_read_blob_f32_exact(path, num_v_heads);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_a_log.bin", blob_dir, layer_idx);
        float *a_log = orion_read_blob_f32_exact(path, num_v_heads);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_norm.bin", blob_dir, layer_idx);
        float *norm_weight = orion_read_blob_f32_exact(path, head_v_dim);
        snprintf(path, sizeof(path), "%s/layer%d/linear_attn_out_proj.bin", blob_dir, layer_idx);
        float *out_proj = orion_read_blob_f32_exact(path, d_model * value_dim);

        if (!in_proj_qkv || !in_proj_z || !in_proj_a || !in_proj_b || !conv1d ||
            !dt_bias || !a_log || !norm_weight || !out_proj) {
            fprintf(stderr, "FAIL: failed to read one or more linear-attention recurrent tensors\n");
            free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
            free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
            goto fail;
        }

        orion_qwen_cpu_linear_attention_recurrent_prefill(
            x_seq, seq_len,
            in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
            conv1d, dt_bias, a_log, norm_weight, out_proj,
            d_model, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel,
            out_seq
        );

        free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
        free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);

        double out_abs = abs_sum(out_seq, seq_len * d_model);
        if (out_abs <= 0.0) {
            fprintf(stderr, "FAIL: expected non-zero recurrent linear attention output\n");
            goto fail;
        }

        printf("PASS: qwen35 linear attention recurrent cpu smoke\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  layer_idx=%d\n", layer_idx);
        printf("  seq_len=%d\n", seq_len);
        printf("  out_abs_sum=%.6f\n", out_abs);
        printf("  next_blocker=%s\n", "implement rotary embeddings and hybrid 1-token logits dispatch");

        free(x_seq);
        free(out_seq);
        orion_qwen35_manifest_free(manifest);
        return 0;

fail:
        free(x_seq);
        free(out_seq);
        orion_qwen35_manifest_free(manifest);
        return 1;
    }
}
