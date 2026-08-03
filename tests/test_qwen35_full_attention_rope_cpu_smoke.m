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

        const int layer_idx = 3;
        const int seq_len = 2;
        const int d_model = manifest->d_model;
        const int n_head = manifest->n_head;
        const int n_kv_head = manifest->n_kv_head;
        const int head_dim = manifest->head_dim;
        const int q_dim = n_head * head_dim;
        const int kv_dim = n_kv_head * head_dim;

        float *x_seq = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *rms_buf = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *attn_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        if (!x_seq || !rms_buf || !attn_out) {
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

        snprintf(path, sizeof(path), "%s/layer%d/input_layernorm.bin", blob_dir, layer_idx);
        float *input_ln = orion_read_blob_f32_exact(path, d_model);
        snprintf(path, sizeof(path), "%s/layer%d/self_attn_q_proj.bin", blob_dir, layer_idx);
        float *q_proj = orion_read_blob_f32_exact(path, (q_dim * 2) * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/self_attn_k_proj.bin", blob_dir, layer_idx);
        float *k_proj = orion_read_blob_f32_exact(path, kv_dim * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/self_attn_v_proj.bin", blob_dir, layer_idx);
        float *v_proj = orion_read_blob_f32_exact(path, kv_dim * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/self_attn_o_proj.bin", blob_dir, layer_idx);
        float *o_proj = orion_read_blob_f32_exact(path, d_model * q_dim);
        snprintf(path, sizeof(path), "%s/layer%d/self_attn_q_norm.bin", blob_dir, layer_idx);
        float *q_norm = orion_read_blob_f32_exact(path, head_dim);
        snprintf(path, sizeof(path), "%s/layer%d/self_attn_k_norm.bin", blob_dir, layer_idx);
        float *k_norm = orion_read_blob_f32_exact(path, head_dim);

        if (!input_ln || !q_proj || !k_proj || !v_proj || !o_proj || !q_norm || !k_norm) {
            fprintf(stderr, "FAIL: failed to read one or more full-attention rope tensors\n");
            free(input_ln); free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
            goto fail;
        }

        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(x_seq + s * d_model, input_ln, d_model, 1e-6f, rms_buf + s * d_model);
        }

        orion_qwen_cpu_full_attention_prefill_with_rope(
            rms_buf,
            seq_len,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            d_model,
            n_head,
            n_kv_head,
            head_dim,
            manifest->rope_theta,
            manifest->partial_rotary_factor,
            attn_out
        );

        double input_abs = abs_sum(x_seq, seq_len * d_model);
        double attn_abs = abs_sum(attn_out, seq_len * d_model);

        free(input_ln); free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);

        if (input_abs <= 0.0 || attn_abs <= 0.0) {
            fprintf(stderr, "FAIL: expected non-zero rope attention activations\n");
            goto fail;
        }

        printf("PASS: qwen35 full attention cpu smoke with rope\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  layer_idx=%d\n", layer_idx);
        printf("  seq_len=%d\n", seq_len);
        printf("  rope_theta=%.1f\n", manifest->rope_theta);
        printf("  partial_rotary_factor=%.4f\n", manifest->partial_rotary_factor);
        printf("  rotary_dim=%d\n", manifest->rotary_dim);
        printf("  input_abs_sum=%.6f\n", input_abs);
        printf("  attn_abs_sum=%.6f\n", attn_abs);
        printf("  next_blocker=%s\n", "implement hybrid decoder-layer dispatch");

        free(x_seq);
        free(rms_buf);
        free(attn_out);
        orion_qwen35_manifest_free(manifest);
        return 0;

fail:
        free(x_seq);
        free(rms_buf);
        free(attn_out);
        orion_qwen35_manifest_free(manifest);
        return 1;
    }
}
