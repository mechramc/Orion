#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>
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

static float *load_exact(const char *blob_dir, int layer_idx, const char *suffix, int count) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/layer%d/%s", blob_dir, layer_idx, suffix);
    return orion_read_blob_f32_exact(path, count);
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

        const int seq_len = 2;
        const int d_model = manifest->d_model;
        const int d_ff = manifest->d_ff;
        const int n_head = manifest->n_head;
        const int n_kv_head = manifest->n_kv_head;
        const int head_dim = manifest->head_dim;
        const int q_dim = n_head * head_dim;
        const int kv_dim = n_kv_head * head_dim;
        const int num_k_heads = 16;
        const int num_v_heads = 16;
        const int head_k_dim = 128;
        const int head_v_dim = 128;
        const int value_dim = num_v_heads * head_v_dim;
        const int conv_kernel = 4;

        float *hidden = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *normed = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *mixer_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *mlp_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *scratch = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        if (!hidden || !normed || !mixer_out || !mlp_out || !scratch) {
            fprintf(stderr, "FAIL: allocation failed\n");
            goto fail;
        }

        char embed_path[2048];
        snprintf(embed_path, sizeof(embed_path), "%s/model/embed_tokens.bin", blob_dir);
        if (!orion_read_blob_row_f32(embed_path, 0, d_model, hidden + 0 * d_model) ||
            !orion_read_blob_row_f32(embed_path, 1, d_model, hidden + 1 * d_model)) {
            fprintf(stderr, "FAIL: failed to read token embeddings\n");
            goto fail;
        }

        const int layer_indices[4] = {0, 1, 2, 3};
        for (int idx = 0; idx < 4; idx++) {
            int layer_idx = layer_indices[idx];

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

            for (int s = 0; s < seq_len; s++) {
                orion_qwen_cpu_rmsnorm(hidden + s * d_model, input_ln, d_model, 1e-6f, normed + s * d_model);
            }

            memset(mixer_out, 0, (size_t)seq_len * d_model * sizeof(float));

            if (layer_idx == 3) {
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
                float *in_proj_qkv = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_qkv.bin", (value_dim + value_dim + value_dim) * d_model);
                float *in_proj_z = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_z.bin", value_dim * d_model);
                float *in_proj_a = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_a.bin", num_v_heads * d_model);
                float *in_proj_b = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_b.bin", num_v_heads * d_model);
                float *conv1d = load_exact(blob_dir, layer_idx, "linear_attn_conv1d.bin", (value_dim + value_dim + value_dim) * conv_kernel);
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

            for (int i = 0; i < seq_len * d_model; i++) {
                hidden[i] += mixer_out[i];
            }

            for (int s = 0; s < seq_len; s++) {
                orion_qwen_cpu_rmsnorm(hidden + s * d_model, post_ln, d_model, 1e-6f, scratch + s * d_model);
                orion_qwen_cpu_swiglu_ffn(scratch + s * d_model, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out + s * d_model);
            }

            for (int i = 0; i < seq_len * d_model; i++) {
                hidden[i] += mlp_out[i];
            }

            free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
        }

        double hidden_abs = abs_sum(hidden, seq_len * d_model);
        if (hidden_abs <= 0.0) {
            fprintf(stderr, "FAIL: expected non-zero hybrid dispatch output\n");
            goto fail;
        }

        printf("PASS: qwen35 hybrid dispatch cpu smoke\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  seq_len=%d\n", seq_len);
        printf("  layers_run=%d\n", 4);
        printf("  last_full_layer_idx=%d\n", 3);
        printf("  hidden_abs_sum=%.6f\n", hidden_abs);
        printf("  next_blocker=%s\n", "implement full 24-layer 1-token logits smoke");

        free(hidden);
        free(normed);
        free(mixer_out);
        free(mlp_out);
        free(scratch);
        orion_qwen35_manifest_free(manifest);
        return 0;

fail:
        free(hidden);
        free(normed);
        free(mixer_out);
        free(mlp_out);
        free(scratch);
        orion_qwen35_manifest_free(manifest);
        return 1;
    }
}
