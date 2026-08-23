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
        const int d_ff = manifest->d_ff;
        const int n_head = manifest->n_head;
        const int n_kv_head = manifest->n_kv_head;
        const int head_dim = manifest->head_dim;
        const int q_dim = n_head * head_dim;
        const int kv_dim = n_kv_head * head_dim;

        float *x_seq = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *attn_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *residual = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *post_ln_weight = (float *)calloc((size_t)d_model, sizeof(float));
        float *rms_buf = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *mlp_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        if (!x_seq || !attn_out || !residual || !post_ln_weight || !rms_buf || !mlp_out) {
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
        snprintf(path, sizeof(path), "%s/layer%d/post_attention_layernorm.bin", blob_dir, layer_idx);
        float *post_ln = orion_read_blob_f32_exact(path, d_model);
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
        snprintf(path, sizeof(path), "%s/layer%d/mlp_gate_proj.bin", blob_dir, layer_idx);
        float *gate_proj = orion_read_blob_f32_exact(path, d_ff * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/mlp_up_proj.bin", blob_dir, layer_idx);
        float *up_proj = orion_read_blob_f32_exact(path, d_ff * d_model);
        snprintf(path, sizeof(path), "%s/layer%d/mlp_down_proj.bin", blob_dir, layer_idx);
        float *down_proj = orion_read_blob_f32_exact(path, d_model * d_ff);

        if (!input_ln || !post_ln || !q_proj || !k_proj || !v_proj || !o_proj ||
            !q_norm || !k_norm || !gate_proj || !up_proj || !down_proj) {
            fprintf(stderr, "FAIL: failed to read one or more full-attention layer tensors\n");
            free(input_ln); free(post_ln); free(q_proj); free(k_proj); free(v_proj); free(o_proj);
            free(q_norm); free(k_norm); free(gate_proj); free(up_proj); free(down_proj);
            goto fail;
        }

        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(x_seq + s * d_model, input_ln, d_model, 1e-6f, rms_buf + s * d_model);
        }

        orion_qwen_cpu_full_attention_prefill_no_rope(
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
            attn_out
        );

        for (int i = 0; i < seq_len * d_model; i++) {
            residual[i] = x_seq[i] + attn_out[i];
        }

        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(residual + s * d_model, post_ln, d_model, 1e-6f, rms_buf + s * d_model);
            orion_qwen_cpu_swiglu_ffn(rms_buf + s * d_model, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out + s * d_model);
        }

        double input_abs = abs_sum(x_seq, seq_len * d_model);
        double attn_abs = abs_sum(attn_out, seq_len * d_model);
        double residual_abs = abs_sum(residual, seq_len * d_model);
        double mlp_abs = abs_sum(mlp_out, seq_len * d_model);

        free(input_ln); free(post_ln); free(q_proj); free(k_proj); free(v_proj); free(o_proj);
        free(q_norm); free(k_norm); free(gate_proj); free(up_proj); free(down_proj);

        if (input_abs <= 0.0 || attn_abs <= 0.0 || residual_abs <= 0.0 || mlp_abs <= 0.0) {
            fprintf(stderr, "FAIL: expected non-zero attention/MLP activations\n");
            goto fail;
        }

        printf("PASS: qwen35 full attention cpu smoke no rope\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  layer_idx=%d\n", layer_idx);
        printf("  seq_len=%d\n", seq_len);
        printf("  q_dim=%d\n", q_dim);
        printf("  kv_dim=%d\n", kv_dim);
        printf("  input_abs_sum=%.6f\n", input_abs);
        printf("  attn_abs_sum=%.6f\n", attn_abs);
        printf("  residual_abs_sum=%.6f\n", residual_abs);
        printf("  mlp_abs_sum=%.6f\n", mlp_abs);
        printf("  next_blocker=%s\n", "implement rotary embeddings and linear_attention CPU kernels");

        free(x_seq);
        free(attn_out);
        free(residual);
        free(post_ln_weight);
        free(rms_buf);
        free(mlp_out);
        orion_qwen35_manifest_free(manifest);
        return 0;

fail:
        free(x_seq);
        free(attn_out);
        free(residual);
        free(post_ln_weight);
        free(rms_buf);
        free(mlp_out);
        orion_qwen35_manifest_free(manifest);
        return 1;
    }
}
