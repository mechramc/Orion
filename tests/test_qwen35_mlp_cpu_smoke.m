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

        int d_model = manifest->d_model;
        int d_ff = manifest->d_ff;
        float *embed = (float *)calloc((size_t)d_model, sizeof(float));
        float *rms_weight = (float *)calloc((size_t)d_model, sizeof(float));
        float *normed = (float *)calloc((size_t)d_model, sizeof(float));
        float *mlp_out = (float *)calloc((size_t)d_model, sizeof(float));
        if (!embed || !rms_weight || !normed || !mlp_out) {
            fprintf(stderr, "FAIL: allocation failed\n");
            goto fail;
        }

        char path[2048];
        snprintf(path, sizeof(path), "%s/model/embed_tokens.bin", blob_dir);
        if (!orion_read_blob_row_f32(path, 0, d_model, embed)) {
            fprintf(stderr, "FAIL: failed to read embedding row\n");
            goto fail;
        }

        snprintf(path, sizeof(path), "%s/layer0/input_layernorm.bin", blob_dir);
        if (!orion_read_blob_row_f32(path, 0, d_model, rms_weight)) {
            fprintf(stderr, "FAIL: failed to read layer0 RMSNorm\n");
            goto fail;
        }

        snprintf(path, sizeof(path), "%s/layer0/mlp_gate_proj.bin", blob_dir);
        float *gate_proj = orion_read_blob_f32_exact(path, d_ff * d_model);
        if (!gate_proj) {
            fprintf(stderr, "FAIL: failed to read gate_proj\n");
            goto fail;
        }

        snprintf(path, sizeof(path), "%s/layer0/mlp_up_proj.bin", blob_dir);
        float *up_proj = orion_read_blob_f32_exact(path, d_ff * d_model);
        if (!up_proj) {
            free(gate_proj);
            fprintf(stderr, "FAIL: failed to read up_proj\n");
            goto fail;
        }

        snprintf(path, sizeof(path), "%s/layer0/mlp_down_proj.bin", blob_dir);
        float *down_proj = orion_read_blob_f32_exact(path, d_model * d_ff);
        if (!down_proj) {
            free(gate_proj);
            free(up_proj);
            fprintf(stderr, "FAIL: failed to read down_proj\n");
            goto fail;
        }

        orion_qwen_cpu_rmsnorm(embed, rms_weight, d_model, 1e-6f, normed);
        orion_qwen_cpu_swiglu_ffn(normed, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out);

        double embed_abs = abs_sum(embed, d_model);
        double normed_abs = abs_sum(normed, d_model);
        double mlp_out_abs = abs_sum(mlp_out, d_model);

        free(gate_proj);
        free(up_proj);
        free(down_proj);

        if (embed_abs <= 0.0 || normed_abs <= 0.0 || mlp_out_abs <= 0.0) {
            fprintf(stderr, "FAIL: expected non-zero activation path\n");
            goto fail;
        }

        printf("PASS: qwen35 mlp cpu smoke\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  token_id=%d\n", 0);
        printf("  embed_abs_sum=%.6f\n", embed_abs);
        printf("  normed_abs_sum=%.6f\n", normed_abs);
        printf("  mlp_out_abs_sum=%.6f\n", mlp_out_abs);
        printf("  next_blocker=%s\n", "implement attention kernels for linear/full layers");

        free(embed);
        free(rms_weight);
        free(normed);
        free(mlp_out);
        orion_qwen35_manifest_free(manifest);
        return 0;

fail:
        free(embed);
        free(rms_weight);
        free(normed);
        free(mlp_out);
        orion_qwen35_manifest_free(manifest);
        return 1;
    }
}
