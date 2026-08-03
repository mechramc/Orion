#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import "../model/weight_loader.h"

static double abs_sum(const float *x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        total += fabs((double)x[i]);
    }
    return total;
}

static int file_exists(const char *path) {
    return [[NSFileManager defaultManager] fileExistsAtPath:[NSString stringWithUTF8String:path]];
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

        float *embed_row = (float *)calloc((size_t)manifest->d_model, sizeof(float));
        float *final_norm = (float *)calloc((size_t)manifest->d_model, sizeof(float));
        float *input_layernorm = (float *)calloc((size_t)manifest->d_model, sizeof(float));
        if (!embed_row || !final_norm || !input_layernorm) {
            fprintf(stderr, "FAIL: allocation failed\n");
            free(embed_row);
            free(final_norm);
            free(input_layernorm);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        char path[2048];
        snprintf(path, sizeof(path), "%s/model/embed_tokens.bin", blob_dir);
        if (!orion_read_blob_row_f32(path, 0, manifest->d_model, embed_row)) {
            fprintf(stderr, "FAIL: failed to read embedding row\n");
            goto fail;
        }

        snprintf(path, sizeof(path), "%s/model/final_norm.bin", blob_dir);
        if (!orion_read_blob_row_f32(path, 0, manifest->d_model, final_norm)) {
            fprintf(stderr, "FAIL: failed to read final norm\n");
            goto fail;
        }

        snprintf(path, sizeof(path), "%s/layer0/input_layernorm.bin", blob_dir);
        if (!orion_read_blob_row_f32(path, 0, manifest->d_model, input_layernorm)) {
            fprintf(stderr, "FAIL: failed to read layer0 input layernorm\n");
            goto fail;
        }

        int detected_linear = 0;
        int detected_full = 0;
        int first_linear_idx = -1;
        int first_full_idx = -1;
        for (int i = 0; i < manifest->n_layer; i++) {
            char linear_qkv[2048];
            char full_q[2048];
            snprintf(linear_qkv, sizeof(linear_qkv), "%s/layer%d/linear_attn_in_proj_qkv.bin", blob_dir, i);
            snprintf(full_q, sizeof(full_q), "%s/layer%d/self_attn_q_proj.bin", blob_dir, i);
            if (file_exists(linear_qkv)) {
                detected_linear += 1;
                if (first_linear_idx < 0) first_linear_idx = i;
            } else if (file_exists(full_q)) {
                detected_full += 1;
                if (first_full_idx < 0) first_full_idx = i;
            } else {
                fprintf(stderr, "FAIL: layer%d has neither linear nor full attention blobs\n", i);
                goto fail;
            }
        }

        double embed_abs = abs_sum(embed_row, manifest->d_model);
        double final_norm_abs = abs_sum(final_norm, manifest->d_model);
        double input_ln_abs = abs_sum(input_layernorm, manifest->d_model);

        if (embed_abs <= 0.0 || final_norm_abs <= 0.0 || input_ln_abs <= 0.0) {
            fprintf(stderr, "FAIL: expected non-zero tensor content\n");
            goto fail;
        }
        if (detected_linear != manifest->n_linear_layers || detected_full != manifest->n_full_layers) {
            fprintf(stderr, "FAIL: layer topology mismatch (detected=%d/%d manifest=%d/%d)\n",
                    detected_linear, detected_full, manifest->n_linear_layers, manifest->n_full_layers);
            goto fail;
        }

        printf("PASS: qwen35 cpu forward scaffold\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  token_id=%d\n", 0);
        printf("  embed_abs_sum=%.6f\n", embed_abs);
        printf("  final_norm_abs_sum=%.6f\n", final_norm_abs);
        printf("  layer0_input_layernorm_abs_sum=%.6f\n", input_ln_abs);
        printf("  detected_linear_layers=%d\n", detected_linear);
        printf("  detected_full_layers=%d\n", detected_full);
        printf("  first_linear_layer_idx=%d\n", first_linear_idx);
        printf("  first_full_layer_idx=%d\n", first_full_idx);
        printf("  next_blocker=%s\n", "implement linear_attention/full_attention CPU kernels");

        free(embed_row);
        free(final_norm);
        free(input_layernorm);
        orion_qwen35_manifest_free(manifest);
        return 0;

fail:
        free(embed_row);
        free(final_norm);
        free(input_layernorm);
        orion_qwen35_manifest_free(manifest);
        return 1;
    }
}
