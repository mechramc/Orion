#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>
#import "../model/weight_loader.h"
#import "../kernels/training/qwen_lora_train.h"
#import "../kernels/inference/qwen_cpu_ops.h"

static float abs_sum(const float *x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)x[i]);
    return (float)total;
}

static float max_abs(const float *x, int n) {
    float best = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabsf(x[i]);
        if (v > best) best = v;
    }
    return best;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <blob_dir> [input_token]\n", argv[0]);
            return 2;
        }

        const char *blob_dir = argv[1];
        const int input_token = argc >= 3 ? atoi(argv[2]) : 0;
        OrionQwen35Manifest *manifest = orion_qwen35_manifest_load(blob_dir);
        if (!manifest) {
            fprintf(stderr, "FAIL: manifest load failed\n");
            return 1;
        }

        const int layer_idx = manifest->n_layer - 1;
        const int d_model = manifest->d_model;
        float *hidden = (float *)calloc((size_t)d_model, sizeof(float));
        float *normed = (float *)calloc((size_t)d_model, sizeof(float));
        float *input_ln = NULL;
        if (!hidden || !normed) {
            fprintf(stderr, "FAIL: alloc failed\n");
            free(hidden);
            free(normed);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        char path[2048];
        snprintf(path, sizeof(path), "%s/layer%d/input_layernorm.bin", blob_dir, layer_idx);
        input_ln = orion_read_blob_f32_exact(path, d_model);
        if (!input_ln) {
            fprintf(stderr, "FAIL: input layernorm load failed\n");
            free(hidden);
            free(normed);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        if (!orion_qwen9b_lora_frozen_prefix_hidden(blob_dir, manifest, input_token, hidden)) {
            fprintf(stderr, "FAIL: frozen prefix hidden probe failed\n");
            free(input_ln);
            free(hidden);
            free(normed);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        orion_qwen_cpu_rmsnorm(hidden, input_ln, d_model, 1e-6f, normed);

        printf("PASS: qwen35 9b lora ane forward probe\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  input_token=%d\n", input_token);
        printf("  layer_idx=%d\n", layer_idx);
        printf("  hidden_abs_sum=%.6f\n", abs_sum(hidden, d_model));
        printf("  hidden_max_abs=%.6f\n", max_abs(hidden, d_model));
        printf("  normed_abs_sum=%.6f\n", abs_sum(normed, d_model));
        printf("  normed_max_abs=%.6f\n", max_abs(normed, d_model));
        printf("  next_blocker=%s\n", "wire ANE q_proj/kv_proj forward on top of frozen prefix hidden");

        free(input_ln);
        free(hidden);
        free(normed);
        orion_qwen35_manifest_free(manifest);
        return 0;
    }
}
