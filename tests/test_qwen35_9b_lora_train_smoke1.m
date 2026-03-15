#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import "../model/weight_loader.h"
#import "../kernels/training/qwen_lora_train.h"

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s <blob_dir> <adapter_out_dir>\n", argv[0]);
            return 2;
        }

        const char *blob_dir = argv[1];
        const char *adapter_out_dir = argv[2];
        OrionQwen35Manifest *manifest = orion_qwen35_manifest_load(blob_dir);
        if (!manifest) {
            fprintf(stderr, "FAIL: manifest load failed\n");
            return 1;
        }

        OrionQwen9BLoRATrainer trainer;
        orion_qwen9b_lora_trainer_init(&trainer, manifest, manifest->n_layer - 1, 8, 16.0f, 1e-3f, 1337u);
        char embed_path[2048];
        snprintf(embed_path, sizeof(embed_path), "%s/model/embed_tokens.bin", blob_dir);
        if (!orion_qwen9b_lora_trainer_attach_ce_context(&trainer, embed_path, manifest)) {
            fprintf(stderr, "FAIL: trainer CE context attach failed\n");
            orion_qwen9b_lora_trainer_free(&trainer);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }
        OrionQwen9BLoRASmokeResult result;
        if (!orion_qwen9b_lora_train_smoke1(blob_dir, &trainer, 0, 1, &result)) {
            fprintf(stderr, "FAIL: smoke1 train step failed\n");
            orion_qwen9b_lora_trainer_free(&trainer);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        if (!isfinite(result.loss) || result.loss <= 0.0f) {
            fprintf(stderr, "FAIL: non-finite loss %.6f\n", result.loss);
            orion_qwen9b_lora_trainer_free(&trainer);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }
        if (result.q_param_abs_sum <= 0.0 || result.v_param_abs_sum <= 0.0) {
            fprintf(stderr, "FAIL: non-positive parameter abs sums\n");
            orion_qwen9b_lora_trainer_free(&trainer);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }
        if (!orion_qwen9b_lora_trainer_save(&trainer, adapter_out_dir)) {
            fprintf(stderr, "FAIL: adapter save failed\n");
            orion_qwen9b_lora_trainer_free(&trainer);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        printf("PASS: qwen35 9b lora train smoke1\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  adapter_out_dir=%s\n", adapter_out_dir);
        printf("  layer_idx=%d\n", trainer.layer_idx);
        printf("  rank=%d\n", trainer.q_proj.rank);
        printf("  alpha=%.1f\n", trainer.q_proj.alpha);
        printf("  step=%d\n", trainer.step);
        printf("  loss=%.6f\n", result.loss);
        printf("  q_grad_abs_sum=%.6f\n", result.q_grad_abs_sum);
        printf("  v_grad_abs_sum=%.6f\n", result.v_grad_abs_sum);
        printf("  q_param_abs_sum=%.6f\n", result.q_param_abs_sum);
        printf("  v_param_abs_sum=%.6f\n", result.v_param_abs_sum);
        printf("  next_blocker=%s\n", "smoke10 and adapter reload drift");

        orion_qwen9b_lora_trainer_free(&trainer);
        orion_qwen35_manifest_free(manifest);
        return 0;
    }
}
