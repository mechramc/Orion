#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import "../model/weight_loader.h"
#import "../kernels/training/qwen_lora_train.h"

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s <blob_dir> <adapter_dir>\n", argv[0]);
            return 2;
        }

        const char *blob_dir = argv[1];
        const char *adapter_dir = argv[2];
        OrionQwen35Manifest *manifest = orion_qwen35_manifest_load(blob_dir);
        if (!manifest) {
            fprintf(stderr, "FAIL: manifest load failed\n");
            return 1;
        }

        OrionQwen9BLoRATrainer lhs;
        OrionQwen9BLoRATrainer rhs;
        orion_qwen9b_lora_trainer_init(&lhs, manifest, manifest->n_layer - 1, 8, 16.0f, 1e-3f, 20260314u);
        orion_qwen9b_lora_trainer_init(&rhs, manifest, manifest->n_layer - 1, 8, 16.0f, 1e-3f, 1u);
        lhs.step = 3;
        if (!orion_qwen9b_lora_trainer_save(&lhs, adapter_dir)) {
            fprintf(stderr, "FAIL: save failed\n");
            goto fail;
        }
        if (!orion_qwen9b_lora_trainer_load(&rhs, adapter_dir)) {
            fprintf(stderr, "FAIL: load failed\n");
            goto fail;
        }
        if (!orion_qwen9b_lora_trainer_compare(&lhs, &rhs, 0.0f)) {
            fprintf(stderr, "FAIL: roundtrip mismatch\n");
            goto fail;
        }

        printf("PASS: qwen35 9b lora adapter roundtrip\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  adapter_dir=%s\n", adapter_dir);
        printf("  rank=%d\n", lhs.q_proj.rank);
        printf("  step=%d\n", rhs.step);

        orion_qwen9b_lora_trainer_free(&lhs);
        orion_qwen9b_lora_trainer_free(&rhs);
        orion_qwen35_manifest_free(manifest);
        return 0;

fail:
        orion_qwen9b_lora_trainer_free(&lhs);
        orion_qwen9b_lora_trainer_free(&rhs);
        orion_qwen35_manifest_free(manifest);
        return 1;
    }
}
