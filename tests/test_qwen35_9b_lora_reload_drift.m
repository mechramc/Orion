#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
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
        orion_qwen9b_lora_trainer_init(&lhs, manifest, manifest->n_layer - 1, 8, 16.0f, 1e-3f, 1337u);
        orion_qwen9b_lora_trainer_init(&rhs, manifest, manifest->n_layer - 1, 8, 16.0f, 1e-3f, 1337u);
        int ok = orion_qwen9b_lora_trainer_load(&lhs, adapter_dir) &&
                 orion_qwen9b_lora_trainer_load(&rhs, adapter_dir) &&
                 orion_qwen9b_lora_trainer_compare(&lhs, &rhs, 1e-6f);

        printf("%s: qwen35 9b lora reload drift\n", ok ? "PASS" : "FAIL");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  adapter_dir=%s\n", adapter_dir);
        printf("  step=%d\n", lhs.step);
        printf("  layer_idx=%d\n", lhs.layer_idx);
        printf("  q_param_abs_sum=%.6f\n", orion_qwen_lora_abs_sum(lhs.q_proj.a, lhs.q_proj.rank * lhs.q_proj.in_dim) +
                                         orion_qwen_lora_abs_sum(lhs.q_proj.b, lhs.q_proj.out_dim * lhs.q_proj.rank));
        printf("  v_param_abs_sum=%.6f\n", orion_qwen_lora_abs_sum(lhs.v_proj.a, lhs.v_proj.rank * lhs.v_proj.in_dim) +
                                         orion_qwen_lora_abs_sum(lhs.v_proj.b, lhs.v_proj.out_dim * lhs.v_proj.rank));
        printf("  compare_ok=%d\n", ok);

        orion_qwen9b_lora_trainer_free(&lhs);
        orion_qwen9b_lora_trainer_free(&rhs);
        orion_qwen35_manifest_free(manifest);
        return ok ? 0 : 1;
    }
}
