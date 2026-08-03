#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import "../model/weight_loader.h"
#import "../kernels/training/qwen_lora_train.h"

static NSArray<NSDictionary *> *load_pairs(NSString *path) {
    NSString *text = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:nil];
    if (!text) return nil;
    NSMutableArray<NSDictionary *> *items = [NSMutableArray array];
    [text enumerateLinesUsingBlock:^(NSString * _Nonnull line, BOOL * _Nonnull stop) {
        (void)stop;
        if (line.length == 0) return;
        NSData *data = [line dataUsingEncoding:NSUTF8StringEncoding];
        NSDictionary *obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if ([obj isKindOfClass:[NSDictionary class]]) {
            [items addObject:obj];
        }
    }];
    return items;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 5) {
            fprintf(stderr, "usage: %s <blob_dir> <pairs_jsonl> <adapter_out_dir> <max_steps> [resume_adapter_dir]\n", argv[0]);
            return 2;
        }

        const char *blob_dir = argv[1];
        NSString *pairsPath = [NSString stringWithUTF8String:argv[2]];
        const char *adapter_out_dir = argv[3];
        int max_steps = atoi(argv[4]);
        const char *resume_adapter_dir = argc >= 6 ? argv[5] : NULL;
        if (max_steps <= 0) {
            fprintf(stderr, "FAIL: max_steps must be > 0\n");
            return 1;
        }

        NSArray<NSDictionary *> *pairs = load_pairs(pairsPath);
        if (!pairs || pairs.count == 0) {
            fprintf(stderr, "FAIL: pair file empty or unreadable\n");
            return 1;
        }

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
        int resume_loaded = 0;
        int resume_start_step = trainer.step;
        if (resume_adapter_dir && strlen(resume_adapter_dir) > 0) {
            if (!orion_qwen9b_lora_trainer_load(&trainer, resume_adapter_dir)) {
                fprintf(stderr, "FAIL: resume adapter load failed\n");
                orion_qwen9b_lora_trainer_free(&trainer);
                orion_qwen35_manifest_free(manifest);
                return 1;
            }
            resume_loaded = 1;
            resume_start_step = trainer.step;
        }

        double loss_sum = 0.0;
        float loss_first = NAN;
        float loss_last = NAN;
        float loss_min = INFINITY;
        float loss_max = -INFINITY;
        double q_grad_last = 0.0;
        double v_grad_last = 0.0;
        double q_param_last = 0.0;
        double v_param_last = 0.0;
        int steps_completed = 0;
        NSMutableSet<NSString *> *sampleIds = [NSMutableSet set];

        for (NSDictionary *pair in pairs) {
            if (steps_completed >= max_steps) break;
            int inputToken = [pair[@"input_token"] intValue];
            int targetToken = [pair[@"target_token"] intValue];
            NSString *sampleId = pair[@"sample_id"] ?: @"unknown";
            [sampleIds addObject:sampleId];

            OrionQwen9BLoRASmokeResult result;
            if (!orion_qwen9b_lora_train_smoke1_ane_qv_base(blob_dir, &trainer, inputToken, targetToken, &result)) {
                fprintf(stderr, "FAIL: ane pair step failed at step %d for sample %s\n", steps_completed + 1, sampleId.UTF8String);
                orion_qwen9b_lora_trainer_free(&trainer);
                orion_qwen35_manifest_free(manifest);
                return 1;
            }
            if (!isfinite(result.loss)) {
                fprintf(stderr, "FAIL: non-finite ane loss at step %d\n", steps_completed + 1);
                orion_qwen9b_lora_trainer_free(&trainer);
                orion_qwen35_manifest_free(manifest);
                return 1;
            }

            if (steps_completed == 0) loss_first = result.loss;
            loss_last = result.loss;
            if (result.loss < loss_min) loss_min = result.loss;
            if (result.loss > loss_max) loss_max = result.loss;
            loss_sum += result.loss;
            q_grad_last = result.q_grad_abs_sum;
            v_grad_last = result.v_grad_abs_sum;
            q_param_last = result.q_param_abs_sum;
            v_param_last = result.v_param_abs_sum;
            steps_completed += 1;
        }

        if (steps_completed == 0) {
            fprintf(stderr, "FAIL: zero ANE steps completed\n");
            orion_qwen9b_lora_trainer_free(&trainer);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        if (!orion_qwen9b_lora_trainer_save(&trainer, adapter_out_dir)) {
            fprintf(stderr, "FAIL: ANE adapter save failed\n");
            orion_qwen9b_lora_trainer_free(&trainer);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        OrionQwen9BLoRATrainer reloaded;
        orion_qwen9b_lora_trainer_init(&reloaded, manifest, manifest->n_layer - 1, 8, 16.0f, 1e-3f, 1337u);
        int reload_ok = orion_qwen9b_lora_trainer_load(&reloaded, adapter_out_dir) &&
                        orion_qwen9b_lora_trainer_compare(&trainer, &reloaded, 1e-6f);

        printf("PASS: qwen35 9b lora ane pair training\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  pairs_path=%s\n", pairsPath.UTF8String);
        printf("  adapter_out_dir=%s\n", adapter_out_dir);
        printf("  layer_idx=%d\n", trainer.layer_idx);
        printf("  rank=%d\n", trainer.q_proj.rank);
        printf("  alpha=%.1f\n", trainer.q_proj.alpha);
        printf("  steps_completed=%d\n", steps_completed);
        printf("  samples_seen=%lu\n", (unsigned long)sampleIds.count);
        printf("  resume_loaded=%d\n", resume_loaded);
        printf("  resume_start_step=%d\n", resume_start_step);
        printf("  trainer_step_final=%d\n", trainer.step);
        printf("  loss_first=%.6f\n", loss_first);
        printf("  loss_last=%.6f\n", loss_last);
        printf("  loss_min=%.6f\n", loss_min);
        printf("  loss_max=%.6f\n", loss_max);
        printf("  loss_avg=%.6f\n", (float)(loss_sum / (double)steps_completed));
        printf("  q_grad_abs_sum_last=%.6f\n", q_grad_last);
        printf("  v_grad_abs_sum_last=%.6f\n", v_grad_last);
        printf("  q_param_abs_sum_last=%.6f\n", q_param_last);
        printf("  v_param_abs_sum_last=%.6f\n", v_param_last);
        printf("  reload_compare_ok=%d\n", reload_ok);
        printf("  compile_cache_hit=%d\n", orion_qwen9b_lora_ane_train_bridge_last_compile_cache_hit());
        printf("  compile_cache_q_hit=%d\n", orion_qwen9b_lora_ane_train_bridge_last_q_cache_hit());
        printf("  compile_cache_kv_hit=%d\n", orion_qwen9b_lora_ane_train_bridge_last_kv_cache_hit());
        printf("  compile_cache_source=%s\n", orion_qwen9b_lora_ane_train_bridge_last_compile_cache_source());
        printf("  next_blocker=%s\n", "ane longer canary/preflight and cpu-vs-ane drift review");

        orion_qwen9b_lora_trainer_free(&reloaded);
        orion_qwen9b_lora_trainer_free(&trainer);
        orion_qwen35_manifest_free(manifest);
        return reload_ok ? 0 : 1;
    }
}
