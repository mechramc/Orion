#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import "../model/weight_loader.h"
#import "../kernels/training/qwen_lora_train.h"

static NSArray<NSDictionary *> *load_sequences(NSString *path) {
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

static NSData *cached_hidden_for_token(NSMutableDictionary<NSNumber *, NSData *> *cache,
                                       const char *blob_dir,
                                       const OrionQwen35Manifest *manifest,
                                       int token_id) {
    NSNumber *key = @(token_id);
    NSData *hiddenData = cache[key];
    if (hiddenData) return hiddenData;

    size_t bytes = (size_t)manifest->d_model * sizeof(float);
    float *hidden = (float *)calloc((size_t)manifest->d_model, sizeof(float));
    if (!hidden) return nil;
    if (!orion_qwen9b_lora_frozen_prefix_hidden(blob_dir, manifest, token_id, hidden)) {
        free(hidden);
        return nil;
    }

    hiddenData = [NSData dataWithBytesNoCopy:hidden length:bytes freeWhenDone:YES];
    if (!hiddenData) {
        free(hidden);
        return nil;
    }
    cache[key] = hiddenData;
    return hiddenData;
}

static NSMutableDictionary<NSNumber *, NSData *> *load_hidden_cache_dir(NSString *cacheDir,
                                                                        const OrionQwen35Manifest *manifest) {
    if (!cacheDir) return nil;
    NSString *indexPath = [cacheDir stringByAppendingPathComponent:@"hidden_cache_index.json"];
    NSData *indexData = [NSData dataWithContentsOfFile:indexPath];
    if (!indexData) return nil;
    NSDictionary *index = [NSJSONSerialization JSONObjectWithData:indexData options:0 error:nil];
    if (![index isKindOfClass:[NSDictionary class]]) return nil;
    if ([index[@"d_model"] intValue] != manifest->d_model) return nil;

    NSString *binPath = index[@"bin_path"];
    NSData *binData = [NSData dataWithContentsOfFile:binPath options:NSDataReadingMappedIfSafe error:nil];
    NSArray *tokens = index[@"tokens"];
    NSArray *offsets = index[@"offsets"];
    NSArray *counts = index[@"counts"];
    if (!binData || ![tokens isKindOfClass:[NSArray class]] ||
        ![offsets isKindOfClass:[NSArray class]] || ![counts isKindOfClass:[NSArray class]] ||
        tokens.count != offsets.count || tokens.count != counts.count) {
        return nil;
    }

    NSMutableDictionary<NSNumber *, NSData *> *cache = [NSMutableDictionary dictionaryWithCapacity:tokens.count];
    for (NSUInteger idx = 0; idx < tokens.count; idx++) {
        NSNumber *token = tokens[idx];
        NSUInteger offset = (NSUInteger)[offsets[idx] unsignedLongLongValue];
        NSUInteger count = (NSUInteger)[counts[idx] unsignedLongLongValue];
        NSUInteger bytes = count * sizeof(float);
        if (offset + bytes > binData.length) return nil;
        NSData *hiddenData = [binData subdataWithRange:NSMakeRange(offset, bytes)];
        cache[token] = hiddenData;
    }
    return cache;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 5) {
            fprintf(stderr, "usage: %s <blob_dir> <sequences_jsonl> <adapter_out_dir> <max_sequences> [hidden_cache_dir] [grad_accum_sequences] [sequence_bucket] [resume_adapter_dir]\n", argv[0]);
            return 2;
        }

        const char *blob_dir = argv[1];
        NSString *sequencesPath = [NSString stringWithUTF8String:argv[2]];
        const char *adapter_out_dir = argv[3];
        int max_sequences = atoi(argv[4]);
        NSString *hiddenCacheDir = argc >= 6 ? [NSString stringWithUTF8String:argv[5]] : nil;
        int grad_accum_sequences = argc >= 7 ? atoi(argv[6]) : 4;
        int sequence_bucket = argc >= 8 ? atoi(argv[7]) : 64;
        const char *resume_adapter_dir = argc >= 9 ? argv[8] : NULL;
        if (max_sequences <= 0) {
            fprintf(stderr, "FAIL: max_sequences must be > 0\n");
            return 1;
        }
        if (grad_accum_sequences <= 0) {
            fprintf(stderr, "FAIL: grad_accum_sequences must be > 0\n");
            return 1;
        }
        if (sequence_bucket <= 0) {
            fprintf(stderr, "FAIL: sequence_bucket must be > 0\n");
            return 1;
        }

        NSArray<NSDictionary *> *sequences = load_sequences(sequencesPath);
        if (!sequences || sequences.count == 0) {
            fprintf(stderr, "FAIL: sequence file empty or unreadable\n");
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
        if (!orion_qwen9b_lora_trainer_attach_cpu_train_context(&trainer, blob_dir, manifest)) {
            fprintf(stderr, "FAIL: trainer CPU train context attach failed\n");
            orion_qwen9b_lora_trainer_free(&trainer);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }
        int resume_loaded = 0;
        int resume_start_step = trainer.step;
        if (resume_adapter_dir && strlen(resume_adapter_dir) > 0) {
            if (!orion_qwen9b_lora_trainer_load(&trainer, resume_adapter_dir)) {
                fprintf(stderr, "FAIL: sequence resume adapter load failed\n");
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
        int sequences_completed = 0;
        int pairs_seen = 0;
        int tokens_seen = 0;
        int packed_batches_completed = 0;
        int packed_batch_tokens_last = 0;
        int packed_batch_tokens_max = 0;
        int packed_sequences_last = 0;
        int packed_sequences_max = 0;
        NSMutableSet<NSString *> *sampleIds = [NSMutableSet set];
        NSMutableDictionary<NSNumber *, NSData *> *hiddenCache = load_hidden_cache_dir(hiddenCacheDir, manifest);
        if (!hiddenCache) hiddenCache = [NSMutableDictionary dictionary];
        NSUInteger hiddenCachePrefilled = hiddenCache.count;
        int hidden_cache_hits = 0;
        int hidden_cache_misses = 0;

        NSUInteger sequenceIndex = 0;
        while (sequenceIndex < sequences.count && sequences_completed < max_sequences) {
            NSMutableArray<NSDictionary *> *batchSequences = [NSMutableArray arrayWithCapacity:(NSUInteger)grad_accum_sequences];
            int batchPairs = 0;
            int batchTokens = 0;

            while (sequenceIndex < sequences.count &&
                   sequences_completed + (int)batchSequences.count < max_sequences &&
                   (int)batchSequences.count < grad_accum_sequences) {
                NSDictionary *sequence = sequences[sequenceIndex];
                sequenceIndex += 1;
                NSArray *tokenIds = sequence[@"token_ids"];
                if (![tokenIds isKindOfClass:[NSArray class]] || tokenIds.count < 2) continue;
                int seq_pairs = (int)tokenIds.count - 1;
                if (batchPairs > 0 && batchPairs + seq_pairs > sequence_bucket) {
                    sequenceIndex -= 1;
                    break;
                }
                [batchSequences addObject:sequence];
                batchPairs += seq_pairs;
                batchTokens += (int)tokenIds.count;
                if (batchPairs >= sequence_bucket) break;
            }

            if (batchSequences.count == 0 || batchPairs <= 0) {
                continue;
            }

            const float **hiddenBatch = (const float **)calloc((size_t)batchPairs, sizeof(float *));
            int *targetBatch = (int *)calloc((size_t)batchPairs, sizeof(int));
            if (!hiddenBatch || !targetBatch) {
                free(hiddenBatch);
                free(targetBatch);
                fprintf(stderr, "FAIL: sequence batch alloc failed at sequence %d\n", sequences_completed + 1);
                orion_qwen9b_lora_trainer_free(&trainer);
                orion_qwen35_manifest_free(manifest);
                return 1;
            }

            int writeIdx = 0;
            NSString *sampleLabel = @"packed-batch";
            for (NSDictionary *sequence in batchSequences) {
                NSArray *tokenIds = sequence[@"token_ids"];
                NSString *sampleId = sequence[@"sample_id"] ?: @"unknown";
                if (sampleLabel.length == 0 || [sampleLabel isEqualToString:@"packed-batch"]) {
                    sampleLabel = sampleId;
                }
                [sampleIds addObject:sampleId];
                for (NSUInteger idx = 0; idx + 1 < tokenIds.count; idx++) {
                    int inputToken = [tokenIds[idx] intValue];
                    int targetToken = [tokenIds[idx + 1] intValue];
                    int hadHidden = hiddenCache[@(inputToken)] != nil;
                    NSData *hiddenData = cached_hidden_for_token(hiddenCache, blob_dir, manifest, inputToken);
                    if (!hiddenData) {
                        free(hiddenBatch);
                        free(targetBatch);
                        fprintf(stderr, "FAIL: hidden cache build failed at sequence %d for token %d\n",
                                sequences_completed + 1, inputToken);
                        orion_qwen9b_lora_trainer_free(&trainer);
                        orion_qwen35_manifest_free(manifest);
                        return 1;
                    }
                    if (hadHidden) hidden_cache_hits += 1;
                    else hidden_cache_misses += 1;
                    hiddenBatch[writeIdx] = (const float *)hiddenData.bytes;
                    targetBatch[writeIdx] = targetToken;
                    writeIdx += 1;
                }
            }

            OrionQwen9BLoRABatchResult batchResult;
            if (!orion_qwen9b_lora_train_hidden_batch(blob_dir, manifest, &trainer,
                                                      hiddenBatch, targetBatch, batchPairs, &batchResult)) {
                free(hiddenBatch);
                free(targetBatch);
                fprintf(stderr, "FAIL: sequence packed batch failed at sequence %d for sample %s\n",
                        sequences_completed + 1, sampleLabel.UTF8String);
                orion_qwen9b_lora_trainer_free(&trainer);
                orion_qwen35_manifest_free(manifest);
                return 1;
            }
            free(hiddenBatch);
            free(targetBatch);

            if (batchResult.items_completed != batchPairs || !isfinite(batchResult.loss_avg)) {
                fprintf(stderr, "FAIL: invalid sequence packed batch result at sequence %d\n", sequences_completed + 1);
                orion_qwen9b_lora_trainer_free(&trainer);
                orion_qwen35_manifest_free(manifest);
                return 1;
            }

            float seq_loss_avg = batchResult.loss_avg;
            if (sequences_completed == 0) loss_first = seq_loss_avg;
            loss_last = seq_loss_avg;
            if (seq_loss_avg < loss_min) loss_min = seq_loss_avg;
            if (seq_loss_avg > loss_max) loss_max = seq_loss_avg;
            loss_sum += batchResult.loss_avg * (double)batchSequences.count;
            q_grad_last = batchResult.q_grad_abs_sum_last;
            v_grad_last = batchResult.v_grad_abs_sum_last;
            q_param_last = batchResult.q_param_abs_sum_last;
            v_param_last = batchResult.v_param_abs_sum_last;
            pairs_seen += batchPairs;
            packed_batches_completed += 1;
            packed_batch_tokens_last = batchPairs;
            if (batchPairs > packed_batch_tokens_max) packed_batch_tokens_max = batchPairs;
            packed_sequences_last = (int)batchSequences.count;
            if ((int)batchSequences.count > packed_sequences_max) packed_sequences_max = (int)batchSequences.count;

            sequences_completed += (int)batchSequences.count;
            tokens_seen += batchTokens;
            fprintf(stderr, "INFO: completed_sequence=%d/%d packed_sequences=%lu sample_id=%s tokens=%d packed_pairs=%d avg_loss=%.6f\n",
                    sequences_completed, max_sequences, (unsigned long)batchSequences.count, sampleLabel.UTF8String,
                    batchTokens, batchPairs, seq_loss_avg);
            fflush(stderr);
        }

        if (sequences_completed == 0) {
            fprintf(stderr, "FAIL: zero sequences completed\n");
            orion_qwen9b_lora_trainer_free(&trainer);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        if (!orion_qwen9b_lora_trainer_save(&trainer, adapter_out_dir)) {
            fprintf(stderr, "FAIL: sequence adapter save failed\n");
            orion_qwen9b_lora_trainer_free(&trainer);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        OrionQwen9BLoRATrainer reloaded;
        orion_qwen9b_lora_trainer_init(&reloaded, manifest, manifest->n_layer - 1, 8, 16.0f, 1e-3f, 1337u);
        int reload_ok = orion_qwen9b_lora_trainer_load(&reloaded, adapter_out_dir) &&
                        orion_qwen9b_lora_trainer_compare(&trainer, &reloaded, 1e-6f);

        printf("PASS: qwen35 9b lora sequence training\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  sequences_path=%s\n", sequencesPath.UTF8String);
        printf("  adapter_out_dir=%s\n", adapter_out_dir);
        printf("  hidden_cache_dir=%s\n", hiddenCacheDir ? hiddenCacheDir.UTF8String : "");
        printf("  layer_idx=%d\n", trainer.layer_idx);
        printf("  rank=%d\n", trainer.q_proj.rank);
        printf("  alpha=%.1f\n", trainer.q_proj.alpha);
        printf("  grad_accum_sequences=%d\n", grad_accum_sequences);
        printf("  sequence_bucket=%d\n", sequence_bucket);
        printf("  sequences_completed=%d\n", sequences_completed);
        printf("  samples_seen=%lu\n", (unsigned long)sampleIds.count);
        printf("  resume_loaded=%d\n", resume_loaded);
        printf("  resume_start_step=%d\n", resume_start_step);
        printf("  trainer_step_final=%d\n", trainer.step);
        printf("  tokens_seen=%d\n", tokens_seen);
        printf("  pairs_seen=%d\n", pairs_seen);
        printf("  hidden_cache_prefilled=%lu\n", (unsigned long)hiddenCachePrefilled);
        printf("  cached_hidden_tokens=%lu\n", (unsigned long)hiddenCache.count);
        printf("  hidden_cache_hits=%d\n", hidden_cache_hits);
        printf("  hidden_cache_misses=%d\n", hidden_cache_misses);
        printf("  packed_batches_completed=%d\n", packed_batches_completed);
        printf("  packed_sequences_last=%d\n", packed_sequences_last);
        printf("  packed_sequences_max=%d\n", packed_sequences_max);
        printf("  packed_batch_tokens_last=%d\n", packed_batch_tokens_last);
        printf("  packed_batch_tokens_max=%d\n", packed_batch_tokens_max);
        printf("  avg_tokens_per_sequence=%.3f\n", (double)tokens_seen / (double)sequences_completed);
        printf("  loss_first=%.6f\n", loss_first);
        printf("  loss_last=%.6f\n", loss_last);
        printf("  loss_min=%.6f\n", loss_min);
        printf("  loss_max=%.6f\n", loss_max);
        printf("  loss_avg=%.6f\n", (float)(loss_sum / (double)sequences_completed));
        printf("  q_grad_abs_sum_last=%.6f\n", q_grad_last);
        printf("  v_grad_abs_sum_last=%.6f\n", v_grad_last);
        printf("  q_param_abs_sum_last=%.6f\n", q_param_last);
        printf("  v_param_abs_sum_last=%.6f\n", v_param_last);
        printf("  reload_compare_ok=%d\n", reload_ok);
        printf("  next_blocker=%s\n", "validation split and hybrid parity gate promotion");

        orion_qwen9b_lora_trainer_free(&reloaded);
        orion_qwen9b_lora_trainer_free(&trainer);
        orion_qwen35_manifest_free(manifest);
        return reload_ok ? 0 : 1;
    }
}
