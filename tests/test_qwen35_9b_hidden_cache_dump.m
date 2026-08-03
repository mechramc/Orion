#import <Foundation/Foundation.h>
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

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 4) {
            fprintf(stderr, "usage: %s <blob_dir> <sequences_jsonl> <out_dir>\n", argv[0]);
            return 2;
        }

        const char *blob_dir = argv[1];
        NSString *sequencesPath = [NSString stringWithUTF8String:argv[2]];
        NSString *outDir = [NSString stringWithUTF8String:argv[3]];

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

        NSMutableOrderedSet<NSNumber *> *orderedTokens = [NSMutableOrderedSet orderedSet];
        for (NSDictionary *sequence in sequences) {
            NSArray *tokenIds = sequence[@"token_ids"];
            if (![tokenIds isKindOfClass:[NSArray class]] || tokenIds.count < 2) continue;
            for (NSUInteger idx = 0; idx + 1 < tokenIds.count; idx++) {
                [orderedTokens addObject:@([tokenIds[idx] intValue])];
            }
        }

        if (orderedTokens.count == 0) {
            fprintf(stderr, "FAIL: no input tokens found in sequences\n");
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        NSError *mkdirError = nil;
        [[NSFileManager defaultManager] createDirectoryAtPath:outDir
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:&mkdirError];
        if (mkdirError) {
            fprintf(stderr, "FAIL: cannot create cache dir %s\n", outDir.UTF8String);
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        NSString *binPath = [outDir stringByAppendingPathComponent:@"hidden_cache.bin"];
        NSString *indexPath = [outDir stringByAppendingPathComponent:@"hidden_cache_index.json"];
        FILE *bin = fopen(binPath.UTF8String, "wb");
        if (!bin) {
            fprintf(stderr, "FAIL: cannot open hidden cache bin for write\n");
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        NSMutableArray<NSNumber *> *tokens = [NSMutableArray arrayWithCapacity:orderedTokens.count];
        NSMutableArray<NSNumber *> *offsets = [NSMutableArray arrayWithCapacity:orderedTokens.count];
        NSMutableArray<NSNumber *> *counts = [NSMutableArray arrayWithCapacity:orderedTokens.count];
        long offset = 0;
        int dumped = 0;
        for (NSNumber *tokenNumber in orderedTokens) {
            int tokenId = tokenNumber.intValue;
            float *hidden = (float *)calloc((size_t)manifest->d_model, sizeof(float));
            if (!hidden) {
                fclose(bin);
                orion_qwen35_manifest_free(manifest);
                return 1;
            }
            if (!orion_qwen9b_lora_frozen_prefix_hidden(blob_dir, manifest, tokenId, hidden)) {
                fprintf(stderr, "FAIL: frozen prefix hidden failed for token %d\n", tokenId);
                free(hidden);
                fclose(bin);
                orion_qwen35_manifest_free(manifest);
                return 1;
            }

            size_t wrote = fwrite(hidden, sizeof(float), (size_t)manifest->d_model, bin);
            free(hidden);
            if (wrote != (size_t)manifest->d_model) {
                fprintf(stderr, "FAIL: hidden cache write short for token %d\n", tokenId);
                fclose(bin);
                orion_qwen35_manifest_free(manifest);
                return 1;
            }

            [tokens addObject:tokenNumber];
            [offsets addObject:@(offset)];
            [counts addObject:@(manifest->d_model)];
            offset += (long)manifest->d_model * (long)sizeof(float);
            dumped += 1;
        }

        fclose(bin);

        NSDictionary *index = @{
            @"status": @"PASS_QWEN35_HIDDEN_CACHE_DUMP",
            @"blob_dir": [NSString stringWithUTF8String:blob_dir],
            @"sequences_path": sequencesPath,
            @"bin_path": binPath,
            @"d_model": @(manifest->d_model),
            @"tokens": tokens,
            @"offsets": offsets,
            @"counts": counts,
            @"tokens_dumped": @(dumped),
        };
        NSData *jsonData = [NSJSONSerialization dataWithJSONObject:index options:NSJSONWritingPrettyPrinted error:nil];
        if (![jsonData writeToFile:indexPath atomically:YES]) {
            fprintf(stderr, "FAIL: cannot write hidden cache index\n");
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        printf("PASS: qwen35 hidden cache dump\n");
        printf("  tokens_dumped=%d\n", dumped);
        printf("  d_model=%d\n", manifest->d_model);
        printf("  out_dir=%s\n", outDir.UTF8String);
        printf("  index_path=%s\n", indexPath.UTF8String);
        printf("  bin_path=%s\n", binPath.UTF8String);

        orion_qwen35_manifest_free(manifest);
        return 0;
    }
}
