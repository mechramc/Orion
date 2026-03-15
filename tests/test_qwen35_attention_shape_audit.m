#import <Foundation/Foundation.h>
#import <stdio.h>
#import "../model/weight_loader.h"

static int product_of_shape(NSArray *shape) {
    int total = 1;
    for (id value in shape) {
        if (![value respondsToSelector:@selector(intValue)]) {
            return -1;
        }
        int dim = [value intValue];
        if (dim <= 0) {
            return -1;
        }
        total *= dim;
    }
    return total;
}

static NSDictionary* entry_for_path(NSArray *entries, NSString *relPath) {
    for (id value in entries) {
        if (![value isKindOfClass:[NSDictionary class]]) continue;
        NSDictionary *entry = (NSDictionary *)value;
        NSString *path = entry[@"path"];
        if ([path isEqualToString:relPath]) {
            return entry;
        }
    }
    return nil;
}

static int validate_entry(NSArray *entries, NSString *blobDir, NSString *relPath) {
    NSDictionary *entry = entry_for_path(entries, relPath);
    if (!entry) {
        fprintf(stderr, "FAIL: missing manifest entry for %s\n", relPath.UTF8String);
        return 0;
    }

    NSArray *shape = entry[@"shape"];
    if (![shape isKindOfClass:[NSArray class]] || [shape count] == 0) {
        fprintf(stderr, "FAIL: missing shape for %s\n", relPath.UTF8String);
        return 0;
    }

    int expected = product_of_shape(shape);
    if (expected <= 0) {
        fprintf(stderr, "FAIL: invalid shape for %s\n", relPath.UTF8String);
        return 0;
    }

    NSString *absPath = [blobDir stringByAppendingPathComponent:relPath];
    int got = orion_blob_element_count(absPath.UTF8String);
    if (got != expected) {
        fprintf(stderr, "FAIL: %s element mismatch expected=%d got=%d\n",
                relPath.UTF8String, expected, got);
        return 0;
    }

    printf("  %s shape=%s elements=%d\n",
           relPath.UTF8String,
           [[shape description] UTF8String],
           got);
    return 1;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <blob_dir>\n", argv[0]);
            return 2;
        }

        NSString *blobDir = [NSString stringWithUTF8String:argv[1]];
        NSString *manifestPath = [blobDir stringByAppendingPathComponent:@"manifest.json"];
        NSData *data = [NSData dataWithContentsOfFile:manifestPath];
        if (!data) {
            fprintf(stderr, "FAIL: cannot read %s\n", manifestPath.UTF8String);
            return 1;
        }

        NSError *error = nil;
        NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (![manifest isKindOfClass:[NSDictionary class]]) {
            fprintf(stderr, "FAIL: invalid manifest json: %s\n", error.localizedDescription.UTF8String);
            return 1;
        }

        NSDictionary *runtime = manifest[@"runtime"];
        NSArray *entries = manifest[@"present_entries"];
        if (![runtime isKindOfClass:[NSDictionary class]] || ![entries isKindOfClass:[NSArray class]]) {
            fprintf(stderr, "FAIL: manifest missing runtime/present_entries\n");
            return 1;
        }

        int nLinear = 0;
        int nFull = 0;
        NSArray *layerTypes = runtime[@"layer_types"];
        if ([layerTypes isKindOfClass:[NSArray class]]) {
            for (id value in layerTypes) {
                if (![value isKindOfClass:[NSString class]]) continue;
                NSString *layerType = (NSString *)value;
                if ([layerType isEqualToString:@"linear_attention"]) nLinear += 1;
                if ([layerType isEqualToString:@"full_attention"]) nFull += 1;
            }
        }

        printf("PASS: qwen35 attention shape audit\n");
        printf("  blob_dir=%s\n", blobDir.UTF8String);
        printf("  n_linear_layers=%d\n", nLinear);
        printf("  n_full_layers=%d\n", nFull);
        printf("  attention_note=%s\n", "Qwen3.5 hybrid attention uses non-standard projected dimensions");

        int ok = 1;
        ok &= validate_entry(entries, blobDir, @"layer3/self_attn_q_proj.bin");
        ok &= validate_entry(entries, blobDir, @"layer3/self_attn_k_proj.bin");
        ok &= validate_entry(entries, blobDir, @"layer3/self_attn_v_proj.bin");
        ok &= validate_entry(entries, blobDir, @"layer3/self_attn_o_proj.bin");
        ok &= validate_entry(entries, blobDir, @"layer0/linear_attn_in_proj_qkv.bin");
        ok &= validate_entry(entries, blobDir, @"layer0/linear_attn_in_proj_z.bin");
        ok &= validate_entry(entries, blobDir, @"layer0/linear_attn_in_proj_a.bin");
        ok &= validate_entry(entries, blobDir, @"layer0/linear_attn_in_proj_b.bin");
        ok &= validate_entry(entries, blobDir, @"layer0/linear_attn_out_proj.bin");
        ok &= validate_entry(entries, blobDir, @"layer0/linear_attn_norm.bin");
        ok &= validate_entry(entries, blobDir, @"layer0/linear_attn_dt_bias.bin");
        ok &= validate_entry(entries, blobDir, @"layer0/linear_attn_a_log.bin");
        ok &= validate_entry(entries, blobDir, @"layer0/linear_attn_conv1d.bin");

        if (!ok) {
            return 1;
        }

        printf("  next_blocker=%s\n", "derive full_attention and linear_attention semantics before CPU kernel implementation");
        return 0;
    }
}
