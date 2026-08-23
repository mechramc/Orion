#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/frontends/qwen35_prefill.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#import "compiler/codegen.h"
#include "core/ane_runtime.h"
#import "../model/weight_loader.h"
#import "../kernels/inference/qwen_cpu_ops.h"
#import "../kernels/training/qwen_lora_train.h"

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

static double mean_abs_diff(const float *a, const float *b, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)a[i] - (double)b[i]);
    return total / (double)n;
}

static double max_abs_diff(const float *a, const float *b, int n) {
    double best = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > best) best = d;
    }
    return best;
}

static double mean_abs_diff_q_half(const float *a, const float *b, int q_dim, int half_idx) {
    const float *row_a = a + half_idx * q_dim;
    const float *row_b = b + half_idx * q_dim;
    return mean_abs_diff(row_a, row_b, q_dim);
}

static double max_abs_diff_q_half(const float *a, const float *b, int q_dim, int half_idx) {
    const float *row_a = a + half_idx * q_dim;
    const float *row_b = b + half_idx * q_dim;
    return max_abs_diff(row_a, row_b, q_dim);
}

static double mean_abs_diff_q_half_swapped(const float *cpu_q, const float *ane_q, int q_dim) {
    double total = 0.0;
    const float *cpu_query = cpu_q;
    const float *cpu_gate = cpu_q + q_dim;
    const float *ane_gate = ane_q;
    const float *ane_query = ane_q + q_dim;
    for (int i = 0; i < q_dim; i++) total += fabs((double)cpu_query[i] - (double)ane_query[i]);
    for (int i = 0; i < q_dim; i++) total += fabs((double)cpu_gate[i] - (double)ane_gate[i]);
    return total / (double)(q_dim * 2);
}

static double max_abs_diff_q_half_swapped(const float *cpu_q, const float *ane_q, int q_dim) {
    double best = 0.0;
    const float *cpu_query = cpu_q;
    const float *cpu_gate = cpu_q + q_dim;
    const float *ane_gate = ane_q;
    const float *ane_query = ane_q + q_dim;
    for (int i = 0; i < q_dim; i++) {
        double d = fabs((double)cpu_query[i] - (double)ane_query[i]);
        if (d > best) best = d;
    }
    for (int i = 0; i < q_dim; i++) {
        double d = fabs((double)cpu_gate[i] - (double)ane_gate[i]);
        if (d > best) best = d;
    }
    return best;
}

static float *load_exact(const char *blob_dir, int layer_idx, const char *suffix, int count) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/layer%d/%s", blob_dir, layer_idx, suffix);
    return orion_read_blob_f32_exact(path, count);
}

static void linear_seq(const float *x_seq,
                       int seq_len,
                       int in_dim,
                       const float *weight,
                       int out_dim,
                       float *out_seq) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x_seq, in_dim, weight, in_dim,
                0.0f, out_seq, out_dim);
}

static IOSurfaceRef make_f32_surface(int count, float fill) {
    size_t bytes = (size_t)count * sizeof(float);
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
    IOSurfaceLock(s, 0, NULL);
    float *ptr = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) ptr[i] = fill;
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static IOSurfaceRef make_cpu_seq_input_surface(const float *x_seq, int seq_len, int bucket, int d_model) {
    IOSurfaceRef s = make_f32_surface(d_model * bucket, 0.0f);
    IOSurfaceLock(s, 0, NULL);
    float *ptr = (float *)IOSurfaceGetBaseAddress(s);
    for (int t = 0; t < seq_len; t++) {
        for (int c = 0; c < d_model; c++) {
            ptr[c * bucket + t] = x_seq[t * d_model + c];
        }
    }
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static IOSurfaceRef make_pattern_input_surface(int d_model, int bucket) {
    IOSurfaceRef s = make_f32_surface(d_model * bucket, 0.0f);
    IOSurfaceLock(s, 0, NULL);
    float *ptr = (float *)IOSurfaceGetBaseAddress(s);
    for (int t = 0; t < bucket; t++) {
        for (int c = 0; c < d_model; c++) {
            ptr[c * bucket + t] = sinf((float)(c * 0.001 + t * 0.01)) * 0.1f;
        }
    }
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static void read_ane_surface_prefix(IOSurfaceRef s, int channels, int seq_len, int bucket, float *out_seq) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    const float *ptr = (const float *)IOSurfaceGetBaseAddress(s);
    for (int t = 0; t < seq_len; t++) {
        for (int c = 0; c < channels; c++) {
            out_seq[t * channels + c] = ptr[c * bucket + t];
        }
    }
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

static void add_blob(NSMutableDictionary *dict, NSString *mil_path, NSString *file_path) {
    NSData *data = [NSData dataWithContentsOfFile:file_path];
    if (data) dict[mil_path] = @{@"offset": @0, @"data": data};
}

static NSDictionary *build_qproj_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/input_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"input_layernorm.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/self_attn_q_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"self_attn_q_proj.bin"]);
    return dict;
}

static NSDictionary *build_qproj_linear_only_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/self_attn_q_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"self_attn_q_proj.bin"]);
    return dict;
}

static NSDictionary *build_kv_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/input_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"input_layernorm.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/self_attn_k_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"self_attn_k_proj.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/self_attn_v_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"self_attn_v_proj.bin"]);
    return dict;
}

static NSDictionary *build_kv_linear_only_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/self_attn_k_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"self_attn_k_proj.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/self_attn_v_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"self_attn_v_proj.bin"]);
    return dict;
}

static NSString *compile_graph(OrionGraph *g) {
    if (!g) return nil;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        NSLog(@"graph validate failed: %s", vr.message);
        orion_graph_free(g);
        return nil;
    }
    orion_pipeline_optimize(g);
    NSString *mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <blob_dir> [input_token] [layer_idx] [bucket]\n", argv[0]);
            return 2;
        }

        const char *blob_dir = argv[1];
        const int input_token = argc >= 3 ? atoi(argv[2]) : 0;
        OrionQwen35Manifest *manifest = orion_qwen35_manifest_load(blob_dir);
        if (!manifest) {
            fprintf(stderr, "FAIL: manifest load failed\n");
            return 1;
        }

        const int layer_idx = (argc >= 4 && argv[3][0] != '\0') ? atoi(argv[3]) : (manifest->n_layer - 1);
        const int seq_len = 1;
        const int bucket = argc >= 5 ? atoi(argv[4]) : 32;
        const int d_model = manifest->d_model;
        const int n_head = manifest->n_head;
        const int n_kv_head = manifest->n_kv_head > 0 ? manifest->n_kv_head : manifest->n_head;
        const int head_dim = manifest->head_dim;
        const int q_dim = n_head * head_dim;
        const int kv_dim = n_kv_head * head_dim;
        NSString *blobDir = [NSString stringWithUTF8String:blob_dir];

        float *hidden = (float *)calloc((size_t)d_model, sizeof(float));
        float *normed = (float *)calloc((size_t)d_model, sizeof(float));
        float *cpu_q = (float *)calloc((size_t)q_dim * 2, sizeof(float));
        float *cpu_k = (float *)calloc((size_t)kv_dim, sizeof(float));
        float *cpu_v = (float *)calloc((size_t)kv_dim, sizeof(float));
        float *ane_q = (float *)calloc((size_t)q_dim * 2, sizeof(float));
        float *ane_k = (float *)calloc((size_t)kv_dim, sizeof(float));
        float *ane_v = (float *)calloc((size_t)kv_dim, sizeof(float));
        float *cpu_attn = (float *)calloc((size_t)d_model, sizeof(float));
        float *ane_attn = (float *)calloc((size_t)d_model, sizeof(float));
        float *input_ln = load_exact(blob_dir, layer_idx, "input_layernorm.bin", d_model);
        float *q_proj = load_exact(blob_dir, layer_idx, "self_attn_q_proj.bin", q_dim * 2 * d_model);
        float *k_proj = load_exact(blob_dir, layer_idx, "self_attn_k_proj.bin", kv_dim * d_model);
        float *v_proj = load_exact(blob_dir, layer_idx, "self_attn_v_proj.bin", kv_dim * d_model);
        float *o_proj = load_exact(blob_dir, layer_idx, "self_attn_o_proj.bin", d_model * q_dim);
        float *q_norm = load_exact(blob_dir, layer_idx, "self_attn_q_norm.bin", head_dim);
        float *k_norm = load_exact(blob_dir, layer_idx, "self_attn_k_norm.bin", head_dim);
        IOSurfaceRef ioIn = NULL;
        IOSurfaceRef ioNormedIn = NULL;
        IOSurfaceRef ioQ = NULL;
        IOSurfaceRef ioK = NULL;
        IOSurfaceRef ioV = NULL;
        OrionProgram *progQ = NULL;
        OrionProgram *progKV = NULL;
        OrionProgram *progQLinear = NULL;
        OrionProgram *progKVLinear = NULL;
        NSString *milQ = nil;
        NSString *milKV = nil;
        NSString *milQLinear = nil;
        NSString *milKVLinear = nil;
        IOSurfaceRef ioPatternIn = NULL;
        const char *graph_mode = "raw_hidden_full";
        int raw_hidden_eval_ok = 0;
        int raw_hidden_q_eval_ok = 0;
        int raw_hidden_kv_eval_ok = 0;
        int cpu_rmsnorm_q_eval_ok = 0;
        int cpu_rmsnorm_kv_eval_ok = 0;
        int pattern_q_eval_ok = 0;
        int pattern_q_linear_eval_ok = 0;
        int exit_code = 1;

        if (!hidden || !normed || !cpu_q || !cpu_k || !cpu_v || !ane_q || !ane_k || !ane_v ||
            !cpu_attn || !ane_attn || !input_ln || !q_proj || !k_proj || !v_proj || !o_proj ||
            !q_norm || !k_norm) {
            fprintf(stderr, "FAIL: alloc or weight load failed\n");
            goto cleanup;
        }

        if (!orion_qwen9b_lora_frozen_prefix_hidden(blob_dir, manifest, input_token, hidden)) {
            fprintf(stderr, "FAIL: frozen prefix hidden probe failed\n");
            goto cleanup;
        }

        orion_qwen_cpu_rmsnorm(hidden, input_ln, d_model, 1e-6f, normed);
        linear_seq(normed, seq_len, d_model, q_proj, q_dim * 2, cpu_q);
        linear_seq(normed, seq_len, d_model, k_proj, kv_dim, cpu_k);
        linear_seq(normed, seq_len, d_model, v_proj, kv_dim, cpu_v);
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q, cpu_k, cpu_v, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim,
            manifest->rope_theta, manifest->partial_rotary_factor,
            cpu_attn
        );

        if (!orion_ane_init()) {
            fprintf(stderr, "FAIL: ane init failed\n");
            goto cleanup;
        }

        const int compile_before = orion_compile_count();
        OrionModelConfig cfg = {
            .n_layer = manifest->n_layer,
            .n_head = manifest->n_head,
            .n_kv_head = n_kv_head,
            .d_model = manifest->d_model,
            .head_dim = manifest->head_dim,
            .hidden_dim = manifest->d_ff,
            .vocab = manifest->vocab,
            .max_seq = manifest->max_seq,
        };
        milQ = compile_graph(orion_frontend_qwen35_prefill_q_proj(layer_idx, bucket, &cfg));
        milKV = compile_graph(orion_frontend_qwen35_prefill_kv_proj(layer_idx, bucket, &cfg));
        milQLinear = compile_graph(orion_frontend_qwen35_prefill_q_proj_linear_only(layer_idx, bucket, &cfg));
        milKVLinear = compile_graph(orion_frontend_qwen35_prefill_kv_proj_linear_only(layer_idx, bucket, &cfg));
        if (!milQ || !milKV || !milQLinear || !milKVLinear) {
            fprintf(stderr, "FAIL: mil graph build failed\n");
            goto cleanup;
        }

        progQ = orion_compile_mil(milQ.UTF8String, build_qproj_wdict(layer_idx, blobDir), "qwen35_9b_lora_probe_q");
        progKV = orion_compile_mil(milKV.UTF8String, build_kv_wdict(layer_idx, blobDir), "qwen35_9b_lora_probe_kv");
        progQLinear = orion_compile_mil(milQLinear.UTF8String, build_qproj_linear_only_wdict(layer_idx, blobDir), "qwen35_9b_lora_probe_q_linear");
        progKVLinear = orion_compile_mil(milKVLinear.UTF8String, build_kv_linear_only_wdict(layer_idx, blobDir), "qwen35_9b_lora_probe_kv_linear");
        if (!progQ || !progKV || !progQLinear || !progKVLinear) {
            fprintf(stderr, "FAIL: ane compile failed\n");
            goto cleanup;
        }
        const int compile_after = orion_compile_count();

        ioIn = make_cpu_seq_input_surface(hidden, seq_len, bucket, d_model);
        ioNormedIn = make_cpu_seq_input_surface(normed, seq_len, bucket, d_model);
        ioPatternIn = make_pattern_input_surface(d_model, bucket);
        ioQ = make_f32_surface(q_dim * 2 * bucket, 0.0f);
        ioK = make_f32_surface(kv_dim * bucket, 0.0f);
        ioV = make_f32_surface(kv_dim * bucket, 0.0f);
        if (!ioIn || !ioNormedIn || !ioPatternIn || !ioQ || !ioK || !ioV) {
            fprintf(stderr, "FAIL: iosurface alloc failed\n");
            goto cleanup;
        }

        IOSurfaceRef insQ[] = { ioIn };
        IOSurfaceRef insQLinear[] = { ioNormedIn };
        IOSurfaceRef insQPattern[] = { ioPatternIn };
        IOSurfaceRef outsQ[] = { ioQ };
        IOSurfaceRef outsKV[] = { ioK, ioV };
        raw_hidden_q_eval_ok = orion_eval(progQ, insQ, 1, outsQ, 1) ? 1 : 0;
        raw_hidden_kv_eval_ok = orion_eval(progKV, insQ, 1, outsKV, 2) ? 1 : 0;
        if (raw_hidden_q_eval_ok && raw_hidden_kv_eval_ok) {
            raw_hidden_eval_ok = 1;
            graph_mode = "raw_hidden_full";
        } else {
            graph_mode = "cpu_rmsnorm_linear_only";
            cpu_rmsnorm_q_eval_ok = orion_eval(progQLinear, insQLinear, 1, outsQ, 1) ? 1 : 0;
            cpu_rmsnorm_kv_eval_ok = orion_eval(progKVLinear, insQLinear, 1, outsKV, 2) ? 1 : 0;
            if (!(cpu_rmsnorm_q_eval_ok && cpu_rmsnorm_kv_eval_ok)) {
                pattern_q_eval_ok = orion_eval(progQ, insQPattern, 1, outsQ, 1) ? 1 : 0;
                pattern_q_linear_eval_ok = orion_eval(progQLinear, insQPattern, 1, outsQ, 1) ? 1 : 0;
                printf("FAIL: qwen35 9b lora ane qv probe\n");
                printf("  blob_dir=%s\n", blob_dir);
                printf("  input_token=%d\n", input_token);
                printf("  layer_idx=%d\n", layer_idx);
                printf("  bucket=%d\n", bucket);
                printf("  graph_mode=%s\n", graph_mode);
                printf("  raw_hidden_eval_ok=%d\n", raw_hidden_eval_ok);
                printf("  raw_hidden_q_eval_ok=%d\n", raw_hidden_q_eval_ok);
                printf("  raw_hidden_kv_eval_ok=%d\n", raw_hidden_kv_eval_ok);
                printf("  cpu_rmsnorm_q_eval_ok=%d\n", cpu_rmsnorm_q_eval_ok);
                printf("  cpu_rmsnorm_kv_eval_ok=%d\n", cpu_rmsnorm_kv_eval_ok);
                printf("  pattern_q_eval_ok=%d\n", pattern_q_eval_ok);
                printf("  pattern_q_linear_eval_ok=%d\n", pattern_q_linear_eval_ok);
                printf("  hidden_abs_sum=%.6f\n", abs_sum(hidden, d_model));
                printf("  hidden_max_abs=%.6f\n", max_abs(hidden, d_model));
                printf("  normed_abs_sum=%.6f\n", abs_sum(normed, d_model));
                printf("  normed_max_abs=%.6f\n", max_abs(normed, d_model));
                printf("  compile_count_before=%d\n", compile_before);
                printf("  compile_count_after=%d\n", compile_after);
                printf("  compile_count_delta=%d\n", compile_after - compile_before);
                printf("  next_blocker=%s\n", "debug layer31 q graph eval path before ANE train splice");
                goto cleanup;
            }
        }

        read_ane_surface_prefix(ioQ, q_dim * 2, seq_len, bucket, ane_q);
        read_ane_surface_prefix(ioK, kv_dim, seq_len, bucket, ane_k);
        read_ane_surface_prefix(ioV, kv_dim, seq_len, bucket, ane_v);
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            ane_q, ane_k, ane_v, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim,
            manifest->rope_theta, manifest->partial_rotary_factor,
            ane_attn
        );

        printf("PASS: qwen35 9b lora ane qv probe\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  input_token=%d\n", input_token);
        printf("  layer_idx=%d\n", layer_idx);
        printf("  bucket=%d\n", bucket);
        printf("  graph_mode=%s\n", graph_mode);
        printf("  raw_hidden_eval_ok=%d\n", raw_hidden_eval_ok);
        printf("  raw_hidden_q_eval_ok=%d\n", raw_hidden_q_eval_ok);
        printf("  raw_hidden_kv_eval_ok=%d\n", raw_hidden_kv_eval_ok);
        printf("  cpu_rmsnorm_q_eval_ok=%d\n", cpu_rmsnorm_q_eval_ok);
        printf("  cpu_rmsnorm_kv_eval_ok=%d\n", cpu_rmsnorm_kv_eval_ok);
        printf("  pattern_q_eval_ok=%d\n", pattern_q_eval_ok);
        printf("  pattern_q_linear_eval_ok=%d\n", pattern_q_linear_eval_ok);
        printf("  hidden_abs_sum=%.6f\n", abs_sum(hidden, d_model));
        printf("  hidden_max_abs=%.6f\n", max_abs(hidden, d_model));
        printf("  normed_abs_sum=%.6f\n", abs_sum(normed, d_model));
        printf("  normed_max_abs=%.6f\n", max_abs(normed, d_model));
        printf("  q_proj_mean_abs_diff=%.6f\n", mean_abs_diff(cpu_q, ane_q, q_dim * 2));
        printf("  q_proj_max_abs_diff=%.6f\n", max_abs_diff(cpu_q, ane_q, q_dim * 2));
        printf("  q_query_mean_abs_diff=%.6f\n", mean_abs_diff_q_half(cpu_q, ane_q, q_dim, 0));
        printf("  q_query_max_abs_diff=%.6f\n", max_abs_diff_q_half(cpu_q, ane_q, q_dim, 0));
        printf("  q_gate_mean_abs_diff=%.6f\n", mean_abs_diff_q_half(cpu_q, ane_q, q_dim, 1));
        printf("  q_gate_max_abs_diff=%.6f\n", max_abs_diff_q_half(cpu_q, ane_q, q_dim, 1));
        printf("  q_half_swap_mean_abs_diff=%.6f\n", mean_abs_diff_q_half_swapped(cpu_q, ane_q, q_dim));
        printf("  q_half_swap_max_abs_diff=%.6f\n", max_abs_diff_q_half_swapped(cpu_q, ane_q, q_dim));
        printf("  k_proj_mean_abs_diff=%.6f\n", mean_abs_diff(cpu_k, ane_k, kv_dim));
        printf("  k_proj_max_abs_diff=%.6f\n", max_abs_diff(cpu_k, ane_k, kv_dim));
        printf("  v_proj_mean_abs_diff=%.6f\n", mean_abs_diff(cpu_v, ane_v, kv_dim));
        printf("  v_proj_max_abs_diff=%.6f\n", max_abs_diff(cpu_v, ane_v, kv_dim));
        printf("  attn_out_mean_abs_diff=%.6f\n", mean_abs_diff(cpu_attn, ane_attn, d_model));
        printf("  attn_out_max_abs_diff=%.6f\n", max_abs_diff(cpu_attn, ane_attn, d_model));
        printf("  compile_count_before=%d\n", compile_before);
        printf("  compile_count_after=%d\n", compile_after);
        printf("  compile_count_delta=%d\n", compile_after - compile_before);
        printf("  next_blocker=%s\n",
               raw_hidden_eval_ok
               ? "wire CPU optimizer on top of ANE q/v forward"
               : "wire CPU RMSNorm plus ANE q/v forward and debug raw hidden path");
        exit_code = 0;

cleanup:
        if (ioIn) CFRelease(ioIn);
        if (ioNormedIn) CFRelease(ioNormedIn);
        if (ioPatternIn) CFRelease(ioPatternIn);
        if (ioQ) CFRelease(ioQ);
        if (ioK) CFRelease(ioK);
        if (ioV) CFRelease(ioV);
        if (progQ) orion_release_program(progQ);
        if (progKV) orion_release_program(progKV);
        if (progQLinear) orion_release_program(progQLinear);
        if (progKVLinear) orion_release_program(progKVLinear);
        free(hidden);
        free(normed);
        free(cpu_q);
        free(cpu_k);
        free(cpu_v);
        free(ane_q);
        free(ane_k);
        free(ane_v);
        free(cpu_attn);
        free(ane_attn);
        free(input_ln);
        free(q_proj);
        free(k_proj);
        free(v_proj);
        free(o_proj);
        free(q_norm);
        free(k_norm);
        orion_qwen35_manifest_free(manifest);
        return exit_code;
    }
}
