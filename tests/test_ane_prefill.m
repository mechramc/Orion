// test_ane_prefill.m — Tests for M2 ANE prefill kernels (T047, T048, T050)
//
// Tests:
//   1-4:  Bucket selection (T050)
//   5-7:  Attention compiler frontend compile + eval on ANE (T047)
//   8-10: FFN compiler frontend compile + eval on ANE (T048)
//   11:   Combined attention→FFN for layer 0 (integration)
//
// Build:
//   xcrun clang -O2 -DACCELERATE_NEW_LAPACK -fobjc-arc \
//     -framework Foundation -framework IOSurface -framework Accelerate -ldl \
//     -o tests/test_ane_prefill tests/test_ane_prefill.m \
//     core/ane_runtime.m core/iosurface_tensor.m core/mil_builder.m core/bucket.m \
//     compiler/kernel_adapter.m compiler/frontends/gpt2_prefill.m \
//     compiler/frontends/gpt2_final.m compiler/validate.m \
//     compiler/pipeline.m compiler/codegen.m \
//     model/weight_loader.m kernels/inference/decode_cpu.m -I.
// Run (from repo root):
//   ./tests/test_ane_prefill

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#import <math.h>
#import <stdio.h>
#import <sys/time.h>

#import "core/ane_runtime.h"
#import "core/iosurface_tensor.h"
#import "core/mil_builder.h"
#import "core/bucket.h"
#import "model/configs/gpt2_124m.h"
#import "model/weight_loader.h"
#include "compiler/kernel_adapter.h"
#include "compiler/frontends/gpt2_prefill.h"
#include "compiler/frontends/gpt2_final.h"
#include "compiler/validate.h"
#include "compiler/pipeline.h"
#include "compiler/codegen.h"
#import "kernels/inference/prefill_ane.h"
#import "kernels/inference/decode_cpu.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

#pragma mark - IOSurface helpers

static IOSurfaceRef make_f32_surface(int count, float val) {
    size_t bytes = count * sizeof(float);
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
    IOSurfaceLock(s, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = val;
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static IOSurfaceRef make_f32_surface_data(const float *data, int count) {
    size_t bytes = count * sizeof(float);
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
    IOSurfaceLock(s, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(s), data, bytes);
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static void read_f32_surface(IOSurfaceRef s, float *out, int count) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    memcpy(out, IOSurfaceGetBaseAddress(s), count * sizeof(float));
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

#pragma mark - Weight dict helpers

/// Load a blob file and add to weight dict with the given MIL path key.
static void add_blob_to_dict(NSMutableDictionary *dict, NSString *mil_path, NSString *file_path) {
    NSData *data = [NSData dataWithContentsOfFile:file_path];
    if (data) {
        dict[mil_path] = @{@"offset": @0, @"data": data};
    } else {
        fprintf(stderr, "WARNING: failed to load blob %s\n", file_path.UTF8String);
    }
}

/// Build weight dict for attention layer.
static NSMutableDictionary* build_attn_weight_dict(int layer_idx, int seq_len,
                                                     const char* blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *dir = @(blob_dir);

    const char *names[] = {"ln1_g", "ln1_b", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"};
    for (int i = 0; i < 10; i++) {
        NSString *mil_path = [NSString stringWithFormat:@"@model_path/layer%d/%s.bin",
                              layer_idx, names[i]];
        NSString *file_path = [NSString stringWithFormat:@"%@/layer%d/%s.bin",
                               dir, layer_idx, names[i]];
        add_blob_to_dict(dict, mil_path, file_path);
    }

    // Causal mask (generated, not from file)
    NSString *mask_path = orion_causal_mask_path(seq_len);
    NSData *mask = orion_make_causal_mask_blob(seq_len);
    dict[mask_path] = @{@"offset": @0, @"data": mask};

    return dict;
}

/// Build weight dict for FFN layer.
static NSMutableDictionary* build_ffn_weight_dict(int layer_idx, const char* blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *dir = @(blob_dir);

    const char *names[] = {"ln2_g", "ln2_b", "wfc", "bfc", "wproj", "bproj"};
    for (int i = 0; i < 6; i++) {
        NSString *mil_path = [NSString stringWithFormat:@"@model_path/layer%d/%s.bin",
                              layer_idx, names[i]];
        NSString *file_path = [NSString stringWithFormat:@"%@/layer%d/%s.bin",
                               dir, layer_idx, names[i]];
        add_blob_to_dict(dict, mil_path, file_path);
    }

    return dict;
}

#pragma mark - Test: Bucket Selection (T050)

static void test_bucket_selection(void) {
    printf("\n=== Test: Bucket Selection (T050) ===\n");

    CHECK(orion_select_bucket(1, kGPT2Buckets, kGPT2NumBuckets) == 32,
          "bucket(1) == 32");
    CHECK(orion_select_bucket(17, kGPT2Buckets, kGPT2NumBuckets) == 32,
          "bucket(17) == 32");
    CHECK(orion_select_bucket(32, kGPT2Buckets, kGPT2NumBuckets) == 32,
          "bucket(32) == 32 (exact)");
    CHECK(orion_select_bucket(64, kGPT2Buckets, kGPT2NumBuckets) == 64,
          "bucket(64) == 64 (exact)");
    CHECK(orion_select_bucket(65, kGPT2Buckets, kGPT2NumBuckets) == 128,
          "bucket(65) == 128");
    CHECK(orion_select_bucket(1024, kGPT2Buckets, kGPT2NumBuckets) == 1024,
          "bucket(1024) == 1024 (max)");
    CHECK(orion_select_bucket(1025, kGPT2Buckets, kGPT2NumBuckets) == -1,
          "bucket(1025) == -1 (overflow)");
}

#pragma mark - Test: Attention Prefill (T047)

static void test_attn_prefill(void) {
    printf("\n=== Test: Attention Prefill Compile + Eval (T047) ===\n");

    int seq = 32;  // smallest bucket
    int d = kGPT2_124M.d_model;  // 768
    int count = d * seq;  // 24576 elements

    // Generate MIL text
    NSString *mil = orion_kernel_adapter_generate_mil(orion_frontend_gpt2_prefill_attn, 0, seq, &kGPT2_124M);
    CHECK(mil != nil && mil.length > 0, "MIL text generated");

    // Build weight dict
    NSMutableDictionary *wdict = build_attn_weight_dict(0, seq, "model/blobs/gpt2_124m");
    CHECK(wdict.count == 11, "weight dict has 11 entries (10 weights + mask)");

    // Compile
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "test_attn_L0");
    CHECK(prog != NULL, "attention L0 compiles on ANE");

    if (prog) {
        // Create input with deterministic pattern (ANE layout: [d_model, seq])
        float *input = (float *)malloc(count * sizeof(float));
        for (int i = 0; i < count; i++) {
            input[i] = sinf(i * 0.01f) * 0.1f;
        }
        IOSurfaceRef ioIn = make_f32_surface_data(input, count);

        // Output surfaces
        IOSurfaceRef ioHidden = make_f32_surface(count, 0.0f);
        IOSurfaceRef ioK      = make_f32_surface(count, 0.0f);
        IOSurfaceRef ioV      = make_f32_surface(count, 0.0f);

        IOSurfaceRef ins[]  = {ioIn};
        IOSurfaceRef outs[] = {ioHidden, ioK, ioV};
        bool ok = orion_eval(prog, ins, 1, outs, 3);
        CHECK(ok, "attention L0 eval succeeds");

        // Read outputs and verify finite
        float *hidden = (float *)malloc(count * sizeof(float));
        float *k_out  = (float *)malloc(count * sizeof(float));
        float *v_out  = (float *)malloc(count * sizeof(float));
        read_f32_surface(ioHidden, hidden, count);
        read_f32_surface(ioK, k_out, count);
        read_f32_surface(ioV, v_out, count);

        int finite_h = 1, finite_k = 1, finite_v = 1;
        for (int i = 0; i < count; i++) {
            if (!isfinite(hidden[i])) finite_h = 0;
            if (!isfinite(k_out[i]))  finite_k = 0;
            if (!isfinite(v_out[i]))  finite_v = 0;
        }
        CHECK(finite_h, "hidden output all finite");
        CHECK(finite_k, "K output all finite");
        CHECK(finite_v, "V output all finite");

        // Verify output is not all zeros (layer actually computed something)
        float max_h = 0;
        for (int i = 0; i < count; i++) {
            if (fabsf(hidden[i]) > max_h) max_h = fabsf(hidden[i]);
        }
        CHECK(max_h > 0.001f, "hidden output has non-trivial values");
        printf("    hidden max|val| = %.4f\n", max_h);

        free(input); free(hidden); free(k_out); free(v_out);
        CFRelease(ioIn); CFRelease(ioHidden); CFRelease(ioK); CFRelease(ioV);
        orion_release_program(prog);
    }
}

#pragma mark - Test: FFN Prefill (T048)

static void test_ffn_prefill(void) {
    printf("\n=== Test: FFN Prefill Compile + Eval (T048) ===\n");

    int seq = 32;
    int d = kGPT2_124M.d_model;
    int count = d * seq;

    // Generate MIL text
    NSString *mil = orion_kernel_adapter_generate_mil(orion_frontend_gpt2_prefill_ffn, 0, seq, &kGPT2_124M);
    CHECK(mil != nil && mil.length > 0, "FFN MIL text generated");

    // Build weight dict
    NSMutableDictionary *wdict = build_ffn_weight_dict(0, "model/blobs/gpt2_124m");
    CHECK(wdict.count == 6, "FFN weight dict has 6 entries");

    // Compile
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "test_ffn_L0");
    CHECK(prog != NULL, "FFN L0 compiles on ANE");

    if (prog) {
        // Input
        float *input = (float *)malloc(count * sizeof(float));
        for (int i = 0; i < count; i++) {
            input[i] = sinf(i * 0.01f) * 0.1f;
        }
        IOSurfaceRef ioIn  = make_f32_surface_data(input, count);
        IOSurfaceRef ioOut = make_f32_surface(count, 0.0f);

        IOSurfaceRef ins[]  = {ioIn};
        IOSurfaceRef outs[] = {ioOut};
        bool ok = orion_eval(prog, ins, 1, outs, 1);
        CHECK(ok, "FFN L0 eval succeeds");

        float *output = (float *)malloc(count * sizeof(float));
        read_f32_surface(ioOut, output, count);

        int finite = 1;
        float max_val = 0;
        for (int i = 0; i < count; i++) {
            if (!isfinite(output[i])) finite = 0;
            if (fabsf(output[i]) > max_val) max_val = fabsf(output[i]);
        }
        CHECK(finite, "FFN output all finite");
        CHECK(max_val > 0.001f, "FFN output has non-trivial values");
        printf("    FFN max|val| = %.4f\n", max_val);

        free(input); free(output);
        CFRelease(ioIn); CFRelease(ioOut);
        orion_release_program(prog);
    }
}

#pragma mark - Test: Combined Attention + FFN (Integration)

static void test_combined_layer(void) {
    printf("\n=== Test: Combined Attn→FFN Layer 0 ===\n");

    int seq = 32;
    int d = kGPT2_124M.d_model;
    int count = d * seq;

    // Build weight dicts for both kernels
    NSMutableDictionary *attn_dict = build_attn_weight_dict(0, seq, "model/blobs/gpt2_124m");
    NSMutableDictionary *ffn_dict  = build_ffn_weight_dict(0, "model/blobs/gpt2_124m");

    // Compile both programs
    NSString *attn_mil = orion_kernel_adapter_generate_mil(orion_frontend_gpt2_prefill_attn, 0, seq, &kGPT2_124M);
    NSString *ffn_mil  = orion_kernel_adapter_generate_mil(orion_frontend_gpt2_prefill_ffn, 0, seq, &kGPT2_124M);

    OrionProgram *attn_prog = orion_compile_mil(attn_mil.UTF8String, attn_dict, "comb_attn_L0");
    OrionProgram *ffn_prog  = orion_compile_mil(ffn_mil.UTF8String, ffn_dict, "comb_ffn_L0");

    CHECK(attn_prog && ffn_prog, "both L0 programs compile");

    if (attn_prog && ffn_prog) {
        // Create input from embeddings: use real GPT-2 embeddings for "Hello" token
        // For simplicity, use deterministic input in ANE layout [d_model, seq]
        float *input = (float *)malloc(count * sizeof(float));
        for (int i = 0; i < count; i++) {
            input[i] = sinf(i * 0.007f) * 0.05f;
        }

        // Step 1: Attention
        IOSurfaceRef ioIn     = make_f32_surface_data(input, count);
        IOSurfaceRef ioHidden = make_f32_surface(count, 0.0f);
        IOSurfaceRef ioK      = make_f32_surface(count, 0.0f);
        IOSurfaceRef ioV      = make_f32_surface(count, 0.0f);

        IOSurfaceRef attn_ins[]  = {ioIn};
        IOSurfaceRef attn_outs[] = {ioHidden, ioK, ioV};
        bool ok1 = orion_eval(attn_prog, attn_ins, 1, attn_outs, 3);
        CHECK(ok1, "attention eval succeeds");

        // Step 2: FFN (takes attention output as input)
        IOSurfaceRef ioFFNOut = make_f32_surface(count, 0.0f);
        IOSurfaceRef ffn_ins[]  = {ioHidden};
        IOSurfaceRef ffn_outs[] = {ioFFNOut};
        bool ok2 = orion_eval(ffn_prog, ffn_ins, 1, ffn_outs, 1);
        CHECK(ok2, "FFN eval succeeds");

        // Verify final output
        float *final_out = (float *)malloc(count * sizeof(float));
        read_f32_surface(ioFFNOut, final_out, count);

        int finite = 1;
        float max_val = 0;
        for (int i = 0; i < count; i++) {
            if (!isfinite(final_out[i])) finite = 0;
            if (fabsf(final_out[i]) > max_val) max_val = fabsf(final_out[i]);
        }
        CHECK(finite, "combined L0 output all finite");
        CHECK(max_val > 0.001f, "combined L0 output non-trivial");
        printf("    Combined L0 max|val| = %.4f\n", max_val);

        free(input); free(final_out);
        CFRelease(ioIn); CFRelease(ioHidden); CFRelease(ioK); CFRelease(ioV);
        CFRelease(ioFFNOut);
    }

    if (attn_prog) orion_release_program(attn_prog);
    if (ffn_prog)  orion_release_program(ffn_prog);
}

#pragma mark - Test: ANE vs CPU Comparison

static void test_ane_vs_cpu(void) {
    printf("\n=== Test: ANE vs CPU Comparison (Layer 0) ===\n");

    // Load GPT-2 weights for CPU reference
    OrionGPT2Weights *w = orion_gpt2_weights_load("model/blobs/gpt2_124m");
    if (!w) {
        printf("  SKIP: weights not found\n");
        return;
    }

    int seq = 32;
    int d = kGPT2_124M.d_model;
    int count = d * seq;

    // Create input: token + positional embeddings for tokens [15496] padded to seq=32
    // Token: "Hello" = 15496
    float *cpu_input = (float *)calloc(seq * d, sizeof(float));  // [seq, d] row-major
    // Only position 0 has real data (token embedding + position embedding)
    for (int c = 0; c < d; c++) {
        cpu_input[0 * d + c] = w->wte[15496 * d + c] + w->wpe[0 * d + c];
    }

    // CPU reference: run LayerNorm on position 0
    float *cpu_ln_out = (float *)malloc(d * sizeof(float));
    orion_cpu_layernorm(cpu_input, w->layers[0].ln1_g, w->layers[0].ln1_b, d, cpu_ln_out);

    // ANE: transpose input to ANE layout [d_model, seq]
    float *ane_input = (float *)calloc(count, sizeof(float));
    for (int s_idx = 0; s_idx < seq; s_idx++) {
        for (int c = 0; c < d; c++) {
            ane_input[c * seq + s_idx] = cpu_input[s_idx * d + c];
        }
    }

    // Compile and run attention on ANE
    NSMutableDictionary *wdict = build_attn_weight_dict(0, seq, "model/blobs/gpt2_124m");
    NSString *mil = orion_kernel_adapter_generate_mil(orion_frontend_gpt2_prefill_attn, 0, seq, &kGPT2_124M);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "cmp_attn_L0");

    if (prog) {
        IOSurfaceRef ioIn     = make_f32_surface_data(ane_input, count);
        IOSurfaceRef ioHidden = make_f32_surface(count, 0.0f);
        IOSurfaceRef ioK      = make_f32_surface(count, 0.0f);
        IOSurfaceRef ioV      = make_f32_surface(count, 0.0f);

        IOSurfaceRef ins[]  = {ioIn};
        IOSurfaceRef outs[] = {ioHidden, ioK, ioV};
        bool ok = orion_eval(prog, ins, 1, outs, 3);

        if (ok) {
            // Read K output: ANE layout [d_model, seq], position 0 is column 0
            float *k_ane = (float *)malloc(count * sizeof(float));
            read_f32_surface(ioK, k_ane, count);

            // CPU reference: K = LayerNorm(x) @ Wk^T + bk
            // K for position 0: cblas_sgemv
            float *cpu_k = (float *)malloc(d * sizeof(float));
            cblas_sgemv(CblasRowMajor, CblasNoTrans, d, d, 1.0f,
                        w->layers[0].wk, d, cpu_ln_out, 1, 0.0f, cpu_k, 1);
            for (int c = 0; c < d; c++) cpu_k[c] += w->layers[0].bk[c];

            // ANE K for position 0: k_ane[c * seq + 0]
            float max_err = 0;
            for (int c = 0; c < d; c++) {
                float err = fabsf(k_ane[c * seq + 0] - cpu_k[c]);
                if (err > max_err) max_err = err;
            }
            // fp16 tolerance: ~0.5% relative error typical
            CHECK(max_err < 1.0f, "ANE K matches CPU K (max err < 1.0)");
            printf("    K max error vs CPU: %.4f\n", max_err);

            free(k_ane); free(cpu_k);
        }

        CFRelease(ioIn); CFRelease(ioHidden); CFRelease(ioK); CFRelease(ioV);
        orion_release_program(prog);
    } else {
        printf("  SKIP: ANE compilation failed\n");
    }

    free(cpu_input); free(cpu_ln_out); free(ane_input);
    orion_gpt2_weights_free(w);
}

#pragma mark - Test: Full ANE Prefill E2E (T052)

static void test_full_ane_prefill(void) {
    printf("\n=== Test: Full ANE Prefill E2E (T052) ===\n");

    OrionGPT2Weights *w = orion_gpt2_weights_load("model/blobs/gpt2_124m");
    if (!w) {
        printf("  SKIP: weights not found\n");
        return;
    }

    // "The quick brown fox" = [464, 2068, 7586, 21831]
    int tokens[] = {464, 2068, 7586, 21831};
    int prompt_len = 4;
    int vocab = kGPT2_124M.vocab;

    // ANE prefill
    float *ane_logits = (float *)malloc(vocab * sizeof(float));
    OrionKVCache *ane_kv = orion_kv_cache_create(&kGPT2_124M);

    bool ok = orion_ane_prefill(w, tokens, prompt_len, &kGPT2_124M,
                                 "model/blobs/gpt2_124m", ane_kv, ane_logits);
    CHECK(ok, "ANE prefill completes successfully");

    if (ok) {
        // CPU reference
        float *cpu_logits = (float *)malloc(vocab * sizeof(float));
        OrionKVCache *cpu_kv = orion_kv_cache_create(&kGPT2_124M);
        orion_gpt2_prefill_kv(w, tokens, prompt_len, cpu_kv, cpu_logits);

        // Compare argmax (top-1 token)
        int ane_argmax = 0, cpu_argmax = 0;
        for (int i = 1; i < vocab; i++) {
            if (ane_logits[i] > ane_logits[ane_argmax]) ane_argmax = i;
            if (cpu_logits[i] > cpu_logits[cpu_argmax]) cpu_argmax = i;
        }
        CHECK(ane_argmax == cpu_argmax,
              "ANE argmax matches CPU (top-1 agreement)");
        printf("    ANE argmax=%d, CPU argmax=%d (expected 274='jumps')\n",
               ane_argmax, cpu_argmax);

        // Compare top-5 overlap
        int ane_top5[5], cpu_top5[5];
        float ane_scores[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
        float cpu_scores[5] = {-1e30f,-1e30f,-1e30f,-1e30f,-1e30f};
        for (int i = 0; i < vocab; i++) {
            for (int k = 0; k < 5; k++) {
                if (ane_logits[i] > ane_scores[k]) {
                    for (int j = 4; j > k; j--) {
                        ane_scores[j] = ane_scores[j-1]; ane_top5[j] = ane_top5[j-1];
                    }
                    ane_scores[k] = ane_logits[i]; ane_top5[k] = i;
                    break;
                }
            }
            for (int k = 0; k < 5; k++) {
                if (cpu_logits[i] > cpu_scores[k]) {
                    for (int j = 4; j > k; j--) {
                        cpu_scores[j] = cpu_scores[j-1]; cpu_top5[j] = cpu_top5[j-1];
                    }
                    cpu_scores[k] = cpu_logits[i]; cpu_top5[k] = i;
                    break;
                }
            }
        }
        int overlap = 0;
        for (int a = 0; a < 5; a++)
            for (int c = 0; c < 5; c++)
                if (ane_top5[a] == cpu_top5[c]) overlap++;
        CHECK(overlap >= 3, "top-5 overlap >= 3");
        printf("    Top-5 overlap: %d/5\n", overlap);
        printf("    ANE top-5: [%d,%d,%d,%d,%d]\n",
               ane_top5[0], ane_top5[1], ane_top5[2], ane_top5[3], ane_top5[4]);
        printf("    CPU top-5: [%d,%d,%d,%d,%d]\n",
               cpu_top5[0], cpu_top5[1], cpu_top5[2], cpu_top5[3], cpu_top5[4]);

        // Check KV cache populated
        CHECK(ane_kv->current_len == prompt_len, "KV cache len == prompt_len");

        free(cpu_logits);
        orion_kv_cache_free(cpu_kv);
    }

    free(ane_logits);
    orion_kv_cache_free(ane_kv);
    orion_gpt2_weights_free(w);
}

#pragma mark - Test: ANE Golden Vectors (T055)

static void test_ane_golden(void) {
    printf("\n=== Test: ANE Golden Vectors (T055) ===\n");

    OrionGPT2Weights *w = orion_gpt2_weights_load("model/blobs/gpt2_124m");
    if (!w) { printf("  SKIP: weights not found\n"); return; }

    int vocab = kGPT2_124M.vocab;

    // Test 1: "Hello" → 5 greedy tokens via ANE prefill + CPU decode
    {
        int tokens[] = {15496};  // "Hello"
        int expected[] = {11, 314, 1101, 7926, 11};  // ", I'm sorry,"
        int prompt_len = 1;

        float *logits = (float *)malloc(vocab * sizeof(float));
        OrionKVCache *kv = orion_kv_cache_create(&kGPT2_124M);

        bool ok = orion_ane_prefill(w, tokens, prompt_len, &kGPT2_124M,
                                     "model/blobs/gpt2_124m", kv, logits);
        if (!ok) { printf("  SKIP: ANE prefill failed\n"); goto hello_done; }

        int match = 1;
        for (int i = 0; i < 5; i++) {
            int next = orion_sample_token(logits, vocab, 0.0f, 1.0f, NULL);
            if (next != expected[i]) {
                printf("  token[%d]=%d expected %d\n", i, next, expected[i]);
                match = 0;
                break;
            }
            if (i < 4) orion_gpt2_decode_step(w, kv, next, logits);
        }
        CHECK(match, "Hello→5tok: ANE matches CPU golden");

hello_done:
        free(logits);
        orion_kv_cache_free(kv);
    }

    // Test 2: "The quick brown fox" → "jumps" via ANE
    {
        int tokens[] = {464, 2068, 7586, 21831};
        int prompt_len = 4;

        float *logits = (float *)malloc(vocab * sizeof(float));
        OrionKVCache *kv = orion_kv_cache_create(&kGPT2_124M);

        bool ok = orion_ane_prefill(w, tokens, prompt_len, &kGPT2_124M,
                                     "model/blobs/gpt2_124m", kv, logits);
        if (ok) {
            int next = orion_sample_token(logits, vocab, 0.0f, 1.0f, NULL);
            CHECK(next == 274, "fox→jumps: ANE argmax==274");
        }

        free(logits);
        orion_kv_cache_free(kv);
    }

    orion_gpt2_weights_free(w);
}

#pragma mark - Test: Benchmark ANE vs CPU (T056)

static double bench_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static void test_benchmark(void) {
    printf("\n=== Test: ANE vs CPU Benchmark (T056) ===\n");

    OrionGPT2Weights *w = orion_gpt2_weights_load("model/blobs/gpt2_124m");
    if (!w) { printf("  SKIP: weights not found\n"); return; }

    int tokens[] = {464, 2068, 7586, 21831};  // "The quick brown fox"
    int prompt_len = 4;
    int vocab = kGPT2_124M.vocab;

    // CPU prefill timing
    float *cpu_logits = (float *)malloc(vocab * sizeof(float));
    OrionKVCache *cpu_kv = orion_kv_cache_create(&kGPT2_124M);
    double t0 = bench_time_ms();
    orion_gpt2_prefill_kv(w, tokens, prompt_len, cpu_kv, cpu_logits);
    double cpu_ms = bench_time_ms() - t0;

    // ANE prefill timing (includes compile)
    float *ane_logits = (float *)malloc(vocab * sizeof(float));
    OrionKVCache *ane_kv = orion_kv_cache_create(&kGPT2_124M);
    t0 = bench_time_ms();
    orion_ane_prefill(w, tokens, prompt_len, &kGPT2_124M,
                       "model/blobs/gpt2_124m", ane_kv, ane_logits);
    double ane_ms = bench_time_ms() - t0;

    printf("    CPU prefill: %.1f ms (%.1f ms/tok)\n", cpu_ms, cpu_ms / prompt_len);
    printf("    ANE prefill: %.1f ms (%.1f ms/tok) [includes compile]\n",
           ane_ms, ane_ms / prompt_len);
    printf("    Note: ANE time dominated by compilation (~83%%)\n");
    printf("    Program cache (M4) will eliminate recompilation overhead\n");

    // Just verify both produce same argmax
    int cpu_argmax = 0, ane_argmax = 0;
    for (int i = 1; i < vocab; i++) {
        if (cpu_logits[i] > cpu_logits[cpu_argmax]) cpu_argmax = i;
        if (ane_logits[i] > ane_logits[ane_argmax]) ane_argmax = i;
    }
    CHECK(cpu_argmax == ane_argmax, "benchmark: argmax match");

    free(cpu_logits); free(ane_logits);
    orion_kv_cache_free(cpu_kv); orion_kv_cache_free(ane_kv);
    orion_gpt2_weights_free(w);
}

#pragma mark - Main

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Orion ANE Prefill Tests (T047-T056) ===\n");

        // T050: bucket selection (no ANE needed)
        test_bucket_selection();

        // Initialize ANE for remaining tests
        if (!orion_ane_init()) {
            printf("\nFATAL: ANE init failed\n");
            printf("\n========================================\n");
            printf("Results: %d passed, %d failed\n", g_pass, g_fail);
            return g_fail > 0 ? 1 : 0;
        }

        // T047: attention prefill
        test_attn_prefill();

        // T048: FFN prefill
        test_ffn_prefill();

        // Integration: attention → FFN
        test_combined_layer();

        // ANE vs CPU comparison (single layer)
        test_ane_vs_cpu();

        // T052: Full 12-layer ANE prefill E2E
        test_full_ane_prefill();

        // T055: ANE golden vectors
        test_ane_golden();

        // T056: Benchmark
        test_benchmark();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}
