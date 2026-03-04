// test_decode_ane_step.m — T101: Test ANE decode step against CPU decode
//
// Verifies that orion_ane_decode_step produces logits matching
// orion_gpt2_decode_step (CPU reference) within fp16 tolerance.
//
// Requires GPT-2 124M weights at model/blobs/gpt2_124m/
//
// Build:
//   xcrun clang -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK \
//     -framework Foundation -framework IOSurface -framework Accelerate -ldl -I . \
//     core/{ane_runtime,iosurface_tensor,mil_builder,ane_program_cache}.m \
//     model/weight_loader.m \
//     kernels/inference/{decode_cpu,kv_cache,decode_ane,gpt2_decode_ane.milgen}.m \
//     tests/test_decode_ane_step.m -o tests/test_decode_ane_step
//
// Run:
//   ./tests/test_decode_ane_step

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <math.h>
#import <sys/time.h>
#import <Accelerate/Accelerate.h>
#import "model/weight_loader.h"
#import "model/configs/gpt2_124m.h"
#import "kernels/inference/decode_cpu.h"
#import "kernels/inference/decode_ane.h"
#import "kernels/inference/kv_cache.h"
#import "core/ane_runtime.h"
#import "core/ane_program_cache.h"

static int g_pass = 0, g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        g_fail++; return; \
    } \
} while(0)

#define PASS(msg) do { \
    printf("  PASS: %s\n", msg); \
    g_pass++; \
} while(0)

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

#pragma mark - Test 1: Single decode step matches CPU

static void test_single_step_matches_cpu(OrionGPT2Weights* w) {
    printf("Test: Single ANE decode step matches CPU...\n");

    int vocab = w->vocab;

    // CPU: prefill "Hello" (token 15496), then decode one step
    OrionKVCache* kv_cpu = orion_kv_cache_create(&kGPT2_124M);
    float* logits_prefill = (float*)malloc(vocab * sizeof(float));
    int prompt[] = {15496};
    orion_gpt2_prefill_kv(w, prompt, 1, kv_cpu, logits_prefill);

    // Get first generated token (greedy)
    int first_token = orion_sample_token(logits_prefill, vocab, 0.0f, 1.0f, NULL);
    printf("  First token after prefill: %d\n", first_token);

    // CPU decode step
    float* logits_cpu = (float*)malloc(vocab * sizeof(float));
    orion_gpt2_decode_step(w, kv_cpu, first_token, logits_cpu);
    int cpu_next = orion_sample_token(logits_cpu, vocab, 0.0f, 1.0f, NULL);
    printf("  CPU decode next token: %d\n", cpu_next);

    // ANE: prefill same prompt (CPU), then ANE decode step
    OrionKVCache* kv_ane = orion_kv_cache_create(&kGPT2_124M);
    float* logits_prefill2 = (float*)malloc(vocab * sizeof(float));
    orion_gpt2_prefill_kv(w, prompt, 1, kv_ane, logits_prefill2);

    float* logits_ane = (float*)malloc(vocab * sizeof(float));
    bool ok = orion_ane_decode_step(w, kv_ane, first_token,
                                     "model/blobs/gpt2_124m", logits_ane);
    ASSERT(ok, "ANE decode step should succeed");

    int ane_next = orion_sample_token(logits_ane, vocab, 0.0f, 1.0f, NULL);
    printf("  ANE decode next token: %d\n", ane_next);

    // Check greedy token matches
    bool greedy_match = (cpu_next == ane_next);
    printf("  Greedy match: %s (CPU=%d, ANE=%d)\n",
           greedy_match ? "YES" : "NO", cpu_next, ane_next);
    ASSERT(greedy_match, "ANE and CPU should pick the same greedy token");

    // Check logits numerical closeness
    // ANE uses fp16 internally so we expect some divergence
    float max_diff = 0.0f;
    double sum_diff = 0.0;
    for (int i = 0; i < vocab; i++) {
        float diff = fabsf(logits_cpu[i] - logits_ane[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }
    float avg_diff = (float)(sum_diff / vocab);
    printf("  Logits diff: max=%.4f, avg=%.6f\n", max_diff, avg_diff);

    // fp16 accumulation through 12 layers — allow generous tolerance
    ASSERT(max_diff < 5.0f, "Max logit diff should be < 5.0 (fp16 tolerance)");

    // Check top-5 overlap
    int top5_cpu[5], top5_ane[5];
    for (int k = 0; k < 5; k++) {
        float best_cpu = -INFINITY, best_ane = -INFINITY;
        top5_cpu[k] = top5_ane[k] = 0;
        for (int i = 0; i < vocab; i++) {
            bool skip_cpu = false, skip_ane = false;
            for (int j = 0; j < k; j++) {
                if (i == top5_cpu[j]) skip_cpu = true;
                if (i == top5_ane[j]) skip_ane = true;
            }
            if (!skip_cpu && logits_cpu[i] > best_cpu) {
                best_cpu = logits_cpu[i]; top5_cpu[k] = i;
            }
            if (!skip_ane && logits_ane[i] > best_ane) {
                best_ane = logits_ane[i]; top5_ane[k] = i;
            }
        }
    }
    int overlap = 0;
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            if (top5_cpu[i] == top5_ane[j]) { overlap++; break; }
        }
    }
    printf("  Top-5 overlap: %d/5\n", overlap);
    ASSERT(overlap >= 3, "Top-5 overlap should be >= 3");

    PASS("Single ANE decode step matches CPU");

    orion_kv_cache_free(kv_cpu);
    orion_kv_cache_free(kv_ane);
    free(logits_prefill); free(logits_prefill2);
    free(logits_cpu); free(logits_ane);
}

#pragma mark - Test 2: Multi-step greedy generation

static void test_multi_step_greedy(OrionGPT2Weights* w) {
    printf("Test: Multi-step ANE decode (3 tokens)...\n");

    int vocab = w->vocab;
    int prompt[] = {15496};  // "Hello"

    // CPU reference: prefill + 3 decode steps
    OrionKVCache* kv_cpu = orion_kv_cache_create(&kGPT2_124M);
    float* logits = (float*)malloc(vocab * sizeof(float));
    orion_gpt2_prefill_kv(w, prompt, 1, kv_cpu, logits);

    int cpu_tokens[3];
    for (int i = 0; i < 3; i++) {
        cpu_tokens[i] = orion_sample_token(logits, vocab, 0.0f, 1.0f, NULL);
        if (i < 2) orion_gpt2_decode_step(w, kv_cpu, cpu_tokens[i], logits);
    }
    printf("  CPU tokens: %d %d %d\n", cpu_tokens[0], cpu_tokens[1], cpu_tokens[2]);

    // ANE: prefill (CPU) + 3 ANE decode steps
    OrionKVCache* kv_ane = orion_kv_cache_create(&kGPT2_124M);
    orion_gpt2_prefill_kv(w, prompt, 1, kv_ane, logits);

    int ane_tokens[3];
    for (int i = 0; i < 3; i++) {
        ane_tokens[i] = orion_sample_token(logits, vocab, 0.0f, 1.0f, NULL);
        if (i < 2) {
            bool ok = orion_ane_decode_step(w, kv_ane, ane_tokens[i],
                                             "model/blobs/gpt2_124m", logits);
            ASSERT(ok, "ANE decode step should succeed");
        }
    }
    printf("  ANE tokens: %d %d %d\n", ane_tokens[0], ane_tokens[1], ane_tokens[2]);

    // First token comes from same prefill logits, must match
    ASSERT(ane_tokens[0] == cpu_tokens[0], "First token must match (from prefill)");

    // Subsequent tokens may diverge due to fp16 but check at least first decode matches
    bool decode1_match = (ane_tokens[1] == cpu_tokens[1]);
    printf("  Decode step 1 match: %s\n", decode1_match ? "YES" : "NO");
    ASSERT(decode1_match, "First ANE decode token should match CPU");

    PASS("Multi-step ANE decode (3 tokens)");

    orion_kv_cache_free(kv_cpu);
    orion_kv_cache_free(kv_ane);
    free(logits);
}

#pragma mark - Test 3: Performance benchmark

static void test_performance(OrionGPT2Weights* w) {
    printf("Test: ANE decode performance...\n");

    int vocab = w->vocab;
    int prompt[] = {15496};  // "Hello"

    OrionKVCache* kv = orion_kv_cache_create(&kGPT2_124M);
    float* logits = (float*)malloc(vocab * sizeof(float));
    orion_gpt2_prefill_kv(w, prompt, 1, kv, logits);

    int first_token = orion_sample_token(logits, vocab, 0.0f, 1.0f, NULL);

    // First call compiles 24 programs — measure separately
    double t0 = time_ms();
    bool ok = orion_ane_decode_step(w, kv, first_token,
                                     "model/blobs/gpt2_124m", logits);
    double t_first = time_ms() - t0;
    ASSERT(ok, "First ANE decode step should succeed");
    printf("  First call (with compile): %.1fms\n", t_first);

    int next = orion_sample_token(logits, vocab, 0.0f, 1.0f, NULL);

    // Subsequent calls hit cache — benchmark 5 iterations
    double times[5];
    for (int i = 0; i < 5; i++) {
        t0 = time_ms();
        ok = orion_ane_decode_step(w, kv, next, "model/blobs/gpt2_124m", logits);
        times[i] = time_ms() - t0;
        ASSERT(ok, "Cached ANE decode step should succeed");
        next = orion_sample_token(logits, vocab, 0.0f, 1.0f, NULL);
    }

    double avg = 0;
    for (int i = 0; i < 5; i++) avg += times[i];
    avg /= 5;
    printf("  Cached calls (5 iters): %.1f %.1f %.1f %.1f %.1f ms\n",
           times[0], times[1], times[2], times[3], times[4]);
    printf("  Average cached: %.1fms\n", avg);

    int cache_size = orion_cache_size();
    printf("  Program cache: %d programs\n", cache_size);
    ASSERT(cache_size == 24, "Should have 24 cached programs (12 proj + 12 ffn)");

    PASS("ANE decode performance");

    orion_kv_cache_free(kv);
    free(logits);
}

#pragma mark - Main

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        printf("=== T101: ANE Decode Step Tests ===\n\n");

        if (!orion_ane_init()) {
            printf("FATAL: Failed to initialize ANE runtime\n");
            return 1;
        }
        printf("ANE runtime initialized\n");

        OrionGPT2Weights* w = orion_gpt2_weights_load("model/blobs/gpt2_124m");
        if (!w) {
            printf("FATAL: Failed to load weights from model/blobs/gpt2_124m\n");
            printf("       (Requires GPT-2 124M weight blobs)\n");
            return 1;
        }
        printf("Weights loaded: %d layers, d=%d, vocab=%d\n\n",
               w->n_layer, w->d_model, w->vocab);

        test_single_step_matches_cpu(w);
        test_multi_step_greedy(w);
        test_performance(w);

        printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);

        orion_gpt2_weights_free(w);
        return g_fail > 0 ? 1 : 0;
    }
}
