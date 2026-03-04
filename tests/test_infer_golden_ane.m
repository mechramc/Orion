// test_infer_golden_ane.m — T103: ANE full forward golden vectors test
//
// Verifies that ANE decode produces the same greedy token sequences
// as CPU decode (golden reference from test_infer_golden.m).
//
// Build:
//   xcrun clang -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK \
//     -framework Foundation -framework IOSurface -framework Accelerate -ldl -I . \
//     core/{ane_runtime,iosurface_tensor,mil_builder,ane_program_cache}.m \
//     model/weight_loader.m \
//     kernels/inference/{decode_cpu,kv_cache,decode_ane,gpt2_decode_ane.milgen}.m \
//     tests/test_infer_golden_ane.m -o tests/test_infer_golden_ane
//
// Run:
//   ./tests/test_infer_golden_ane

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <string.h>
#import "model/weight_loader.h"
#import "model/configs/gpt2_124m.h"
#import "kernels/inference/decode_cpu.h"
#import "kernels/inference/decode_ane.h"
#import "kernels/inference/kv_cache.h"
#import "core/ane_runtime.h"

static int tests_passed = 0;
static int tests_failed = 0;

/// Generate tokens using CPU prefill + ANE decode, compare against expected.
static void test_ane_greedy(OrionGPT2Weights* w,
                             const char* name,
                             const int* prompt_tokens, int prompt_len,
                             const int* expected_tokens, int expected_len) {
    OrionKVCache* kv = orion_kv_cache_create(&kGPT2_124M);
    float* logits = (float*)malloc(w->vocab * sizeof(float));

    // CPU prefill (same as reference)
    orion_gpt2_prefill_kv(w, prompt_tokens, prompt_len, kv, logits);

    int match = 1;
    int generated[64];

    for (int i = 0; i < expected_len; i++) {
        int next = orion_sample_token(logits, w->vocab, 0.0f, 1.0f, NULL);
        generated[i] = next;

        if (next != expected_tokens[i]) {
            printf("FAIL %s: token[%d]=%d, expected %d\n", name, i, next, expected_tokens[i]);
            match = 0;
            break;
        }

        if (i < expected_len - 1) {
            bool ok = orion_ane_decode_step(w, kv, next,
                                             "model/blobs/gpt2_124m", logits);
            if (!ok) {
                printf("FAIL %s: ANE decode step %d failed\n", name, i);
                match = 0;
                break;
            }
        }
    }

    if (match) {
        printf("PASS: %s → %d tokens match\n", name, expected_len);
        tests_passed++;
    } else {
        tests_failed++;
    }

    orion_kv_cache_free(kv);
    free(logits);
}

/// Compare ANE decode against CPU decode for N tokens.
static void test_ane_vs_cpu(OrionGPT2Weights* w,
                              const char* name,
                              const int* prompt_tokens, int prompt_len,
                              int gen_len) {
    float* logits = (float*)malloc(w->vocab * sizeof(float));

    // CPU reference
    OrionKVCache* kv_cpu = orion_kv_cache_create(&kGPT2_124M);
    orion_gpt2_prefill_kv(w, prompt_tokens, prompt_len, kv_cpu, logits);
    int cpu_tokens[64];
    for (int i = 0; i < gen_len; i++) {
        cpu_tokens[i] = orion_sample_token(logits, w->vocab, 0.0f, 1.0f, NULL);
        if (i < gen_len - 1) orion_gpt2_decode_step(w, kv_cpu, cpu_tokens[i], logits);
    }

    // ANE
    OrionKVCache* kv_ane = orion_kv_cache_create(&kGPT2_124M);
    orion_gpt2_prefill_kv(w, prompt_tokens, prompt_len, kv_ane, logits);
    int ane_tokens[64];
    int match_count = 0;
    bool ane_failed = false;
    for (int i = 0; i < gen_len; i++) {
        ane_tokens[i] = orion_sample_token(logits, w->vocab, 0.0f, 1.0f, NULL);
        if (ane_tokens[i] == cpu_tokens[i]) match_count++;
        if (i < gen_len - 1) {
            bool ok = orion_ane_decode_step(w, kv_ane, ane_tokens[i],
                                             "model/blobs/gpt2_124m", logits);
            if (!ok) { ane_failed = true; break; }
        }
    }

    if (ane_failed) {
        printf("FAIL %s: ANE decode step failed\n", name);
        tests_failed++;
    } else if (match_count == gen_len) {
        printf("PASS: %s → %d/%d tokens match (exact)\n", name, match_count, gen_len);
        tests_passed++;
    } else {
        // fp16 drift may cause divergence after several steps
        printf("INFO %s: %d/%d tokens match (CPU: ", name, match_count, gen_len);
        for (int i = 0; i < gen_len; i++) printf("%d%s", cpu_tokens[i], i < gen_len-1 ? "," : "");
        printf(", ANE: ");
        for (int i = 0; i < gen_len; i++) printf("%d%s", ane_tokens[i], i < gen_len-1 ? "," : "");
        printf(")\n");
        // Require at least first 3 tokens match
        bool first3_match = true;
        int check_len = gen_len < 3 ? gen_len : 3;
        for (int i = 0; i < check_len; i++) {
            if (ane_tokens[i] != cpu_tokens[i]) first3_match = false;
        }
        if (first3_match) {
            printf("PASS: %s → first %d tokens match (fp16 drift after)\n", name, check_len);
            tests_passed++;
        } else {
            printf("FAIL: %s → early divergence\n", name);
            tests_failed++;
        }
    }

    orion_kv_cache_free(kv_cpu);
    orion_kv_cache_free(kv_ane);
    free(logits);
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        printf("=== T103: ANE Full Forward Golden Tests ===\n\n");

        if (!orion_ane_init()) {
            printf("FATAL: Failed to initialize ANE runtime\n");
            return 1;
        }

        OrionGPT2Weights* w = orion_gpt2_weights_load("model/blobs/gpt2_124m");
        if (!w) {
            printf("FATAL: Failed to load weights\n");
            return 1;
        }

        // Test 1: "Hello" → 5 greedy tokens (same golden as CPU test)
        {
            int prompt[] = {15496};
            int expected[] = {11, 314, 1101, 7926, 11};  // ", I'm sorry,"
            test_ane_greedy(w, "Hello→5tok (ANE)", prompt, 1, expected, 5);
        }

        // Test 2: "The quick brown fox" → 1 token
        {
            int prompt[] = {464, 2068, 7586, 21831};
            int expected[] = {274};  // "jumps"
            test_ane_greedy(w, "fox→jumps (ANE)", prompt, 4, expected, 1);
        }

        // Test 3: ANE vs CPU for 10 tokens from "Once upon a time"
        {
            int prompt[] = {7454, 2402, 257, 640};
            test_ane_vs_cpu(w, "once_upon→10tok (ANE vs CPU)", prompt, 4, 10);
        }

        // Test 4: ANE vs CPU for 10 tokens from "Hello"
        {
            int prompt[] = {15496};
            test_ane_vs_cpu(w, "Hello→10tok (ANE vs CPU)", prompt, 1, 10);
        }

        printf("\n=== Results: %d passed, %d failed ===\n",
               tests_passed, tests_failed);

        orion_gpt2_weights_free(w);
        return tests_failed > 0 ? 1 : 0;
    }
}
