#import <Foundation/Foundation.h>
#import <stdio.h>
#import <math.h>
#import <string.h>
#import "../model/weight_loader.h"
#import "../model/configs/gpt2_124m.h"
#import "../kernels/inference/decode_cpu.h"
#import "../kernels/inference/kv_cache.h"

// T037-T040 test: KV cache + decode step + sampling

static int tests_passed = 0;
static int tests_failed = 0;

#define ASSERT(cond, msg, ...) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: " msg "\n", ##__VA_ARGS__); \
        tests_failed++; \
        return; \
    } \
} while(0)

#define PASS(msg, ...) do { \
    printf("PASS: " msg "\n", ##__VA_ARGS__); \
    tests_passed++; \
} while(0)

static OrionGPT2Weights* g_weights = NULL;

static void test_prefill_kv(void) {
    // "Hello" = [15496] → prefill with KV cache should produce same logits as forward
    int tokens[] = {15496};
    float* logits_fwd = (float*)malloc(g_weights->vocab * sizeof(float));
    float* logits_kv = (float*)malloc(g_weights->vocab * sizeof(float));

    orion_gpt2_forward_cpu(g_weights, tokens, 1, logits_fwd);

    OrionKVCache* kv = orion_kv_cache_create(&kGPT2_124M);
    orion_gpt2_prefill_kv(g_weights, tokens, 1, kv, logits_kv);

    ASSERT(kv->current_len == 1, "KV cache len=%d, expected 1", kv->current_len);

    int argmax_fwd = orion_sample_token(logits_fwd, g_weights->vocab, 0.0f, 1.0f, NULL);
    int argmax_kv = orion_sample_token(logits_kv, g_weights->vocab, 0.0f, 1.0f, NULL);

    float max_err = 0.0f;
    for (int i = 0; i < g_weights->vocab; i++) {
        float err = fabsf(logits_fwd[i] - logits_kv[i]);
        if (err > max_err) max_err = err;
    }

    printf("  prefill_kv: argmax_fwd=%d, argmax_kv=%d, max_logit_err=%.6f\n",
           argmax_fwd, argmax_kv, max_err);

    orion_kv_cache_free(kv);
    free(logits_fwd);
    free(logits_kv);

    ASSERT(argmax_fwd == argmax_kv, "Argmax mismatch: fwd=%d, kv=%d", argmax_fwd, argmax_kv);
    PASS("T037: prefill_kv matches forward pass");
}

static void test_decode_step(void) {
    // Prefill "Hello" then decode next token
    // Compare: prefill "Hello," (2 tokens) logits vs prefill "Hello" + decode ","
    int tokens_2[] = {15496, 11};  // "Hello,"
    float* logits_ref = (float*)malloc(g_weights->vocab * sizeof(float));
    orion_gpt2_forward_cpu(g_weights, tokens_2, 2, logits_ref);
    int argmax_ref = orion_sample_token(logits_ref, g_weights->vocab, 0.0f, 1.0f, NULL);

    // Prefill "Hello" with KV cache, then decode step with token 11 (",")
    int tokens_1[] = {15496};
    float* logits_decode = (float*)malloc(g_weights->vocab * sizeof(float));

    OrionKVCache* kv = orion_kv_cache_create(&kGPT2_124M);
    float* logits_prefill = (float*)malloc(g_weights->vocab * sizeof(float));
    orion_gpt2_prefill_kv(g_weights, tokens_1, 1, kv, logits_prefill);

    // Now decode with token 11
    orion_gpt2_decode_step(g_weights, kv, 11, logits_decode);
    int argmax_decode = orion_sample_token(logits_decode, g_weights->vocab, 0.0f, 1.0f, NULL);

    ASSERT(kv->current_len == 2, "KV cache len=%d, expected 2", kv->current_len);

    printf("  decode_step: argmax_ref=%d, argmax_decode=%d\n", argmax_ref, argmax_decode);

    orion_kv_cache_free(kv);
    free(logits_ref);
    free(logits_decode);
    free(logits_prefill);

    ASSERT(argmax_ref == argmax_decode,
           "Decode step argmax=%d doesn't match prefill argmax=%d", argmax_decode, argmax_ref);
    PASS("T039: decode_step matches 2-token prefill");
}

static void test_multi_step_decode(void) {
    // Prefill "The quick brown fox" then decode 3 tokens
    // Verify each decode step produces the same argmax as the equivalent N-token prefill
    int prompt[] = {464, 2068, 7586, 21831};  // "The quick brown fox"
    int prompt_len = 4;

    OrionKVCache* kv = orion_kv_cache_create(&kGPT2_124M);
    float* logits = (float*)malloc(g_weights->vocab * sizeof(float));

    // Prefill
    orion_gpt2_prefill_kv(g_weights, prompt, prompt_len, kv, logits);
    int tok1 = orion_sample_token(logits, g_weights->vocab, 0.0f, 1.0f, NULL);

    // Decode 2 more tokens
    orion_gpt2_decode_step(g_weights, kv, tok1, logits);
    int tok2 = orion_sample_token(logits, g_weights->vocab, 0.0f, 1.0f, NULL);

    orion_gpt2_decode_step(g_weights, kv, tok2, logits);
    int tok3 = orion_sample_token(logits, g_weights->vocab, 0.0f, 1.0f, NULL);

    printf("  multi_step: generated tokens: %d %d %d (cache_len=%d)\n",
           tok1, tok2, tok3, kv->current_len);

    // prefill(4) + 2 decode steps = cache_len 6 (tok3 sampled but not yet decoded)
    ASSERT(kv->current_len == 6, "KV cache len=%d, expected 6", kv->current_len);

    // Cross-check: run full prefill for prompt + tok1 and verify tok2 matches
    int extended[] = {464, 2068, 7586, 21831, tok1};
    float* logits_check = (float*)malloc(g_weights->vocab * sizeof(float));
    orion_gpt2_forward_cpu(g_weights, extended, 5, logits_check);
    int tok2_check = orion_sample_token(logits_check, g_weights->vocab, 0.0f, 1.0f, NULL);

    printf("  multi_step: tok2_decode=%d, tok2_prefill=%d\n", tok2, tok2_check);

    orion_kv_cache_free(kv);
    free(logits);
    free(logits_check);

    ASSERT(tok2 == tok2_check,
           "Decode tok2=%d doesn't match prefill tok2=%d", tok2, tok2_check);
    PASS("T039: multi-step decode matches prefill cross-check");
}

static void test_sampling(void) {
    // Test argmax (temp=0)
    float logits[] = {1.0f, 3.0f, 2.0f, 0.5f};
    int argmax = orion_sample_token(logits, 4, 0.0f, 1.0f, NULL);
    ASSERT(argmax == 1, "Argmax=%d, expected 1", argmax);

    // Test sampling (temp=1.0, top_p=0.9) — should produce varied results
    uint64_t rng = 12345;
    int counts[4] = {0};
    for (int i = 0; i < 1000; i++) {
        int tok = orion_sample_token(logits, 4, 1.0f, 0.9f, &rng);
        ASSERT(tok >= 0 && tok < 4, "Sampled token %d out of range", tok);
        counts[tok]++;
    }

    // Token 1 (logit=3.0) should be most common
    printf("  sampling: counts=[%d, %d, %d, %d]\n",
           counts[0], counts[1], counts[2], counts[3]);
    ASSERT(counts[1] > counts[0] && counts[1] > counts[2] && counts[1] > counts[3],
           "Token 1 should be most frequent");

    PASS("T040: sampling — argmax correct, nucleus sampling varied");
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        printf("=== Orion Decode + KV Cache Tests ===\n\n");

        printf("Loading weights...\n");
        g_weights = orion_gpt2_weights_load("model/blobs/gpt2_124m");
        if (!g_weights) {
            printf("FATAL: Failed to load weights\n");
            return 1;
        }

        test_sampling();
        test_prefill_kv();
        test_decode_step();
        test_multi_step_decode();

        printf("\n=== Results: %d passed, %d failed ===\n",
               tests_passed, tests_failed);

        orion_gpt2_weights_free(g_weights);
        return tests_failed > 0 ? 1 : 0;
    }
}
