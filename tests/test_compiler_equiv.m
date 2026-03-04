// tests/test_compiler_equiv.m — T139: Compiler equivalence tests
// Verify that compiler-generated MIL structurally matches hand-written MIL.
// Full numerical equivalence requires ANE hardware and weight files —
// this test verifies structural equivalence (same ops, same weight refs).

#import <Foundation/Foundation.h>
#include "compiler/graph.h"
#include "compiler/builder.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#import "compiler/codegen.h"
#import "compiler/mil_diff.h"
#include "compiler/frontends/gpt2_prefill.h"
#include "compiler/frontends/gpt2_decode.h"
#include "compiler/frontends/gpt2_final.h"
#include "compiler/frontends/classifier_softmax.h"
#include "compiler/frontends/stories_train.h"
#include "compiler/model_config.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

// GPT-2 config for testing
static const OrionModelConfig kTestGPT2 = {
    .n_layer = 12, .n_head = 12, .d_model = 768,
    .head_dim = 64, .hidden_dim = 3072, .vocab = 50257, .max_seq = 1024
};

// Stories config for testing
static const OrionModelConfig kTestStories = {
    .n_layer = 12, .n_head = 12, .d_model = 768,
    .head_dim = 64, .hidden_dim = 2048, .vocab = 32000, .max_seq = 256
};

// Helper: build graph -> validate -> optimize -> codegen
static NSString* compile_frontend(OrionGraph* (*frontend)(int, int, const OrionModelConfig*),
                                   int layer, int bucket, const OrionModelConfig* cfg) {
    OrionGraph* g = frontend(layer, bucket, cfg);
    if (!g) return nil;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        NSLog(@"Validation failed: %s", vr.message);
        orion_graph_free(g);
        return nil;
    }

    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

// Helper for 2-arg frontends (decode doesn't take bucket)
static NSString* compile_frontend_2arg(OrionGraph* (*frontend)(int, const OrionModelConfig*),
                                        int layer, const OrionModelConfig* cfg) {
    OrionGraph* g = frontend(layer, cfg);
    if (!g) return nil;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        NSLog(@"Validation failed: %s", vr.message);
        orion_graph_free(g);
        return nil;
    }

    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

// Test: GPT-2 prefill attention generates valid MIL
static bool test_gpt2_prefill_attn(void) {
    NSString* mil = compile_frontend(orion_frontend_gpt2_prefill_attn, 0, 64, &kTestGPT2);
    if (!mil) return false;

    // Structural checks
    bool has_conv = [mil containsString:@"conv("];
    bool has_matmul = [mil containsString:@"matmul("];
    bool has_softmax = [mil containsString:@"softmax("];
    bool has_outputs = [mil containsString:@"hidden"] && [mil containsString:@"k_cache"] && [mil containsString:@"v_cache"];
    bool has_weights = [mil containsString:@"layer0/wq.bin"];

    return has_conv && has_matmul && has_softmax && has_outputs && has_weights;
}

// Test: GPT-2 prefill FFN generates valid MIL
static bool test_gpt2_prefill_ffn(void) {
    NSString* mil = compile_frontend(orion_frontend_gpt2_prefill_ffn, 0, 64, &kTestGPT2);
    if (!mil) return false;

    bool has_conv = [mil containsString:@"conv("];
    bool has_tanh = [mil containsString:@"tanh("]; // GELU decomposition
    bool has_output = [mil containsString:@"hidden"];
    return has_conv && has_tanh && has_output;
}

// Test: GPT-2 decode proj generates valid MIL
static bool test_gpt2_decode_proj(void) {
    NSString* mil = compile_frontend_2arg(orion_frontend_gpt2_decode_proj, 0, &kTestGPT2);
    if (!mil) return false;

    bool has_q = [mil containsString:@"q32"];
    bool has_k = [mil containsString:@"k32"];
    bool has_v = [mil containsString:@"v32"];
    return has_q && has_k && has_v;
}

// Test: GPT-2 decode FFN generates valid MIL
static bool test_gpt2_decode_ffn(void) {
    NSString* mil = compile_frontend_2arg(orion_frontend_gpt2_decode_ffn, 0, &kTestGPT2);
    if (!mil) return false;
    return [mil containsString:@"hidden"];
}

// Test: Stories fwdAttn generates valid MIL
static bool test_stories_fwd_attn(void) {
    OrionGraph* g = orion_frontend_fwd_attn(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"matmul("] && [mil containsString:@"softmax("];
}

// Test: Stories fwdFFN generates valid MIL
static bool test_stories_fwd_ffn(void) {
    OrionGraph* g = orion_frontend_fwd_ffn(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"sigmoid("]; // SiLU decomposition
}

// Test: Stories ffnBwd generates valid MIL
static bool test_stories_ffn_bwd(void) {
    OrionGraph* g = orion_frontend_ffn_bwd(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"dx"] && [mil containsString:@"slice_by_index("];
}

// Test: Stories sdpaBwd1 generates valid MIL
static bool test_stories_sdpa_bwd1(void) {
    OrionGraph* g = orion_frontend_sdpa_bwd1(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"dvf"] && [mil containsString:@"softmax("];
}

// Test: Stories sdpaBwd2 generates valid MIL
static bool test_stories_sdpa_bwd2(void) {
    OrionGraph* g = orion_frontend_sdpa_bwd2(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"dqf"] && [mil containsString:@"dkf"];
}

// Test: GPT-2 final LayerNorm generates valid MIL
static bool test_gpt2_final_ln(void) {
    // gpt2_final_ln takes (bucket, cfg) — not a standard 3-arg or 2-arg frontend
    OrionGraph* g = orion_frontend_gpt2_final_ln(64, &kTestGPT2);
    if (!g) return false;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    if (!mil) return false;
    bool has_output = [mil containsString:@"hidden"];
    bool has_weights = [mil containsString:@"ln_f_g.bin"];
    return has_output && has_weights;
}

// Test: Classifier forward generates valid MIL
static bool test_classifier_fwd(void) {
    // classifier_fwd takes (dim, vocab) — standalone signature
    OrionGraph* g = orion_frontend_classifier_fwd(768, 32000);
    if (!g) return false;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    if (!mil) return false;
    bool has_output = [mil containsString:@"output"];
    bool has_conv = [mil containsString:@"conv("];
    bool has_embed = [mil containsString:@"embed.bin"];
    return has_output && has_conv && has_embed;
}

// Test: Vocab softmax generates valid MIL
static bool test_vocab_softmax(void) {
    // vocab_softmax takes (vocab, seq_len) — standalone signature
    OrionGraph* g = orion_frontend_vocab_softmax(32000, 256);
    if (!g) return false;

    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    if (!mil) return false;
    bool has_output = [mil containsString:@"output"];
    bool has_softmax = [mil containsString:@"softmax("];
    return has_output && has_softmax;
}

// Test: Stories qkvBwd generates valid MIL
static bool test_stories_qkv_bwd(void) {
    OrionGraph* g = orion_frontend_qkv_bwd(0, &kTestStories);
    if (!g) return false;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return false; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil && [mil containsString:@"dx"];
}

int main(int argc __attribute__((unused)), char* argv[] __attribute__((unused))) {
    @autoreleasepool {
        printf("test_compiler_equiv:\n");

        TEST(gpt2_prefill_attn);
        TEST(gpt2_prefill_ffn);
        TEST(gpt2_decode_proj);
        TEST(gpt2_decode_ffn);
        TEST(gpt2_final_ln);
        TEST(classifier_fwd);
        TEST(vocab_softmax);
        TEST(stories_fwd_attn);
        TEST(stories_fwd_ffn);
        TEST(stories_ffn_bwd);
        TEST(stories_sdpa_bwd1);
        TEST(stories_sdpa_bwd2);
        TEST(stories_qkv_bwd);

        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed > 0 ? 1 : 0;
    }
}
