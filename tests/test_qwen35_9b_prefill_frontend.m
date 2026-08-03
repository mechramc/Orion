// tests/test_qwen35_9b_prefill_frontend.m — Qwen3.5-9B ANE frontend smoke

#import <Foundation/Foundation.h>
#include "compiler/frontends/qwen35_prefill.h"
#include "compiler/model_config.h"
#include "compiler/pass_uniform_outputs.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#import "compiler/codegen.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-40s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

static const OrionModelConfig kQwen35_9B = {
    .n_layer = 32,
    .n_head = 16,
    .n_kv_head = 4,
    .d_model = 4096,
    .head_dim = 256,
    .hidden_dim = 12288,
    .vocab = 248320,
    .max_seq = 262144,
};

static NSString* compile_graph(OrionGraph* g) {
    if (!g) return nil;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        NSLog(@"graph validate failed: %s", vr.message);
        orion_graph_free(g);
        return nil;
    }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

static bool test_q_proj(void) {
    OrionGraph* g = orion_frontend_qwen35_prefill_q_proj(3, 64, &kQwen35_9B);
    NSString* mil = compile_graph(g);
    if (!mil) return false;
    return [mil containsString:@"self_attn_q_proj.bin"] &&
           [mil containsString:@"input_layernorm.bin"] &&
           [mil containsString:@"q_proj"];
}

static bool test_kv_proj(void) {
    OrionGraph* g = orion_frontend_qwen35_prefill_kv_proj(3, 64, &kQwen35_9B);
    bool needs_uniform_pad = orion_pass_uniform_outputs(g);
    if (needs_uniform_pad) {
        NSLog(@"kv_proj unexpectedly needs uniform output padding");
        orion_graph_free(g);
        return false;
    }
    NSString* mil = compile_graph(g);
    if (!mil) return false;
    return [mil containsString:@"self_attn_k_proj.bin"] &&
           [mil containsString:@"self_attn_v_proj.bin"] &&
           [mil containsString:@"k_proj"] &&
           [mil containsString:@"v_proj"];
}

static bool test_ffn(void) {
    OrionGraph* g = orion_frontend_qwen35_prefill_ffn(3, 64, &kQwen35_9B);
    NSString* mil = compile_graph(g);
    if (!mil) return false;
    return [mil containsString:@"post_attention_layernorm.bin"] &&
           [mil containsString:@"mlp_gate_proj.bin"] &&
           [mil containsString:@"mlp_up_proj.bin"] &&
           [mil containsString:@"mlp_down_proj.bin"] &&
           [mil containsString:@"tanh("] &&
           [mil containsString:@"_sig"];
}

int main(int argc __attribute__((unused)), char* argv[] __attribute__((unused))) {
    @autoreleasepool {
        printf("test_qwen35_9b_prefill_frontend:\n");
        TEST(q_proj);
        TEST(kv_proj);
        TEST(ffn);
        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed > 0 ? 1 : 0;
    }
}
