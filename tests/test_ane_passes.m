// tests/test_ane_passes.m — T133: ANE-specific pass tests

#import <Foundation/Foundation.h>
#include "compiler/graph.h"
#include "compiler/builder.h"
#include "compiler/pass_sram.h"
#include "compiler/pass_uniform_outputs.h"
#include "compiler/pass_ane_validate.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

// Test: SRAM estimation for known dimensions
static bool test_sram_normal(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 256}; // 768*256*2 = 393KB < 32MB
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");

    bool spill = orion_pass_sram(g);
    orion_graph_free(g);
    return !spill; // Should NOT have spill risk
}

// Test: SRAM estimation flags large tensors
static bool test_sram_large_tensor(void) {
    OrionGraph* g = orion_graph_create();
    // 32768 * 1024 * 2 = 64MB > 32MB
    int shape[4] = {1, 32768, 1, 1024};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");

    bool spill = orion_pass_sram(g);
    orion_graph_free(g);
    return spill; // Should flag spill risk
}

// Test: tensor bytes calculation
static bool test_tensor_bytes(void) {
    OrionNode n;
    memset(&n, 0, sizeof(n));
    n.dtype = ORION_DTYPE_FP16;
    n.shape[0] = 1; n.shape[1] = 768; n.shape[2] = 1; n.shape[3] = 64;

    int64_t bytes = orion_node_tensor_bytes(&n);
    return bytes == 1 * 768 * 1 * 64 * 2; // fp16 = 2 bytes
}

// Test: uniform outputs detects mixed sizes
static bool test_uniform_outputs_mixed(void) {
    OrionGraph* g = orion_graph_create();
    int shape_d[4] = {1, 768, 1, 256};
    int shape_h[4] = {1, 2048, 1, 256};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape_d);
    int y = orion_gb_input(g, "y", ORION_DTYPE_FP16, shape_h);

    orion_gb_output(g, x, "out_d");
    orion_gb_output(g, y, "out_h");

    bool needs_pad = orion_pass_uniform_outputs(g);
    int max_ch = orion_outputs_max_channels(g);

    orion_graph_free(g);
    return needs_pad && max_ch == 2048;
}

// Test: uniform outputs passes for same sizes
static bool test_uniform_outputs_same(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 256};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");

    orion_gb_output(g, x, "out1");
    orion_gb_output(g, y, "out2");

    bool needs_pad = orion_pass_uniform_outputs(g);
    orion_graph_free(g);
    return !needs_pad; // Same size, no padding needed
}

// Test: ANE validation catches concat
static bool test_ane_validate_concat(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);

    OrionNode n;
    memset(&n, 0, sizeof(n));
    n.op = ORION_OP_CONCAT_BANNED;
    snprintf(n.name, ORION_MAX_NAME, "bad");
    n.dtype = ORION_DTYPE_FP16;
    memcpy(n.shape, shape, sizeof(n.shape));
    n.inputs[0] = x;
    n.n_inputs = 1;
    n.is_live = true;
    int bad = orion_graph_add_node(g, &n);
    orion_gb_output(g, bad, "bad");

    OrionANEValidationResult r = orion_pass_ane_validate(g);
    orion_graph_free(g);
    return !r.valid && r.n_errors > 0 && r.errors[0].constraint_id == 1;
}

// Test: ANE validation passes valid graph
static bool test_ane_validate_valid(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64}; // 768*64*2 = 98KB > 49KB minimum
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");

    OrionANEValidationResult r = orion_pass_ane_validate(g);
    orion_graph_free(g);
    return r.valid;
}

// Test: ANE validation catches small tensors
static bool test_ane_validate_small_tensor(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 32, 1, 1}; // 32*1*2 = 64 bytes << 49KB
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");

    OrionANEValidationResult r = orion_pass_ane_validate(g);
    orion_graph_free(g);
    return !r.valid; // Should fail — tensor too small
}

int main(int argc __attribute__((unused)), char* argv[] __attribute__((unused))) {
    @autoreleasepool {
        printf("test_ane_passes:\n");

        TEST(sram_normal);
        TEST(sram_large_tensor);
        TEST(tensor_bytes);
        TEST(uniform_outputs_mixed);
        TEST(uniform_outputs_same);
        TEST(ane_validate_concat);
        TEST(ane_validate_valid);
        TEST(ane_validate_small_tensor);

        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed > 0 ? 1 : 0;
    }
}
