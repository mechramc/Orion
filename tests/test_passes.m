// tests/test_passes.m — T129: Optimization pass tests

#import <Foundation/Foundation.h>
#include "compiler/graph.h"
#include "compiler/builder.h"
#include "compiler/pass_dce.h"
#include "compiler/pass_identity.h"
#include "compiler/pass_conv_bias.h"
#include "compiler/pass_cast.h"
#include "compiler/pipeline.h"
#import "compiler/codegen.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

// Count live nodes
static int count_live(const OrionGraph* g) {
    int c = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        if (g->nodes[i].is_live) c++;
    }
    return c;
}

// Test: DCE removes unreachable nodes
static bool test_dce_removes_dead(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");        // used
    int z = orion_gb_mul(g, x, x, "z_dead");   // NOT used by output
    orion_gb_output(g, y, "y");

    int before = count_live(g);
    bool changed = orion_pass_dce(g);
    int after = count_live(g);

    orion_graph_free(g);
    return changed && after < before && after == 2; // x + y
}

// Test: identity elimination removes redundant cast
static bool test_identity_removes_same_cast(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);

    // Cast fp16 -> fp16 (redundant)
    int y = orion_gb_cast(g, x, ORION_DTYPE_FP16, "y", NULL);
    orion_gb_output(g, y, "y");

    bool changed = orion_pass_identity(g);
    orion_graph_free(g);
    return changed; // Should eliminate the redundant cast
}

// Test: identity elimination removes identity op
static bool test_identity_removes_identity_op(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_identity(g, x, "y");
    orion_gb_output(g, y, "y");

    int before = count_live(g);
    bool changed = orion_pass_identity(g);
    int after = count_live(g);
    orion_graph_free(g);
    return changed && after < before;
}

// Test: cast round-trip elimination
static bool test_cast_roundtrip(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);

    // fp16 -> fp32 -> fp16 (round-trip)
    int y = orion_gb_cast(g, x, ORION_DTYPE_FP32, "y", NULL);
    int z = orion_gb_cast(g, y, ORION_DTYPE_FP16, "z", NULL);
    orion_gb_output(g, z, "z");

    bool changed = orion_pass_cast(g);
    orion_graph_free(g);
    return changed; // Should eliminate the round-trip
}

// Test: pipeline reaches fixpoint
static bool test_pipeline_fixpoint(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);

    // Add some redundancy
    int y = orion_gb_identity(g, x, "ident");
    int z = orion_gb_cast(g, y, ORION_DTYPE_FP16, "cast_same", NULL); // same dtype
    int w = orion_gb_mul(g, z, z, "w");
    int dead = orion_gb_add(g, x, x, "dead"); // unreachable
    orion_gb_output(g, w, "w");

    int before = count_live(g);
    orion_pipeline_optimize(g);
    int after = count_live(g);

    // Should have eliminated identity, same-cast, and dead node
    bool ok = after < before;
    orion_graph_free(g);
    return ok;
}

// Test: pipeline doesn't infinite loop
static bool test_pipeline_no_infinite_loop(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    orion_gb_output(g, y, "y");

    // Pipeline should terminate quickly on a clean graph
    orion_pipeline_optimize(g);
    bool ok = true; // If we get here, no infinite loop
    orion_graph_free(g);
    return ok;
}

int main(int argc __attribute__((unused)), char* argv[] __attribute__((unused))) {
    @autoreleasepool {
        printf("test_passes:\n");

        TEST(dce_removes_dead);
        TEST(identity_removes_same_cast);
        TEST(identity_removes_identity_op);
        TEST(cast_roundtrip);
        TEST(pipeline_fixpoint);
        TEST(pipeline_no_infinite_loop);

        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed > 0 ? 1 : 0;
    }
}
