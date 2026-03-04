// tests/test_graph_ir.m — T122: Graph IR tests
// Build small graphs, emit MIL, validate output.

#import <Foundation/Foundation.h>
#include "compiler/graph.h"
#include "compiler/builder.h"
#include "compiler/validate.h"
#include "compiler/topo.h"
#import "compiler/codegen.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-50s ", #name); \
    if (test_##name()) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

// Test: create graph, add nodes, validate
static bool test_graph_create_and_validate(void) {
    OrionGraph* g = orion_graph_create();
    if (!g) return false;

    // Input
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, shape);
    if (x != 0) { orion_graph_free(g); return false; }

    // Cast to fp16
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", NULL);

    // Simple add: x16 + x16
    int y = orion_gb_add(g, x16, x16, "y");
    orion_gb_output(g, y, "y");

    OrionValidationResult vr = orion_graph_validate(g);
    orion_graph_free(g);
    return vr.valid;
}

// Test: topological sort
static bool test_topo_sort(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    int y = orion_gb_add(g, x, x, "y");
    int z = orion_gb_mul(g, y, x, "z");
    orion_gb_output(g, z, "z");

    int count = 0;
    int* order = orion_topo_sort(g, &count);
    bool ok = order != NULL && count == 3;
    // First should be x (no deps), then y, then z
    if (ok) {
        ok = order[0] == x && order[1] == y && order[2] == z;
    }
    free(order);
    orion_graph_free(g);
    return ok;
}

// Test: MIL codegen produces valid-looking output
static bool test_codegen_simple(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", NULL);
    int y = orion_gb_add(g, x16, x16, "y");

    int out_shape[4] = {1, 768, 1, 64};
    int out = orion_gb_cast(g, y, ORION_DTYPE_FP32, "out", out_shape);
    orion_gb_output(g, out, "out");

    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    if (!mil) return false;
    // Check for key MIL elements
    bool has_program = [mil containsString:@"program(1.3)"];
    bool has_func = [mil containsString:@"func main<ios18>"];
    bool has_cast = [mil containsString:@"cast("];
    bool has_add = [mil containsString:@"add("];
    bool has_output = [mil containsString:@"-> (out)"];
    return has_program && has_func && has_cast && has_add && has_output;
}

// Test: layernorm composite expands correctly
static bool test_layernorm_composite(void) {
    OrionGraph* g = orion_graph_create();
    int d = 768, s = 64;
    int shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);

    int g_shape[4] = {1, d, 1, 1};
    int gamma = orion_gb_const_weight(g, "gamma", ORION_DTYPE_FP16, g_shape,
                                       "@model_path/gamma.bin", 64);
    int beta = orion_gb_const_weight(g, "beta", ORION_DTYPE_FP16, g_shape,
                                      "@model_path/beta.bin", 64);

    int ln = orion_gb_layernorm(g, x, gamma, beta, 1e-5f, "ln", d, s);
    orion_gb_output(g, ln, "ln_out");

    OrionValidationResult vr = orion_graph_validate(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return vr.valid && mil != nil &&
           [mil containsString:@"reduce_mean"] &&
           [mil containsString:@"pow"];
}

// Test: GELU composite expands correctly
static bool test_gelu_composite(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 3072, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);

    int gelu = orion_gb_gelu(g, x, "act", 3072, 64);
    orion_gb_output(g, gelu, "act_out");

    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil &&
           [mil containsString:@"tanh("] &&
           [mil containsString:@"0.044715"] &&
           [mil containsString:@"0.7979"];
}

// Test: validation rejects banned concat
static bool test_validate_banned_concat(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);

    // Manually add a banned concat node
    OrionNode n;
    memset(&n, 0, sizeof(n));
    n.op = ORION_OP_CONCAT_BANNED;
    snprintf(n.name, ORION_MAX_NAME, "bad_concat");
    n.dtype = ORION_DTYPE_FP16;
    memcpy(n.shape, shape, sizeof(n.shape));
    n.inputs[0] = x;
    n.n_inputs = 1;
    n.is_live = true;
    int bad = orion_graph_add_node(g, &n);
    orion_gb_output(g, bad, "bad");

    OrionValidationResult vr = orion_graph_validate(g);
    orion_graph_free(g);

    return !vr.valid; // Should fail validation
}

// Test: validation detects missing output
static bool test_validate_no_output(void) {
    OrionGraph* g = orion_graph_create();
    int shape[4] = {1, 768, 1, 64};
    orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);
    // No output marked

    OrionValidationResult vr = orion_graph_validate(g);
    orion_graph_free(g);
    return !vr.valid; // Should fail
}

// Test: linear (conv1x1) + bias
static bool test_linear_with_bias(void) {
    OrionGraph* g = orion_graph_create();
    int d = 768, s = 64;
    int shape[4] = {1, d, 1, s};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, shape);

    int out = orion_gb_linear(g, x, "fc", d, 3072, s,
                               "@model_path/wfc.bin", "@model_path/bfc.bin");
    orion_gb_output(g, out, "fc_out");

    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);

    return mil != nil &&
           [mil containsString:@"conv("] &&
           [mil containsString:@"BLOBFILE"];
}

int main(int argc __attribute__((unused)), char* argv[] __attribute__((unused))) {
    @autoreleasepool {
        printf("test_graph_ir:\n");

        TEST(graph_create_and_validate);
        TEST(topo_sort);
        TEST(codegen_simple);
        TEST(layernorm_composite);
        TEST(gelu_composite);
        TEST(validate_banned_concat);
        TEST(validate_no_output);
        TEST(linear_with_bias);

        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed > 0 ? 1 : 0;
    }
}
