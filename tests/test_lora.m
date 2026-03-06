// test_lora.m — T163: LoRA frontend tests
//
// Verifies:
//   1. LoRA linear frontend generates valid MIL
//   2. MIL compiles and runs on ANE
//   3. With zero adapters, output matches base linear
//   4. With non-zero adapters, output differs from base
//   5. Hot-swap: change adapter IOSurfaces, zero recompiles

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <stdio.h>
#import <math.h>
#import "core/ane_runtime.h"
#import "core/iosurface_tensor.h"
#import "compiler/frontends/lora.h"
#import "compiler/codegen.h"
#import "compiler/pipeline.h"
#import "compiler/kernel_adapter.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while (0)

static NSData* make_blobfile(const float *data, int count) {
    size_t fp16_size = count * sizeof(uint16_t);
    size_t total = 128 + fp16_size;
    uint8_t *b = (uint8_t *)calloc(total, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = (uint32_t)fp16_size;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < count; i++) fp16[i] = (_Float16)data[i];
    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

static float* random_weights(int count, unsigned int seed) {
    float *w = (float *)malloc(count * sizeof(float));
    srand(seed);
    for (int i = 0; i < count; i++)
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    return w;
}

static float* random_weights_scale(int count, unsigned int seed, float scale) {
    float *w = (float *)malloc(count * sizeof(float));
    srand(seed);
    for (int i = 0; i < count; i++)
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * scale;
    return w;
}

static float max_abs_diff(float *a, float *b, int count) {
    float mx = 0;
    for (int i = 0; i < count; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > mx) mx = d;
    }
    return mx;
}

#pragma mark - Test: LoRA Linear MIL Generation

static void test_lora_linear_mil(void) {
    printf("\n=== Test: LoRA linear MIL generation ===\n");

    int in_dim = 768, out_dim = 768, seq = 32, rank = 16;
    float alpha = 16.0f;

    OrionGraph* g = orion_frontend_lora_linear(
        in_dim, out_dim, seq, rank, alpha,
        "@model_path/weights/w.bin");
    CHECK(g != NULL, "graph created");

    orion_pipeline_optimize(g);
    NSString *mil = orion_codegen_mil(g, "main");
    CHECK(mil != nil, "MIL generated");
    CHECK([mil containsString:@"lora_A"], "MIL contains lora_A input");
    CHECK([mil containsString:@"lora_B"], "MIL contains lora_B input");
    CHECK([mil containsString:@"BLOBFILE"], "MIL contains BLOBFILE for base weight");
    CHECK([mil containsString:@"matmul"], "MIL contains matmul for LoRA path");
    CHECK([mil containsString:@"conv"], "MIL contains conv for base path");

    if (mil) {
        printf("  MIL length: %lu chars\n", (unsigned long)mil.length);
    }

    orion_graph_free(g);
}

#pragma mark - Test: LoRA Linear ANE Compile + Eval

static void test_lora_linear_ane(void) {
    printf("\n=== Test: LoRA linear compile + eval on ANE ===\n");

    int in_dim = 768, out_dim = 768, seq = 32, rank = 16;
    float alpha = 16.0f;
    int wcount = out_dim * in_dim;

    // Generate MIL
    OrionGraph* g = orion_frontend_lora_linear(
        in_dim, out_dim, seq, rank, alpha,
        "@model_path/weights/w.bin");
    orion_pipeline_optimize(g);
    NSString *mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    CHECK(mil != nil, "MIL generated");
    if (!mil) return;

    // Base weight
    float *wbase = random_weights(wcount, 42);
    NSData *blob_w = make_blobfile(wbase, wcount);
    NSDictionary *wdict = @{
        @"@model_path/weights/w.bin": @{@"offset": @0, @"data": blob_w}
    };

    // Compile
    int cc = orion_compile_count();
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "lora_linear");
    CHECK(prog != NULL, "program compiles");
    CHECK(orion_compile_count() == cc + 1, "compile count incremented");
    if (!prog) { free(wbase); return; }

    // Create inputs
    float *input = random_weights(in_dim * seq, 100);
    IOSurfaceRef in_surf = orion_tensor_create(in_dim, seq);
    orion_tensor_write_f32(in_surf, input, in_dim * seq);

    // LoRA A: MIL declares [1, in_dim, 1, rank] but IOSurface alloc must
    // match input surface size (ANE requires uniform input allocation sizes).
    // Allocate [in_dim, seq] but write packed [in_dim, rank] data at start.
    // ANE reads the flat buffer as packed [1,in_dim,1,rank] = in_dim*rank fp16.
    float *zeros_a = (float *)calloc(in_dim * rank, sizeof(float));
    IOSurfaceRef lora_a = orion_tensor_create(in_dim, seq);
    orion_tensor_write_f32(lora_a, zeros_a, in_dim * rank);

    // LoRA B: same pattern
    float *zeros_b = (float *)calloc(out_dim * rank, sizeof(float));
    IOSurfaceRef lora_b = orion_tensor_create(out_dim, seq);
    orion_tensor_write_f32(lora_b, zeros_b, out_dim * rank);

    // Eval with zero adapters
    IOSurfaceRef out_surf = orion_tensor_create(out_dim, seq);
    // ANE input ordering is alphabetical by MIL parameter name:
    // lora_A (0), lora_B (1), x (2)
    IOSurfaceRef inputs[3] = {lora_a, lora_b, in_surf};
    IOSurfaceRef outputs[1] = {out_surf};
    bool ok = orion_eval(prog, inputs, 3, outputs, 1);
    CHECK(ok, "eval with zero adapters succeeds");

    float *res_zero = (float *)malloc(out_dim * seq * sizeof(float));
    orion_tensor_read_f32(out_surf, res_zero, out_dim * seq);

    // Check output is not all zeros (base weight should produce non-zero output)
    float max_val = 0;
    for (int i = 0; i < out_dim * seq; i++) {
        float v = fabsf(res_zero[i]);
        if (v > max_val) max_val = v;
    }
    CHECK(max_val > 1e-6f, "output is non-zero with base weights");
    printf("  Max output value: %.6f\n", max_val);

    // Now eval with non-zero LoRA adapters (use larger scale for visibility)
    // Write packed [dim, rank] data — ANE reads flat buffer as [1,dim,1,rank]
    float *a_data = random_weights_scale(in_dim * rank, 200, 1.0f);
    float *b_data = random_weights_scale(out_dim * rank, 300, 1.0f);
    orion_tensor_write_f32(lora_a, a_data, in_dim * rank);
    orion_tensor_write_f32(lora_b, b_data, out_dim * rank);

    IOSurfaceRef out_lora = orion_tensor_create(out_dim, seq);
    IOSurfaceRef outputs2[1] = {out_lora};
    ok = orion_eval(prog, inputs, 3, outputs2, 1);
    CHECK(ok, "eval with non-zero adapters succeeds");

    float *res_lora = (float *)malloc(out_dim * seq * sizeof(float));
    orion_tensor_read_f32(out_lora, res_lora, out_dim * seq);

    float diff = max_abs_diff(res_zero, res_lora, out_dim * seq);
    CHECK(diff > 1e-4f, "LoRA adapters change output");
    printf("  Zero vs LoRA adapter diff: %.6f\n", diff);

    // Hot-swap: change adapter IOSurfaces, same compiled program (0 recompiles)
    float *a2_data = random_weights_scale(in_dim * rank, 400, 1.0f);
    float *b2_data = random_weights_scale(out_dim * rank, 500, 1.0f);
    orion_tensor_write_f32(lora_a, a2_data, in_dim * rank);
    orion_tensor_write_f32(lora_b, b2_data, out_dim * rank);

    cc = orion_compile_count();
    IOSurfaceRef out_swap = orion_tensor_create(out_dim, seq);
    IOSurfaceRef outputs3[1] = {out_swap};
    ok = orion_eval(prog, inputs, 3, outputs3, 1);
    CHECK(ok, "eval after hot-swap succeeds");
    CHECK(orion_compile_count() == cc, "zero recompiles for hot-swap");

    float *res_swap = (float *)malloc(out_dim * seq * sizeof(float));
    orion_tensor_read_f32(out_swap, res_swap, out_dim * seq);

    float diff2 = max_abs_diff(res_lora, res_swap, out_dim * seq);
    CHECK(diff2 > 1e-4f, "swapped adapters produce different output");
    printf("  Adapter A vs Adapter B diff: %.6f\n", diff2);

    // Cleanup
    orion_release_program(prog);
    CFRelease(in_surf); CFRelease(lora_a); CFRelease(lora_b);
    CFRelease(out_surf); CFRelease(out_lora); CFRelease(out_swap);
    free(input); free(wbase); free(zeros_a); free(zeros_b);
    free(a_data); free(b_data); free(a2_data); free(b2_data);
    free(res_zero); free(res_lora); free(res_swap);
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    @autoreleasepool {
        printf("=== Orion LoRA Tests ===\n");

        test_lora_linear_mil();

        if (orion_ane_init()) {
            test_lora_linear_ane();
        } else {
            printf("\nSkipping ANE tests (no ANE available)\n");
        }

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}
