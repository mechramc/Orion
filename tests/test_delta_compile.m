// test_delta_compile.m — T151: Test orion_program_patch_weights
//
// Verifies that delta weight patching:
//   1. Produces output identical to fresh compile
//   2. Does not increment compile count
//   3. Works for multiple consecutive patches
//   4. Works at Stories110M scale (768x768 weights)
//
// Build:
//   cd /Users/murai-labs/Github/Orion && \
//   xcrun clang -O2 -fobjc-arc \
//     -framework Foundation -framework IOSurface -framework Accelerate -ldl -I . \
//     core/ane_runtime.m core/iosurface_tensor.m \
//     tests/test_delta_compile.m -o tests/test_delta_compile
//
// Run:
//   ./tests/test_delta_compile

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <stdio.h>
#import <math.h>
#import <mach/mach_time.h>
#import "core/ane_runtime.h"
#import "core/iosurface_tensor.h"

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while (0)

#pragma mark - BLOBFILE helper

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

#pragma mark - MIL text

static NSString* mil_linear(int in_dim, int out_dim, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
            "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = conv("
            "dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, "
            "weight=W, x=x)[name=string(\"y\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, seq, out_dim, in_dim, out_dim, in_dim, out_dim, seq];
}

#pragma mark - Helpers

static float* random_weights(int count, unsigned int seed) {
    float *w = (float *)malloc(count * sizeof(float));
    srand(seed);
    for (int i = 0; i < count; i++)
        w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    return w;
}

static float max_abs_diff(float *a, float *b, int count) {
    float max = 0;
    for (int i = 0; i < count; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > max) max = d;
    }
    return max;
}

static double time_ms(void) {
    static mach_timebase_info_data_t info;
    if (info.denom == 0) mach_timebase_info(&info);
    return (double)mach_absolute_time() * info.numer / info.denom / 1e6;
}

#pragma mark - Tests

static void test_basic_patch(void) {
    printf("\n=== Test: Basic delta patch (768x768) ===\n");

    int dim = 768, seq = 32;
    int wcount = dim * dim;
    NSString *mil = mil_linear(dim, dim, seq);

    // Create test input
    float *input = random_weights(dim * seq, 100);
    IOSurfaceRef in_surf = orion_tensor_create(dim, seq);
    orion_tensor_write_f32(in_surf, input, dim * seq);

    // Weight set A — compile normally (donor)
    float *wa = random_weights(wcount, 42);
    NSData *blob_a = make_blobfile(wa, wcount);
    NSDictionary *wdict_a = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": blob_a}};

    int count_before = orion_compile_count();
    OrionProgram *donor = orion_compile_mil(mil.UTF8String, wdict_a, "donor");
    CHECK(donor != NULL, "donor compiles");
    CHECK(orion_compile_count() == count_before + 1, "compile count incremented");

    // Eval donor
    IOSurfaceRef out_donor = orion_tensor_create(dim, seq);
    bool ok = orion_eval(donor, &in_surf, 1, &out_donor, 1);
    CHECK(ok, "donor eval succeeds");
    float *res_donor = (float *)malloc(dim * seq * sizeof(float));
    orion_tensor_read_f32(out_donor, res_donor, dim * seq);

    // Weight set B — patch
    float *wb = random_weights(wcount, 99);
    NSData *blob_b = make_blobfile(wb, wcount);
    NSDictionary *wdict_b = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": blob_b}};

    count_before = orion_compile_count();
    double t0 = time_ms();
    OrionProgram *patched = orion_program_patch_weights(donor, mil.UTF8String, wdict_b, "patched");
    double patch_ms = time_ms() - t0;
    CHECK(patched != NULL, "patch succeeds");
    CHECK(orion_compile_count() == count_before, "compile count NOT incremented");
    printf("  Patch time: %.2f ms\n", patch_ms);

    // Eval patched
    IOSurfaceRef out_patch = orion_tensor_create(dim, seq);
    ok = orion_eval(patched, &in_surf, 1, &out_patch, 1);
    CHECK(ok, "patched eval succeeds");
    float *res_patch = (float *)malloc(dim * seq * sizeof(float));
    orion_tensor_read_f32(out_patch, res_patch, dim * seq);

    // Compare: patched should differ from donor (different weights)
    float diff_ab = max_abs_diff(res_donor, res_patch, dim * seq);
    CHECK(diff_ab > 1e-4f, "patched output differs from donor");
    printf("  Donor vs patched max diff: %.6f\n", diff_ab);

    // Compare: patched should match fresh compile with B
    t0 = time_ms();
    OrionProgram *fresh_b = orion_compile_mil(mil.UTF8String, wdict_b, "fresh_B");
    double compile_ms = time_ms() - t0;
    CHECK(fresh_b != NULL, "fresh B compiles");
    printf("  Fresh compile time: %.2f ms\n", compile_ms);
    printf("  Speedup: %.1fx\n", compile_ms / patch_ms);

    IOSurfaceRef out_fresh = orion_tensor_create(dim, seq);
    ok = orion_eval(fresh_b, &in_surf, 1, &out_fresh, 1);
    CHECK(ok, "fresh B eval succeeds");
    float *res_fresh = (float *)malloc(dim * seq * sizeof(float));
    orion_tensor_read_f32(out_fresh, res_fresh, dim * seq);

    float diff_exact = max_abs_diff(res_patch, res_fresh, dim * seq);
    CHECK(diff_exact == 0.0f, "patched output EXACTLY matches fresh compile");
    printf("  Patch vs fresh max diff: %.8f\n", diff_exact);

    // Cleanup
    orion_release_program(donor);
    orion_release_program(patched);
    orion_release_program(fresh_b);
    CFRelease(in_surf); CFRelease(out_donor); CFRelease(out_patch); CFRelease(out_fresh);
    free(input); free(wa); free(wb);
    free(res_donor); free(res_patch); free(res_fresh);
}

static void test_consecutive_patches(void) {
    printf("\n=== Test: 10 consecutive patches (no compile) ===\n");

    int dim = 768, seq = 32;
    int wcount = dim * dim;
    NSString *mil = mil_linear(dim, dim, seq);

    float *input = random_weights(dim * seq, 200);
    IOSurfaceRef in_surf = orion_tensor_create(dim, seq);
    orion_tensor_write_f32(in_surf, input, dim * seq);

    // Compile donor once
    float *w0 = random_weights(wcount, 0);
    NSData *blob0 = make_blobfile(w0, wcount);
    NSDictionary *wdict0 = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": blob0}};
    OrionProgram *donor = orion_compile_mil(mil.UTF8String, wdict0, "donor");
    CHECK(donor != NULL, "donor compiles");

    int count_before = orion_compile_count();
    double total_patch_ms = 0;

    for (int i = 1; i <= 10; i++) {
        float *wi = random_weights(wcount, (unsigned int)i * 1000);
        NSData *blobi = make_blobfile(wi, wcount);
        NSDictionary *wdicti = @{@"@model_path/weights/w.bin": @{@"offset": @0, @"data": blobi}};

        double t0 = time_ms();
        OrionProgram *pi = orion_program_patch_weights(donor, mil.UTF8String, wdicti, "patch");
        double ms = time_ms() - t0;
        total_patch_ms += ms;

        CHECK(pi != NULL, "patch succeeds");

        IOSurfaceRef out = orion_tensor_create(dim, seq);
        bool ok = orion_eval(pi, &in_surf, 1, &out, 1);
        CHECK(ok, "eval succeeds");

        float *res = (float *)malloc(dim * seq * sizeof(float));
        orion_tensor_read_f32(out, res, dim * seq);
        // Just verify output is finite
        bool finite = true;
        for (int j = 0; j < dim * seq; j++) {
            if (!isfinite(res[j])) { finite = false; break; }
        }
        CHECK(finite, "output is finite");

        orion_release_program(pi);
        CFRelease(out);
        free(wi); free(res);
    }

    CHECK(orion_compile_count() == count_before, "no compiles during 10 patches");
    printf("  Total patch time for 10 patches: %.2f ms (avg %.2f ms)\n",
           total_patch_ms, total_patch_ms / 10.0);

    orion_release_program(donor);
    CFRelease(in_surf);
    free(input); free(w0);
}

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Orion Delta Compile Test (T151) ===\n");

        if (!orion_ane_init()) {
            printf("FATAL: ANE init failed\n");
            return 1;
        }

        test_basic_patch();
        test_consecutive_patches();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Compiles used: %d (of ~119 budget)\n", orion_compile_count());
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}
