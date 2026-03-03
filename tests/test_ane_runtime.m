// test_ane_runtime.m — T018: ANE Runtime Integration Test
// Tests: orion_ane_init, orion_compile_mil, orion_eval, orion_release_program
// Also tests: orion_tensor_create, orion_tensor_write_f32, orion_tensor_read_f32
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I ../core ../core/ane_runtime.m ../core/iosurface_tensor.m \
//     test_ane_runtime.m -o test_ane_runtime
// Run:
//   ./test_ane_runtime

#import <Foundation/Foundation.h>
#import <stdio.h>
#import <math.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"

#define CH 256
#define SP 64

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// Generate a simple MIL program: z = x + y (fp32 I/O, fp16 internal)
static const char *add_mil_text(void) {
    static char buf[2048];
    snprintf(buf, sizeof(buf),
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x, tensor<fp32, [1, %d, 1, %d]> y) {\n"
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = cast(dtype = to16, x = y)[name = string(\"cy\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> z16 = add(x = x16, y = y16)[name = string(\"add_op\")];\n"
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> z = cast(dtype = to32, x = z16)[name = string(\"out\")];\n"
        "    } -> (z);\n"
        "}\n",
        CH, SP, CH, SP, CH, SP, CH, SP, CH, SP, CH, SP);
    return buf;
}

// Generate MIL: z = x * 2.0 (to verify different operation)
static const char *scale_mil_text(void) {
    static char buf[2048];
    snprintf(buf, sizeof(buf),
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n"
        "        fp16 two = const()[name = string(\"two\"), val = fp16(2.0)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> z16 = mul(x = x16, y = two)[name = string(\"mul_op\")];\n"
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> z = cast(dtype = to32, x = z16)[name = string(\"out\")];\n"
        "    } -> (z);\n"
        "}\n",
        CH, SP, CH, SP, CH, SP, CH, SP);
    return buf;
}

static void test_init(void) {
    printf("\n=== Test: ANE Init ===\n");
    bool ok = orion_ane_init();
    CHECK(ok, "orion_ane_init() succeeds");
    // Calling twice should be idempotent
    ok = orion_ane_init();
    CHECK(ok, "orion_ane_init() idempotent");
}

static void test_tensor_roundtrip(void) {
    printf("\n=== Test: Tensor fp32 Round-trip (T012-T014) ===\n");
    int count = CH * SP;
    IOSurfaceRef s = orion_tensor_create(CH, SP);
    CHECK(s != NULL, "orion_tensor_create returns non-NULL");

    float *src = (float *)malloc(count * sizeof(float));
    float *dst = (float *)malloc(count * sizeof(float));
    for (int i = 0; i < count; i++) {
        src[i] = (float)(i % 1000) * 0.01f - 5.0f; // range [-5, 5)
    }

    orion_tensor_write_f32(s, src, count);
    orion_tensor_read_f32(s, dst, count);

    float max_err = 0;
    for (int i = 0; i < count; i++) {
        float err = fabsf(src[i] - dst[i]);
        if (err > max_err) max_err = err;
    }
    CHECK(max_err < 1e-2f, "fp32→fp16→fp32 round-trip max error < 0.01");
    printf("    max_err = %.6f\n", max_err);

    free(src);
    free(dst);
    orion_tensor_release(s);
}

static void test_compile_eval_add(void) {
    printf("\n=== Test: Compile + Eval (z = x + y) (T015-T016) ===\n");
    OrionProgram *prog = orion_compile_mil(add_mil_text(), nil, "test_add");
    CHECK(prog != NULL, "orion_compile_mil returns non-NULL");
    if (!prog) return;

    int count = CH * SP;

    // Create fp32 IOSurfaces (matching MIL fp32 I/O)
    size_t bytes = count * sizeof(float);
    IOSurfaceRef ioX = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
    IOSurfaceRef ioY = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
    IOSurfaceRef ioZ = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});

    // Fill inputs
    IOSurfaceLock(ioX, 0, NULL);
    float *xp = (float *)IOSurfaceGetBaseAddress(ioX);
    for (int i = 0; i < count; i++) xp[i] = 1.5f;
    IOSurfaceUnlock(ioX, 0, NULL);

    IOSurfaceLock(ioY, 0, NULL);
    float *yp = (float *)IOSurfaceGetBaseAddress(ioY);
    for (int i = 0; i < count; i++) yp[i] = 2.5f;
    IOSurfaceUnlock(ioY, 0, NULL);

    IOSurfaceRef ins[] = {ioX, ioY};
    IOSurfaceRef outs[] = {ioZ};
    bool ok = orion_eval(prog, ins, 2, outs, 1);
    CHECK(ok, "orion_eval returns true");

    // Read and verify
    IOSurfaceLock(ioZ, kIOSurfaceLockReadOnly, NULL);
    float *zp = (float *)IOSurfaceGetBaseAddress(ioZ);
    int correct = 0;
    for (int i = 0; i < count; i++) {
        if (fabsf(zp[i] - 4.0f) < 0.1f) correct++;
    }
    IOSurfaceUnlock(ioZ, kIOSurfaceLockReadOnly, NULL);

    CHECK(correct == count, "all elements ≈ 4.0 (1.5 + 2.5)");
    printf("    correct: %d/%d\n", correct, count);

    CFRelease(ioX); CFRelease(ioY); CFRelease(ioZ);
    orion_release_program(prog);
}

static void test_compile_eval_scale(void) {
    printf("\n=== Test: Compile + Eval (z = x * 2) — Single Input ===\n");
    OrionProgram *prog = orion_compile_mil(scale_mil_text(), nil, "test_scale");
    CHECK(prog != NULL, "compile scale program");
    if (!prog) return;

    int count = CH * SP;
    size_t bytes = count * sizeof(float);
    IOSurfaceRef ioX = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
    IOSurfaceRef ioZ = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});

    IOSurfaceLock(ioX, 0, NULL);
    float *xp = (float *)IOSurfaceGetBaseAddress(ioX);
    for (int i = 0; i < count; i++) xp[i] = 3.0f;
    IOSurfaceUnlock(ioX, 0, NULL);

    IOSurfaceRef ins[] = {ioX};
    IOSurfaceRef outs[] = {ioZ};
    bool ok = orion_eval(prog, ins, 1, outs, 1);
    CHECK(ok, "eval scale program");

    IOSurfaceLock(ioZ, kIOSurfaceLockReadOnly, NULL);
    float *zp = (float *)IOSurfaceGetBaseAddress(ioZ);
    int correct = 0;
    for (int i = 0; i < count; i++) {
        if (fabsf(zp[i] - 6.0f) < 0.1f) correct++;
    }
    IOSurfaceUnlock(ioZ, kIOSurfaceLockReadOnly, NULL);

    CHECK(correct == count, "all elements ≈ 6.0 (3.0 * 2)");
    printf("    correct: %d/%d\n", correct, count);

    CFRelease(ioX); CFRelease(ioZ);
    orion_release_program(prog);
}

static void test_release_loop(void) {
    printf("\n=== Test: Compile + Release 10× (T017) ===\n");
    int compiled = 0;
    for (int i = 0; i < 10; i++) {
        OrionProgram *prog = orion_compile_mil(add_mil_text(), nil, "release_test");
        if (prog) {
            compiled++;
            orion_release_program(prog);
        }
    }
    CHECK(compiled == 10, "10/10 compile+release cycles");
    printf("    compile_count = %d\n", orion_compile_count());
}

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Orion ANE Runtime Integration Test ===\n");

        test_init();
        test_tensor_roundtrip();
        test_compile_eval_add();
        test_compile_eval_scale();
        test_release_loop();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}
