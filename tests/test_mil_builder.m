// test_mil_builder.m — Test MIL builder helpers (T019-T023)
// Validates that generated MIL text compiles and evaluates correctly on ANE.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl \
//     -I ../core ../core/ane_runtime.m ../core/iosurface_tensor.m ../core/mil_builder.m \
//     test_mil_builder.m -o test_mil_builder
// Run:
//   ./test_mil_builder

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <math.h>
#import <stdio.h>
#import "ane_runtime.h"
#import "iosurface_tensor.h"
#import "mil_builder.h"

#define CH 256
#define SP 64

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

// Helper: build weight blob with constant fp16 values
static NSData *make_blob_const(int rows, int cols, float val) {
    int ws = rows * cols * 2; // fp16
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    // Header
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    // Fill fp16 data
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < rows * cols; i++) {
        fp16[i] = (_Float16)val;
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// Helper: build identity-like weight blob for conv [out, in, 1, 1]
// Sets diagonal to 1, others to 0 (only works when out == in)
static NSData *make_blob_identity(int dim) {
    int ws = dim * dim * 2;
    int tot = 128 + ws;
    uint8_t *b = (uint8_t *)calloc(tot, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = ws;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < dim; i++) {
        fp16[i * dim + i] = (_Float16)1.0f;
    }
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// Helper: create fp32 IOSurface with constant value
static IOSurfaceRef make_f32_surface(int count, float val) {
    size_t bytes = count * sizeof(float);
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
    IOSurfaceLock(s, 0, NULL);
    float *p = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = val;
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static float read_f32_elem(IOSurfaceRef s, int idx) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    float v = ((float *)IOSurfaceGetBaseAddress(s))[idx];
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    return v;
}

#pragma mark - Test: Program Wrapper (T023)

static void test_program_wrapper(void) {
    printf("\n=== Test: Program Wrapper (T023) ===\n");

    // Simple add program using orion_mil_program
    NSString *body =
        @"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        @"        tensor<fp16, [1, 256, 1, 64]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n"
        @"        tensor<fp16, [1, 256, 1, 64]> y16 = cast(dtype = to16, x = y)[name = string(\"cy\")];\n"
        @"        tensor<fp16, [1, 256, 1, 64]> z16 = add(x = x16, y = y16)[name = string(\"add\")];\n"
        @"        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        @"        tensor<fp32, [1, 256, 1, 64]> z = cast(dtype = to32, x = z16)[name = string(\"z\")];\n";

    NSString *prog_text = orion_mil_program(body,
        @[@"tensor<fp32, [1, 256, 1, 64]> x", @"tensor<fp32, [1, 256, 1, 64]> y"],
        @"z");

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], nil, "test_wrapper");
    CHECK(prog != NULL, "program wrapper compiles");

    if (prog) {
        int count = CH * SP;
        IOSurfaceRef ioX = make_f32_surface(count, 2.0f);
        IOSurfaceRef ioY = make_f32_surface(count, 3.0f);
        IOSurfaceRef ioZ = make_f32_surface(count, 0.0f);

        IOSurfaceRef ins[] = {ioX, ioY};
        IOSurfaceRef outs[] = {ioZ};
        bool ok = orion_eval(prog, ins, 2, outs, 1);
        CHECK(ok, "wrapper eval succeeds");

        float z0 = read_f32_elem(ioZ, 0);
        CHECK(fabsf(z0 - 5.0f) < 0.1f, "output ≈ 5.0 (2.0 + 3.0)");

        CFRelease(ioX); CFRelease(ioY); CFRelease(ioZ);
        orion_release_program(prog);
    }
}

#pragma mark - Test: SiLU Activation (T021)

static void test_silu(void) {
    printf("\n=== Test: SiLU Activation (T021) ===\n");

    // SiLU(x) = x * sigmoid(x). For x=1: sigmoid(1) ≈ 0.7311, SiLU(1) ≈ 0.7311
    NSString *body_ops = orion_mil_silu("silu", "x16", CH, SP);

    NSString *body = [NSString stringWithFormat:
        @"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n"
        "%@"
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> z = cast(dtype = to32, x = silu_out)[name = string(\"z\")];\n",
        CH, SP, body_ops, CH, SP];

    NSString *prog_text = orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", CH, SP]],
        @"z");

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], nil, "test_silu");
    CHECK(prog != NULL, "SiLU program compiles");

    if (prog) {
        int count = CH * SP;
        IOSurfaceRef ioX = make_f32_surface(count, 1.0f);
        IOSurfaceRef ioZ = make_f32_surface(count, 0.0f);

        IOSurfaceRef ins[] = {ioX};
        IOSurfaceRef outs[] = {ioZ};
        bool ok = orion_eval(prog, ins, 1, outs, 1);
        CHECK(ok, "SiLU eval succeeds");

        float z0 = read_f32_elem(ioZ, 0);
        float expected = 1.0f * (1.0f / (1.0f + expf(-1.0f))); // ~0.7311
        CHECK(fabsf(z0 - expected) < 0.01f, "SiLU(1.0) ≈ 0.731");
        printf("    z0 = %.4f (expected %.4f)\n", z0, expected);

        CFRelease(ioX); CFRelease(ioZ);
        orion_release_program(prog);
    }
}

#pragma mark - Test: GELU Activation (T021)

static void test_gelu(void) {
    printf("\n=== Test: GELU Activation (T021) ===\n");

    NSString *body_ops = orion_mil_gelu("gelu", "x16", CH, SP);

    NSString *body = [NSString stringWithFormat:
        @"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n"
        "%@"
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> z = cast(dtype = to32, x = gelu_out)[name = string(\"z\")];\n",
        CH, SP, body_ops, CH, SP];

    NSString *prog_text = orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", CH, SP]],
        @"z");

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], nil, "test_gelu");
    CHECK(prog != NULL, "GELU program compiles");

    if (prog) {
        int count = CH * SP;
        IOSurfaceRef ioX = make_f32_surface(count, 1.0f);
        IOSurfaceRef ioZ = make_f32_surface(count, 0.0f);

        IOSurfaceRef ins[] = {ioX};
        IOSurfaceRef outs[] = {ioZ};
        bool ok = orion_eval(prog, ins, 1, outs, 1);
        CHECK(ok, "GELU eval succeeds");

        float z0 = read_f32_elem(ioZ, 0);
        // GELU(1.0) ≈ 0.8412
        CHECK(z0 > 0.8f && z0 < 0.9f, "GELU(1.0) ≈ 0.841");
        printf("    z0 = %.4f\n", z0);

        CFRelease(ioX); CFRelease(ioZ);
        orion_release_program(prog);
    }
}

#pragma mark - Test: RMSNorm (T020)

static void test_rmsnorm(void) {
    printf("\n=== Test: RMSNorm (T020) ===\n");

    // RMSNorm with weight=1.0: out = x / rms(x)
    // For constant input x=2.0, rms = 2.0, so out ≈ 1.0
    NSData *wblob = make_blob_const(1, CH, 1.0f); // [1, CH, 1, 1] weight = 1.0

    NSString *body_ops = orion_mil_rmsnorm("rms", "x16", CH, SP,
                                            "@model_path/weights/rms_w.bin", 1e-5f);

    NSString *body = [NSString stringWithFormat:
        @"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n"
        "%@"
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> z = cast(dtype = to32, x = rms_out)[name = string(\"z\")];\n",
        CH, SP, body_ops, CH, SP];

    NSString *prog_text = orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", CH, SP]],
        @"z");

    NSDictionary *wdict = @{
        @"@model_path/weights/rms_w.bin": @{@"offset": @0, @"data": wblob}
    };

    OrionProgram *prog = orion_compile_mil([prog_text UTF8String], wdict, "test_rms");
    CHECK(prog != NULL, "RMSNorm program compiles");

    if (prog) {
        int count = CH * SP;
        IOSurfaceRef ioX = make_f32_surface(count, 2.0f);
        IOSurfaceRef ioZ = make_f32_surface(count, 0.0f);

        IOSurfaceRef ins[] = {ioX};
        IOSurfaceRef outs[] = {ioZ};
        bool ok = orion_eval(prog, ins, 1, outs, 1);
        CHECK(ok, "RMSNorm eval succeeds");

        float z0 = read_f32_elem(ioZ, 0);
        // For constant x=2.0, rms(x) = sqrt(mean(x^2)) = sqrt(4) = 2.0
        // out = x/rms(x) * w = 2/2 * 1 = 1.0
        CHECK(fabsf(z0 - 1.0f) < 0.05f, "RMSNorm(2.0, w=1.0) ≈ 1.0");
        printf("    z0 = %.4f (expected ~1.0)\n", z0);

        CFRelease(ioX); CFRelease(ioZ);
        orion_release_program(prog);
    }
}

int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Orion MIL Builder Test ===\n");

        if (!orion_ane_init()) {
            printf("FAIL: ANE init failed\n");
            return 1;
        }

        test_program_wrapper();
        test_silu();
        test_gelu();
        test_rmsnorm();

        printf("\n========================================\n");
        printf("Results: %d passed, %d failed\n", g_pass, g_fail);
        printf("Compiles used: %d\n", orion_compile_count());
        printf("========================================\n");

        return g_fail > 0 ? 1 : 0;
    }
}
