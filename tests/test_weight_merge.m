// test_weight_merge.m — regression tests for multi-BLOBFILE ANE programs.
//
// macOS 27's ANECompilerService rejects some program bundles containing more
// than one BLOBFILE weight file ("verifyBundleAtPath: invalid model"). The
// runtime now merges them into one Core ML storage-v2 file and rewrites the
// MIL references. These tests cover both the compile path and eval correctness.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>
#import "ane_runtime.h"
#import "mil_builder.h"
#import "compiler/kernel_adapter.h"
#import "compiler/frontends/gpt2_prefill.h"
#import "model/configs/gpt2_124m.h"

#define CONV_DIM 64
#define CONV_SEQ 64

static NSData* make_blob_const(int count, float value) {
    int data_bytes = count * (int)sizeof(_Float16);
    int total = 128 + data_bytes;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    buf[0] = 1;
    buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = (uint32_t)data_bytes;
    *(uint32_t *)(buf + 80) = 128;

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < count; i++) fp16[i] = (_Float16)value;

    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static NSData* make_identity_blob(int dim) {
    int count = dim * dim;
    int data_bytes = count * (int)sizeof(_Float16);
    int total = 128 + data_bytes;
    uint8_t *buf = (uint8_t *)calloc(total, 1);

    buf[0] = 1;
    buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = (uint32_t)data_bytes;
    *(uint32_t *)(buf + 80) = 128;

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int r = 0; r < dim; r++) fp16[r * dim + r] = (_Float16)1.0f;

    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

static IOSurfaceRef make_surface(int count) {
    size_t bytes = count * sizeof(float);
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

// Compile the real GPT-2 attention prefill program, but with generated blobs
// instead of downloaded model weights. This reproduces the 11-weight bundle
// that ANECompilerService rejects on macOS 27 without the merge.
static int test_ffn_bundle_compile(void) {
    printf("=== multi-BLOBFILE GPT-2 FFN compile ===\n");

    NSString *mil = orion_kernel_adapter_generate_mil(
        orion_frontend_gpt2_prefill_ffn, 0, 32, &kGPT2_124M);
    if (!mil) {
        printf("FAIL: could not generate FFN MIL\n");
        return 1;
    }

    NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
    struct { const char *name; int count; } weights[] = {
        {"ln2_g", 768}, {"ln2_b", 768},
        {"wfc", 3072 * 768}, {"bfc", 3072},
        {"wproj", 768 * 3072}, {"bproj", 768},
    };
    for (size_t i = 0; i < sizeof(weights) / sizeof(weights[0]); i++) {
        NSString *path = [NSString stringWithFormat:@"@model_path/layer0/%s.bin", weights[i].name];
        wdict[path] = @{@"offset": @0, @"data": make_blob_const(weights[i].count, 0.01f)};
    }
    OrionProgram *prog = orion_compile_mil_merged([mil UTF8String], wdict, "ffn_merge");
    if (!prog) {
        printf("FAIL: 6-weight FFN program did not compile\n");
        return 1;
    }
    orion_release_program(prog);
    printf("PASS: 6-weight FFN program compiled\n");
    return 0;
}

// Two identity convs, each backed by its own BLOBFILE. Verifies that the
// merged offset table points at the right tensors, not just that compile passes.
static int test_conv_merge_eval(void) {
    printf("=== two-BLOBFILE conv->conv eval ===\n");

    NSString *l1 = orion_mil_linear("l1", "x16", CONV_DIM, CONV_DIM, CONV_SEQ,
                                    "@model_path/weights/w1.bin", NULL);
    NSString *l2 = orion_mil_linear("l2", "l1_out", CONV_DIM, CONV_DIM, CONV_SEQ,
                                    "@model_path/weights/w2.bin", NULL);
    NSString *body = [NSString stringWithFormat:
        @"        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
         "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"x16\")];\n"
         "%@%@"
         "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
         "        tensor<fp32, [1, %d, 1, %d]> z = cast(dtype = to32, x = l2_out)[name = string(\"z\")];\n",
        CONV_DIM, CONV_SEQ, l1, l2, CONV_DIM, CONV_SEQ];
    NSString *mil = orion_mil_program(body,
        @[[NSString stringWithFormat:@"tensor<fp32, [1, %d, 1, %d]> x", CONV_DIM, CONV_SEQ]],
        @"z");

    NSDictionary *wdict = @{
        @"@model_path/weights/w1.bin": @{@"offset": @0, @"data": make_identity_blob(CONV_DIM)},
        @"@model_path/weights/w2.bin": @{@"offset": @0, @"data": make_identity_blob(CONV_DIM)},
    };

    OrionProgram *prog = orion_compile_mil_merged([mil UTF8String], wdict, "conv_merge");
    if (!prog) {
        printf("FAIL: two-weight conv->conv program did not compile\n");
        return 1;
    }

    int count = CONV_DIM * CONV_SEQ;
    IOSurfaceRef in = make_surface(count);
    IOSurfaceRef out = make_surface(count);
    IOSurfaceLock(in, 0, NULL);
    float *ip = (float *)IOSurfaceGetBaseAddress(in);
    for (int i = 0; i < count; i++) ip[i] = 2.0f;
    IOSurfaceUnlock(in, 0, NULL);

    IOSurfaceRef ins[] = {in};
    IOSurfaceRef outs[] = {out};
    if (!orion_eval(prog, ins, 1, outs, 1)) {
        printf("FAIL: two-weight conv->conv program did not evaluate\n");
        CFRelease(in); CFRelease(out); orion_release_program(prog);
        return 1;
    }

    int correct = 0;
    IOSurfaceLock(out, kIOSurfaceLockReadOnly, NULL);
    float *op = (float *)IOSurfaceGetBaseAddress(out);
    for (int i = 0; i < count; i++) {
        if (fabsf(op[i] - 2.0f) < 0.01f) correct++;
    }
    IOSurfaceUnlock(out, kIOSurfaceLockReadOnly, NULL);
    CFRelease(in); CFRelease(out); orion_release_program(prog);

    printf("two identity convs: %d/%d correct\n", correct, count);
    if (correct != count) {
        printf("FAIL: two identity convs should preserve input 2.0\n");
        return 1;
    }
    printf("PASS: two weight files merged and evaluated correctly\n");
    return 0;
}

int main(void) {
    @autoreleasepool {
        if (!orion_ane_init()) {
            printf("FAIL: orion_ane_init()\n");
            return 1;
        }

        int failed = 0;
        failed += test_ffn_bundle_compile();
        failed += test_conv_merge_eval();
        printf("\n%s\n", failed ? "FAILED" : "ALL PASS");
        return failed ? 1 : 0;
    }
}
