// test_program_cache.m — T084: Program Cache Test
// Tests: orion_cache_lookup, orion_cache_store, orion_cache_evict, orion_cache_clear, orion_cache_size
//
// Build:
//   xcrun clang -O2 -fobjc-arc \
//     -framework Foundation -framework IOSurface -ldl -I . \
//     core/ane_runtime.m core/iosurface_tensor.m core/ane_program_cache.m \
//     tests/test_program_cache.m -o tests/test_program_cache
// Run:
//   ./tests/test_program_cache

#import <Foundation/Foundation.h>
#import <stdio.h>
#import "core/ane_runtime.h"
#import "core/ane_program_cache.h"

#define CH 256
#define SP 64

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; printf("  PASS: %s\n", msg); } \
    else { g_fail++; printf("  FAIL: %s\n", msg); } \
} while(0)

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

static OrionProgram* compile_test_program(const char *tag) {
    return orion_compile_mil(add_mil_text(), @{}, tag);
}

#pragma mark - Tests

static void test_empty_cache(void) {
    printf("\ntest_empty_cache:\n");
    CHECK(orion_cache_size() == 0, "initial cache size is 0");

    OrionWeightsBinding wb = { .weights_id = "base", .bucket = 64 };
    OrionProgram *prog = orion_cache_lookup("test_kernel", 0, &wb);
    CHECK(prog == NULL, "lookup on empty cache returns NULL");
}

static void test_store_and_lookup(void) {
    printf("\ntest_store_and_lookup:\n");
    orion_cache_clear();

    OrionProgram *p = compile_test_program("cache_test_1");
    CHECK(p != NULL, "compiled test program");

    OrionWeightsBinding wb = { .weights_id = "base", .bucket = 64 };
    orion_cache_store("test_attn", 0, &wb, p);
    CHECK(orion_cache_size() == 1, "cache size is 1 after store");

    OrionProgram *hit = orion_cache_lookup("test_attn", 0, &wb);
    CHECK(hit == p, "lookup returns same pointer");

    // Different layer → miss
    OrionProgram *miss = orion_cache_lookup("test_attn", 1, &wb);
    CHECK(miss == NULL, "different layer_idx is a miss");

    // Different kernel → miss
    miss = orion_cache_lookup("test_ffn", 0, &wb);
    CHECK(miss == NULL, "different kernel_name is a miss");

    // Different weights_id → miss
    OrionWeightsBinding wb2 = { .weights_id = "ckpt_001", .bucket = 64 };
    miss = orion_cache_lookup("test_attn", 0, &wb2);
    CHECK(miss == NULL, "different weights_id is a miss");

    // Different bucket → miss
    OrionWeightsBinding wb3 = { .weights_id = "base", .bucket = 128 };
    miss = orion_cache_lookup("test_attn", 0, &wb3);
    CHECK(miss == NULL, "different bucket is a miss");

    orion_cache_clear();
}

static void test_store_multiple(void) {
    printf("\ntest_store_multiple:\n");
    orion_cache_clear();

    OrionWeightsBinding wb = { .weights_id = "base", .bucket = 64 };

    OrionProgram *p1 = compile_test_program("multi_1");
    OrionProgram *p2 = compile_test_program("multi_2");
    OrionProgram *p3 = compile_test_program("multi_3");
    CHECK(p1 && p2 && p3, "compiled 3 programs");

    orion_cache_store("attn", 0, &wb, p1);
    orion_cache_store("attn", 1, &wb, p2);
    orion_cache_store("ffn", 0, &wb, p3);
    CHECK(orion_cache_size() == 3, "cache size is 3");

    CHECK(orion_cache_lookup("attn", 0, &wb) == p1, "lookup attn L0");
    CHECK(orion_cache_lookup("attn", 1, &wb) == p2, "lookup attn L1");
    CHECK(orion_cache_lookup("ffn", 0, &wb) == p3, "lookup ffn L0");

    orion_cache_clear();
    CHECK(orion_cache_size() == 0, "clear empties cache");
}

static void test_evict_by_weights_id(void) {
    printf("\ntest_evict_by_weights_id:\n");
    orion_cache_clear();

    OrionWeightsBinding wb_base = { .weights_id = "base", .bucket = 64 };
    OrionWeightsBinding wb_ckpt = { .weights_id = "ckpt_001", .bucket = 64 };

    OrionProgram *p1 = compile_test_program("evict_1");
    OrionProgram *p2 = compile_test_program("evict_2");
    OrionProgram *p3 = compile_test_program("evict_3");
    OrionProgram *p4 = compile_test_program("evict_4");

    orion_cache_store("attn", 0, &wb_base, p1);
    orion_cache_store("attn", 1, &wb_base, p2);
    orion_cache_store("attn", 0, &wb_ckpt, p3);
    orion_cache_store("ffn", 0, &wb_ckpt, p4);
    CHECK(orion_cache_size() == 4, "cache size is 4");

    // Evict ckpt_001 — should remove p3 and p4
    orion_cache_evict("ckpt_001");
    CHECK(orion_cache_size() == 2, "evict ckpt_001 leaves 2");

    CHECK(orion_cache_lookup("attn", 0, &wb_base) == p1, "base attn L0 still cached");
    CHECK(orion_cache_lookup("attn", 1, &wb_base) == p2, "base attn L1 still cached");
    CHECK(orion_cache_lookup("attn", 0, &wb_ckpt) == NULL, "ckpt attn L0 evicted");
    CHECK(orion_cache_lookup("ffn", 0, &wb_ckpt) == NULL, "ckpt ffn L0 evicted");

    orion_cache_clear();
}

static void test_store_replace(void) {
    printf("\ntest_store_replace:\n");
    orion_cache_clear();

    OrionWeightsBinding wb = { .weights_id = "base", .bucket = 64 };

    OrionProgram *p1 = compile_test_program("replace_1");
    OrionProgram *p2 = compile_test_program("replace_2");

    orion_cache_store("attn", 0, &wb, p1);
    CHECK(orion_cache_size() == 1, "size 1 after first store");
    CHECK(orion_cache_lookup("attn", 0, &wb) == p1, "lookup returns p1");

    // Replace with p2 (p1 should be released by cache)
    orion_cache_store("attn", 0, &wb, p2);
    CHECK(orion_cache_size() == 1, "size still 1 after replace");
    CHECK(orion_cache_lookup("attn", 0, &wb) == p2, "lookup returns p2 after replace");

    orion_cache_clear();
}

static void test_evict_nonexistent(void) {
    printf("\ntest_evict_nonexistent:\n");
    orion_cache_clear();

    // Should not crash
    orion_cache_evict("nonexistent_id");
    CHECK(orion_cache_size() == 0, "evict nonexistent does not crash");

    // Clear empty cache should not crash
    orion_cache_clear();
    CHECK(orion_cache_size() == 0, "clear empty does not crash");
}

static void test_cache_with_eval(void) {
    printf("\ntest_cache_with_eval:\n");
    orion_cache_clear();

    OrionWeightsBinding wb = { .weights_id = "base", .bucket = SP };

    // Compile and store
    OrionProgram *prog = compile_test_program("eval_cached");
    CHECK(prog != NULL, "compiled program for eval test");
    orion_cache_store("add_op", -1, &wb, prog);

    // Look up and eval
    OrionProgram *cached = orion_cache_lookup("add_op", -1, &wb);
    CHECK(cached == prog, "lookup returns cached program");

    // Create I/O surfaces and eval
    int count = CH * SP;
    size_t bytes = count * sizeof(float);
    IOSurfaceRef ioX = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
    IOSurfaceRef ioY = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
    IOSurfaceRef ioZ = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});

    // Fill X with 1.0, Y with 2.0
    IOSurfaceLock(ioX, 0, NULL);
    float *px = IOSurfaceGetBaseAddress(ioX);
    for (int i = 0; i < count; i++) px[i] = 1.0f;
    IOSurfaceUnlock(ioX, 0, NULL);

    IOSurfaceLock(ioY, 0, NULL);
    float *py = IOSurfaceGetBaseAddress(ioY);
    for (int i = 0; i < count; i++) py[i] = 2.0f;
    IOSurfaceUnlock(ioY, 0, NULL);

    IOSurfaceRef ins[] = {ioX, ioY};
    IOSurfaceRef outs[] = {ioZ};
    bool ok = orion_eval(cached, ins, 2, outs, 1);
    CHECK(ok, "eval with cached program succeeds");

    // Read result
    IOSurfaceLock(ioZ, kIOSurfaceLockReadOnly, NULL);
    float *pz = IOSurfaceGetBaseAddress(ioZ);
    float sum = 0;
    for (int i = 0; i < count; i++) sum += pz[i];
    IOSurfaceUnlock(ioZ, kIOSurfaceLockReadOnly, NULL);

    float expected = 3.0f * count;
    CHECK(fabsf(sum - expected) < count * 0.01f, "cached program produces correct result (1+2=3)");

    CFRelease(ioX); CFRelease(ioY); CFRelease(ioZ);
    orion_cache_clear();
}

static void test_weights_id_step(void) {
    printf("\ntest_weights_id_step:\n");

    orion_weights_id_reset(0);
    CHECK(orion_weights_id_current_step() == 0, "reset to 0");

    const char *id1 = orion_weights_id_next_step();
    CHECK(strcmp(id1, "step_00000001") == 0, "first step is step_00000001");
    CHECK(orion_weights_id_current_step() == 1, "current step is 1");

    const char *id2 = orion_weights_id_next_step();
    CHECK(strcmp(id2, "step_00000002") == 0, "second step is step_00000002");

    // Reset to checkpoint step
    orion_weights_id_reset(1000);
    CHECK(orion_weights_id_current_step() == 1000, "reset to 1000");

    const char *id3 = orion_weights_id_next_step();
    CHECK(strcmp(id3, "step_00001001") == 0, "after reset(1000), next is step_00001001");
}

static void test_weights_id_with_cache(void) {
    printf("\ntest_weights_id_with_cache:\n");
    orion_cache_clear();
    orion_weights_id_reset(0);

    // Simulate training: compile with step_1, then evict and recompile with step_2
    // Must strdup() because next_step returns static buffer
    char *wid1 = strdup(orion_weights_id_next_step()); // "step_00000001"
    OrionWeightsBinding wb1 = { .weights_id = wid1, .bucket = 64 };

    OrionProgram *p1 = compile_test_program("train_step1");
    orion_cache_store("fwd_attn", 0, &wb1, p1);
    CHECK(orion_cache_size() == 1, "stored step 1 program");

    // Adam update → new weights_id
    char *wid2 = strdup(orion_weights_id_next_step()); // "step_00000002"
    OrionWeightsBinding wb2 = { .weights_id = wid2, .bucket = 64 };

    // Evict old weights
    orion_cache_evict(wid1);
    CHECK(orion_cache_size() == 0, "evicted step 1");

    OrionProgram *p2 = compile_test_program("train_step2");
    orion_cache_store("fwd_attn", 0, &wb2, p2);
    CHECK(orion_cache_size() == 1, "stored step 2 program");
    CHECK(orion_cache_lookup("fwd_attn", 0, &wb2) == p2, "lookup step 2 hits");
    CHECK(orion_cache_lookup("fwd_attn", 0, &wb1) == NULL, "lookup step 1 misses");

    free(wid1); free(wid2);
    orion_cache_clear();
}

int main(void) {
    @autoreleasepool {
        printf("=== test_program_cache ===\n");

        if (!orion_ane_init()) {
            fprintf(stderr, "SKIP: ANE not available\n");
            return 0;
        }

        test_empty_cache();
        test_store_and_lookup();
        test_store_multiple();
        test_evict_by_weights_id();
        test_store_replace();
        test_evict_nonexistent();
        test_cache_with_eval();
        test_weights_id_step();
        test_weights_id_with_cache();

        printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
        return g_fail > 0 ? 1 : 0;
    }
}
