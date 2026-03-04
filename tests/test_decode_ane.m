// test_decode_ane.m — T100: Test ANE decode MIL generators
//
// Tests that compiler frontend generates valid MIL programs
// that compile and eval on ANE with seq=ORION_GRAPH_DECODE_SEQ (16).
//
// Build:
//   cd /Users/murai-labs/Github/Orion && xcrun clang -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK \
//     -framework Foundation -framework IOSurface -framework Accelerate -ldl -I . \
//     core/{ane_runtime,iosurface_tensor,mil_builder}.m \
//     compiler/{kernel_adapter,frontends/gpt2_decode}.m \
//     tests/test_decode_ane.m -o tests/test_decode_ane
//
// Run:
//   ./tests/test_decode_ane

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <stdio.h>
#import <math.h>
#import <sys/time.h>
#import "core/ane_runtime.h"
#import "core/iosurface_tensor.h"
#import "core/mil_builder.h"
#include "compiler/kernel_adapter.h"
#include "compiler/frontends/gpt2_decode.h"

static int g_pass = 0, g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        g_fail++; return; \
    } \
} while(0)

#define PASS(msg) do { \
    printf("  PASS: %s\n", msg); \
    g_pass++; \
} while(0)

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

#pragma mark - BLOBFILE helper

static NSData* make_blobfile(int num_elements, float fill_value) {
    size_t data_bytes = num_elements * sizeof(uint16_t);
    size_t total = 128 + data_bytes;
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = (uint32_t)data_bytes;
    *(uint32_t *)(buf + 80) = 128;
    _Float16 *fp16_data = (_Float16 *)(buf + 128);
    for (int i = 0; i < num_elements; i++) fp16_data[i] = (_Float16)fill_value;
    NSData *blob = [NSData dataWithBytes:buf length:total];
    free(buf);
    return blob;
}

#pragma mark - IOSurface helpers

// Create fp32 surface for [1, channels, 1, seq] ANE layout
static IOSurfaceRef make_f32_surface(int count) {
    size_t bytes = count * sizeof(float);
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

// Write fp32 data into IOSurface in ANE layout [1, C, 1, S].
// data is in column-major: data[c * seq + s] for channel c at seq position s.
static void write_f32_surface(IOSurfaceRef s, const float *data, int count) {
    IOSurfaceLock(s, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(s), data, count * sizeof(float));
    IOSurfaceUnlock(s, 0, NULL);
}

static void read_f32_surface(IOSurfaceRef s, float *out, int count) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    memcpy(out, IOSurfaceGetBaseAddress(s), count * sizeof(float));
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// Create a [1, channels, 1, seq] surface with val at seq position 0, zero elsewhere.
// ANE layout: element at (c, s) is stored at offset c * seq + s.
static IOSurfaceRef make_decode_input(int channels, int seq, float val) {
    int count = channels * seq;
    IOSurfaceRef s = make_f32_surface(count);
    float *data = (float *)calloc(count, sizeof(float));
    // Set position 0 for each channel
    for (int c = 0; c < channels; c++) {
        data[c * seq + 0] = val;
    }
    write_f32_surface(s, data, count);
    free(data);
    return s;
}

// Read seq position 0 from a [1, channels, 1, seq] surface.
static void read_decode_output(IOSurfaceRef s, float *out, int channels, int seq) {
    int count = channels * seq;
    float *data = (float *)malloc(count * sizeof(float));
    read_f32_surface(s, data, count);
    for (int c = 0; c < channels; c++) {
        out[c] = data[c * seq + 0];
    }
    free(data);
}

#pragma mark - Weight dict builders

static NSDictionary* make_proj_weights(int layer_idx, int d) {
    NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
    NSString *ln1_g = [NSString stringWithFormat:@"@model_path/layer%d/ln1_g.bin", layer_idx];
    NSString *ln1_b = [NSString stringWithFormat:@"@model_path/layer%d/ln1_b.bin", layer_idx];
    wdict[ln1_g] = @{@"offset": @0, @"data": make_blobfile(d, 1.0f)};
    wdict[ln1_b] = @{@"offset": @0, @"data": make_blobfile(d, 0.0f)};

    NSString *wq = [NSString stringWithFormat:@"@model_path/layer%d/wq.bin", layer_idx];
    NSString *bq = [NSString stringWithFormat:@"@model_path/layer%d/bq.bin", layer_idx];
    wdict[wq] = @{@"offset": @0, @"data": make_blobfile(d * d, 0.001f)};
    wdict[bq] = @{@"offset": @0, @"data": make_blobfile(d, 0.0f)};

    NSString *wk = [NSString stringWithFormat:@"@model_path/layer%d/wk.bin", layer_idx];
    NSString *bk = [NSString stringWithFormat:@"@model_path/layer%d/bk.bin", layer_idx];
    wdict[wk] = @{@"offset": @0, @"data": make_blobfile(d * d, 0.001f)};
    wdict[bk] = @{@"offset": @0, @"data": make_blobfile(d, 0.0f)};

    NSString *wv = [NSString stringWithFormat:@"@model_path/layer%d/wv.bin", layer_idx];
    NSString *bv = [NSString stringWithFormat:@"@model_path/layer%d/bv.bin", layer_idx];
    wdict[wv] = @{@"offset": @0, @"data": make_blobfile(d * d, 0.001f)};
    wdict[bv] = @{@"offset": @0, @"data": make_blobfile(d, 0.0f)};

    return wdict;
}

static NSDictionary* make_ffn_weights(int layer_idx, int d, int h) {
    NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
    NSString *ln2_g = [NSString stringWithFormat:@"@model_path/layer%d/ln2_g.bin", layer_idx];
    NSString *ln2_b = [NSString stringWithFormat:@"@model_path/layer%d/ln2_b.bin", layer_idx];
    wdict[ln2_g] = @{@"offset": @0, @"data": make_blobfile(d, 1.0f)};
    wdict[ln2_b] = @{@"offset": @0, @"data": make_blobfile(d, 0.0f)};

    NSString *wfc = [NSString stringWithFormat:@"@model_path/layer%d/wfc.bin", layer_idx];
    NSString *bfc = [NSString stringWithFormat:@"@model_path/layer%d/bfc.bin", layer_idx];
    wdict[wfc] = @{@"offset": @0, @"data": make_blobfile(h * d, 0.001f)};
    wdict[bfc] = @{@"offset": @0, @"data": make_blobfile(h, 0.0f)};

    NSString *wproj = [NSString stringWithFormat:@"@model_path/layer%d/wproj.bin", layer_idx];
    NSString *bproj = [NSString stringWithFormat:@"@model_path/layer%d/bproj.bin", layer_idx];
    wdict[wproj] = @{@"offset": @0, @"data": make_blobfile(d * h, 0.001f)};
    wdict[bproj] = @{@"offset": @0, @"data": make_blobfile(d, 0.0f)};

    return wdict;
}

#pragma mark - Tests

static void test_decode_proj_mil_generation(void) {
    printf("\n--- test_decode_proj_mil_generation ---\n");
    OrionModelConfig cfg = {
        .n_layer = 12, .n_head = 12, .d_model = 768,
        .head_dim = 64, .hidden_dim = 3072, .vocab = 50257, .max_seq = 1024
    };
    NSString *mil = orion_kernel_adapter_generate_mil_2arg(orion_frontend_gpt2_decode_proj,0, &cfg);
    ASSERT(mil != nil, "MIL text should not be nil");
    ASSERT(mil.length > 100, "MIL text should be substantial");
    ASSERT([mil containsString:@"func main<ios18>"], "Should have function header");
    ASSERT([mil containsString:@"[1,768,1,16]"], "Should use seq=16 tensors");
    ASSERT([mil containsString:@"-> (q32, k32, v32)"], "Should have 3 outputs");
    PASS("decode_proj MIL generation correct");
}

static void test_decode_ffn_mil_generation(void) {
    printf("\n--- test_decode_ffn_mil_generation ---\n");
    OrionModelConfig cfg = {
        .n_layer = 12, .n_head = 12, .d_model = 768,
        .head_dim = 64, .hidden_dim = 3072, .vocab = 50257, .max_seq = 1024
    };
    NSString *mil = orion_kernel_adapter_generate_mil_2arg(orion_frontend_gpt2_decode_ffn,0, &cfg);
    ASSERT(mil != nil, "MIL text should not be nil");
    ASSERT(mil.length > 100, "MIL text should be substantial");
    ASSERT([mil containsString:@"func main<ios18>"], "Should have function header");
    ASSERT([mil containsString:@"[1,768,1,16]"], "Should use seq=16 tensors");
    ASSERT([mil containsString:@"-> (hidden)"], "Should have single output");
    PASS("decode_ffn MIL generation correct");
}

static void test_decode_proj_compile_eval(void) {
    printf("\n--- test_decode_proj_compile_eval ---\n");
    OrionModelConfig cfg = {
        .n_layer = 12, .n_head = 12, .d_model = 768,
        .head_dim = 64, .hidden_dim = 3072, .vocab = 50257, .max_seq = 1024
    };
    int d = cfg.d_model;
    int seq = ORION_GRAPH_DECODE_SEQ;
    int count = d * seq;

    NSString *mil = orion_kernel_adapter_generate_mil_2arg(orion_frontend_gpt2_decode_proj,0, &cfg);
    NSDictionary *wdict = make_proj_weights(0, d);

    double t0 = time_ms();
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "decode_proj_L0");
    double t1 = time_ms();
    ASSERT(prog != NULL, "decode_proj should compile");
    printf("  compile time: %.1f ms\n", t1 - t0);

    // Input: x [1, d, 1, seq] with val=1.0 at position 0
    IOSurfaceRef ioX = make_decode_input(d, seq, 1.0f);

    // 3 outputs: Q, K, V — each [1, d, 1, seq]
    IOSurfaceRef ioQ = make_f32_surface(count);
    IOSurfaceRef ioK = make_f32_surface(count);
    IOSurfaceRef ioV = make_f32_surface(count);

    IOSurfaceRef ins[] = {ioX};
    IOSurfaceRef outs[] = {ioQ, ioK, ioV};

    // Warmup
    bool ok = orion_eval(prog, ins, 1, outs, 3);
    ASSERT(ok, "eval should succeed");

    // Measure
    double t2 = time_ms();
    for (int i = 0; i < 10; i++) {
        orion_eval(prog, ins, 1, outs, 3);
    }
    double t3 = time_ms();
    printf("  eval time: %.3f ms (avg of 10)\n", (t3 - t2) / 10);

    // Read position 0 from outputs
    float q_buf[768], k_buf[768], v_buf[768];
    read_decode_output(ioQ, q_buf, d, seq);
    read_decode_output(ioK, k_buf, d, seq);
    read_decode_output(ioV, v_buf, d, seq);

    // Check finite
    bool all_finite = true;
    for (int i = 0; i < d; i++) {
        if (!isfinite(q_buf[i]) || !isfinite(k_buf[i]) || !isfinite(v_buf[i]))
            all_finite = false;
    }
    ASSERT(all_finite, "Q/K/V outputs at position 0 should be finite");

    printf("  Q[0:4] = [%.4f, %.4f, %.4f, %.4f]\n", q_buf[0], q_buf[1], q_buf[2], q_buf[3]);
    printf("  K[0:4] = [%.4f, %.4f, %.4f, %.4f]\n", k_buf[0], k_buf[1], k_buf[2], k_buf[3]);
    printf("  V[0:4] = [%.4f, %.4f, %.4f, %.4f]\n", v_buf[0], v_buf[1], v_buf[2], v_buf[3]);

    orion_release_program(prog);
    CFRelease(ioX); CFRelease(ioQ); CFRelease(ioK); CFRelease(ioV);
    PASS("decode_proj compile + eval succeeds");
}

static void test_decode_ffn_compile_eval(void) {
    printf("\n--- test_decode_ffn_compile_eval ---\n");
    OrionModelConfig cfg = {
        .n_layer = 12, .n_head = 12, .d_model = 768,
        .head_dim = 64, .hidden_dim = 3072, .vocab = 50257, .max_seq = 1024
    };
    int d = cfg.d_model;
    int h = cfg.hidden_dim;
    int seq = ORION_GRAPH_DECODE_SEQ;
    int count = d * seq;

    NSString *mil = orion_kernel_adapter_generate_mil_2arg(orion_frontend_gpt2_decode_ffn,0, &cfg);
    NSDictionary *wdict = make_ffn_weights(0, d, h);

    double t0 = time_ms();
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "decode_ffn_L0");
    double t1 = time_ms();
    ASSERT(prog != NULL, "decode_ffn should compile");
    printf("  compile time: %.1f ms\n", t1 - t0);

    // Input: x [1, d, 1, seq] with val=0.5 at position 0
    IOSurfaceRef ioX = make_decode_input(d, seq, 0.5f);
    IOSurfaceRef ioH = make_f32_surface(count);

    IOSurfaceRef ins[] = {ioX};
    IOSurfaceRef outs[] = {ioH};

    bool ok = orion_eval(prog, ins, 1, outs, 1);
    ASSERT(ok, "eval should succeed");

    // Measure
    double t2 = time_ms();
    for (int i = 0; i < 10; i++) {
        orion_eval(prog, ins, 1, outs, 1);
    }
    double t3 = time_ms();
    printf("  eval time: %.3f ms (avg of 10)\n", (t3 - t2) / 10);

    // Read position 0
    float h_buf[768];
    read_decode_output(ioH, h_buf, d, seq);

    bool h_finite = true;
    for (int i = 0; i < d; i++) {
        if (!isfinite(h_buf[i])) h_finite = false;
    }
    ASSERT(h_finite, "hidden output at position 0 should be finite");

    // FFN includes residual: hidden = x + ffnproj(gelu(fc(ln2(x))))
    // With uniform x=0.5 at pos 0: LN(uniform)=0 → fc→0 → gelu(0)=0 → proj→0
    // So residual = x + 0 = 0.5
    float sum = 0;
    for (int i = 0; i < d; i++) sum += h_buf[i];
    float mean = sum / d;
    printf("  hidden[0:4] = [%.4f, %.4f, %.4f, %.4f]\n", h_buf[0], h_buf[1], h_buf[2], h_buf[3]);
    printf("  hidden mean = %.4f (expected ~0.5)\n", mean);
    ASSERT(fabsf(mean - 0.5f) < 0.1f, "hidden mean at pos 0 should be ~0.5 (residual)");

    orion_release_program(prog);
    CFRelease(ioX); CFRelease(ioH);
    PASS("decode_ffn compile + eval with correct residual");
}

static void test_decode_proj_multiple_layers(void) {
    printf("\n--- test_decode_proj_multiple_layers ---\n");
    OrionModelConfig cfg = {
        .n_layer = 12, .n_head = 12, .d_model = 768,
        .head_dim = 64, .hidden_dim = 3072, .vocab = 50257, .max_seq = 1024
    };
    int d = cfg.d_model;

    int layers[] = {0, 5, 11};
    for (int i = 0; i < 3; i++) {
        int l = layers[i];
        NSString *mil = orion_kernel_adapter_generate_mil_2arg(orion_frontend_gpt2_decode_proj,l, &cfg);
        NSString *expected_path = [NSString stringWithFormat:@"layer%d/wq.bin", l];
        bool has_path = [mil containsString:expected_path];
        ASSERT(has_path, "should reference correct blob path for layer");

        NSDictionary *wdict = make_proj_weights(l, d);
        OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "multi_layer_proj");
        ASSERT(prog != NULL, "layer should compile");
        orion_release_program(prog);
    }
    PASS("decode_proj compiles for multiple layer indices");
}

static void test_decode_ffn_multiple_layers(void) {
    printf("\n--- test_decode_ffn_multiple_layers ---\n");
    OrionModelConfig cfg = {
        .n_layer = 12, .n_head = 12, .d_model = 768,
        .head_dim = 64, .hidden_dim = 3072, .vocab = 50257, .max_seq = 1024
    };
    int d = cfg.d_model;
    int h = cfg.hidden_dim;

    int layers[] = {0, 5, 11};
    for (int i = 0; i < 3; i++) {
        int l = layers[i];
        NSString *mil = orion_kernel_adapter_generate_mil_2arg(orion_frontend_gpt2_decode_ffn,l, &cfg);
        NSString *expected_path = [NSString stringWithFormat:@"layer%d/wfc.bin", l];
        bool has_path = [mil containsString:expected_path];
        ASSERT(has_path, "should reference correct blob path for layer");

        NSDictionary *wdict = make_ffn_weights(l, d, h);
        OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "multi_layer_ffn");
        ASSERT(prog != NULL, "layer should compile");
        orion_release_program(prog);
    }
    PASS("decode_ffn compiles for multiple layer indices");
}

static void test_decode_perf(void) {
    printf("\n--- test_decode_perf ---\n");
    OrionModelConfig cfg = {
        .n_layer = 12, .n_head = 12, .d_model = 768,
        .head_dim = 64, .hidden_dim = 3072, .vocab = 50257, .max_seq = 1024
    };
    int d = cfg.d_model;
    int h = cfg.hidden_dim;
    int seq = ORION_GRAPH_DECODE_SEQ;
    int count = d * seq;

    NSString *proj_mil = orion_kernel_adapter_generate_mil_2arg(orion_frontend_gpt2_decode_proj,0, &cfg);
    NSDictionary *proj_wdict = make_proj_weights(0, d);
    OrionProgram *proj_prog = orion_compile_mil(proj_mil.UTF8String, proj_wdict, "perf_proj");
    ASSERT(proj_prog != NULL, "proj should compile");

    NSString *ffn_mil = orion_kernel_adapter_generate_mil_2arg(orion_frontend_gpt2_decode_ffn,0, &cfg);
    NSDictionary *ffn_wdict = make_ffn_weights(0, d, h);
    OrionProgram *ffn_prog = orion_compile_mil(ffn_mil.UTF8String, ffn_wdict, "perf_ffn");
    ASSERT(ffn_prog != NULL, "ffn should compile");

    IOSurfaceRef ioX = make_decode_input(d, seq, 0.5f);
    IOSurfaceRef ioQ = make_f32_surface(count);
    IOSurfaceRef ioK = make_f32_surface(count);
    IOSurfaceRef ioV = make_f32_surface(count);
    IOSurfaceRef ioH = make_f32_surface(count);

    // Warmup
    IOSurfaceRef proj_ins[] = {ioX};
    IOSurfaceRef proj_outs[] = {ioQ, ioK, ioV};
    orion_eval(proj_prog, proj_ins, 1, proj_outs, 3);

    IOSurfaceRef ffn_ins[] = {ioX};
    IOSurfaceRef ffn_outs[] = {ioH};
    orion_eval(ffn_prog, ffn_ins, 1, ffn_outs, 1);

    // Measure combined: proj + ffn per layer
    int n_runs = 50;
    double t0 = time_ms();
    for (int i = 0; i < n_runs; i++) {
        orion_eval(proj_prog, proj_ins, 1, proj_outs, 3);
        orion_eval(ffn_prog, ffn_ins, 1, ffn_outs, 1);
    }
    double t1 = time_ms();
    double per_layer_ms = (t1 - t0) / n_runs;

    printf("  ANE per-layer decode: %.3f ms (proj + ffn, seq=%d)\n", per_layer_ms, seq);
    printf("  Projected 12-layer ANE portion: %.1f ms\n", per_layer_ms * 12);
    ASSERT(per_layer_ms < 5.0f, "Per-layer ANE decode should be under 5ms");

    orion_release_program(proj_prog);
    orion_release_program(ffn_prog);
    CFRelease(ioX); CFRelease(ioQ); CFRelease(ioK); CFRelease(ioV); CFRelease(ioH);
    PASS("ANE decode within latency threshold");
}

int main(void) {
    @autoreleasepool {
        printf("=== T100: ANE Decode MIL Generator Tests ===\n");

        if (!orion_ane_init()) {
            fprintf(stderr, "FATAL: ANE not available\n");
            return 1;
        }

        test_decode_proj_mil_generation();
        test_decode_ffn_mil_generation();
        test_decode_proj_compile_eval();
        test_decode_ffn_compile_eval();
        test_decode_proj_multiple_layers();
        test_decode_ffn_multiple_layers();
        test_decode_perf();

        printf("\n=== Results: %d passed, %d failed (compiles used: %d) ===\n",
               g_pass, g_fail, orion_compile_count());
        return g_fail > 0 ? 1 : 0;
    }
}
