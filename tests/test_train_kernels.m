#import <Foundation/Foundation.h>
#import "core/ane_runtime.h"
#import "core/iosurface_tensor.h"
#import "model/configs/stories110m.h"
#import "kernels/training/stories_train_kernels.milgen.h"
#import "kernels/training/classifier_softmax.milgen.h"
#import "kernels/inference/gpt2_prefill_attn.milgen.h" // for causal mask

static int g_pass = 0, g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        NSLog(@"FAIL: %s — %@", __func__, msg); \
        g_fail++; return; \
    } \
} while(0)

#define PASS() do { NSLog(@"PASS: %s", __func__); g_pass++; } while(0)

#pragma mark - Helpers

// Build a weight dict with random fp16 blobs for a given set of blob paths.
// Paths should be relative (e.g., "layer0/rms1.bin"); @model_path/ prefix is added.
static NSDictionary* make_random_wdict(NSArray<NSString*>* paths, NSArray<NSNumber*>* sizes) {
    NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
    for (NSUInteger i = 0; i < paths.count; i++) {
        int n_elements = sizes[i].intValue;
        int data_bytes = n_elements * sizeof(_Float16);
        int total = 128 + data_bytes; // BLOBFILE: 64 header + 64 chunk + data

        uint8_t *buf = (uint8_t *)calloc(total, 1);
        buf[0] = 1; buf[4] = 2;
        buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
        buf[68] = 1;
        *(uint32_t *)(buf + 72) = data_bytes;
        *(uint32_t *)(buf + 80) = 128;

        _Float16 *fp16 = (_Float16 *)(buf + 128);
        for (int j = 0; j < n_elements; j++) {
            fp16[j] = (_Float16)(0.01f * ((j % 100) - 50));
        }

        NSData *blob = [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
        // Key must have @model_path/ prefix to match MIL blob references
        NSString *key = [NSString stringWithFormat:@"@model_path/%@", paths[i]];
        wdict[key] = @{@"offset": @0, @"data": blob};
    }
    return wdict;
}

#pragma mark - T064: fwdAttn MIL Generation + Compile Tests

static void test_fwd_attn_milgen(void) {
    NSString *mil = orion_milgen_fwd_attn(0, &kStories110M);
    ASSERT(mil != nil, @"should generate MIL");
    ASSERT(mil.length > 100, @"MIL should be non-trivial");
    ASSERT([mil containsString:@"rms1"], @"should contain RMSNorm");
    ASSERT([mil containsString:@"conv"], @"should contain conv (linear)");
    ASSERT([mil containsString:@"matmul"], @"should contain matmul (attention)");
    ASSERT([mil containsString:@"softmax"], @"should contain softmax");
    // Multi-output: should return 6 fp16 outputs
    ASSERT([mil containsString:@"-> (wo_out, q_out, k_out, v_out, attn_out, rms1_out)"], @"should have 6 outputs");
    PASS();
}

static void test_fwd_attn_compile(void) {
    orion_ane_init();

    NSString *mil = orion_milgen_fwd_attn(0, &kStories110M);
    int d = kStories110M.d_model;
    int s = kStories110M.max_seq;

    // Build weight dict
    NSArray *paths = @[
        @"layer0/rms1.bin", @"layer0/wq.bin", @"layer0/wk.bin",
        @"layer0/wv.bin", @"layer0/wo.bin",
        [NSString stringWithFormat:@"masks/causal_%d.bin", s]
    ];
    NSArray *sizes = @[
        @(d),         // rms weight [1, d, 1, 1]
        @(d * d),     // wq [d, d, 1, 1]
        @(d * d),     // wk
        @(d * d),     // wv
        @(d * d),     // wo
        @(s * s),     // causal mask [1, 1, s, s]
    ];

    NSDictionary *wdict = make_random_wdict(paths, sizes);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, NULL);
    ASSERT(prog != NULL, @"fwdAttn should compile on ANE");

    orion_release_program(prog);
    PASS();
}

static void test_fwd_attn_eval(void) {
    orion_ane_init();

    NSString *mil = orion_milgen_fwd_attn(0, &kStories110M);
    int d = kStories110M.d_model;
    int s = kStories110M.max_seq;

    NSArray *paths = @[
        @"layer0/rms1.bin", @"layer0/wq.bin", @"layer0/wk.bin",
        @"layer0/wv.bin", @"layer0/wo.bin",
        [NSString stringWithFormat:@"masks/causal_%d.bin", s]
    ];
    NSArray *sizes = @[@(d), @(d*d), @(d*d), @(d*d), @(d*d), @(s*s)];

    NSDictionary *wdict = make_random_wdict(paths, sizes);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, NULL);
    ASSERT(prog != NULL, @"should compile");

    // Create input tensor [1, d, 1, s] fp16
    IOSurfaceRef input = orion_tensor_create(d, s);
    float *in_data = calloc(d * s, sizeof(float));
    for (int i = 0; i < d * s; i++) in_data[i] = 0.01f * sinf((float)i * 0.1f);
    orion_tensor_write_f32(input, in_data, d * s);
    free(in_data);

    // 6 fp16 output surfaces — each [1, d, 1, s]
    IOSurfaceRef out_oo = orion_tensor_create(d, s);
    IOSurfaceRef out_qf = orion_tensor_create(d, s);
    IOSurfaceRef out_kf = orion_tensor_create(d, s);
    IOSurfaceRef out_vf = orion_tensor_create(d, s);
    IOSurfaceRef out_af = orion_tensor_create(d, s);
    IOSurfaceRef out_xn = orion_tensor_create(d, s);

    IOSurfaceRef inputs[] = {input};
    IOSurfaceRef outputs[] = {out_oo, out_qf, out_kf, out_vf, out_af, out_xn};
    bool ok = orion_eval(prog, inputs, 1, outputs, 6);
    ASSERT(ok, @"eval should succeed");

    // Verify each output has finite values (fp16→fp32 conversion on read)
    float *buf = calloc(d * s, sizeof(float));
    for (int o = 0; o < 6; o++) {
        orion_tensor_read_f32(outputs[o], buf, d * s);
        int finite_count = 0;
        for (int i = 0; i < d * s; i++) {
            if (isfinite(buf[i])) finite_count++;
        }
        NSString *msg = [NSString stringWithFormat:@"output %d should be all finite", o];
        ASSERT(finite_count == d * s, msg);
    }

    free(buf);
    CFRelease(input);
    for (int i = 0; i < 6; i++) CFRelease(outputs[i]);
    orion_release_program(prog);
    PASS();
}

#pragma mark - T065: fwdFFN Tests

static void test_fwd_ffn_milgen(void) {
    NSString *mil = orion_milgen_fwd_ffn(0, &kStories110M);
    ASSERT(mil != nil, @"should generate MIL");
    ASSERT([mil containsString:@"sigmoid"], @"should contain sigmoid (for SiLU)");
    // Multi-output: 5 fp16 outputs
    ASSERT([mil containsString:@"-> (w2_out, w1_out, w3_out, gate, rms2_out)"], @"should have 5 outputs");
    PASS();
}

static void test_fwd_ffn_compile(void) {
    orion_ane_init();

    NSString *mil = orion_milgen_fwd_ffn(0, &kStories110M);
    int d = kStories110M.d_model;
    int h = kStories110M.hidden_dim;

    NSArray *paths = @[
        @"layer0/rms2.bin", @"layer0/w1.bin", @"layer0/w3.bin", @"layer0/w2.bin"
    ];
    NSArray *sizes = @[@(d), @(h*d), @(h*d), @(d*h)];

    NSDictionary *wdict = make_random_wdict(paths, sizes);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, NULL);
    ASSERT(prog != NULL, @"fwdFFN should compile on ANE");

    orion_release_program(prog);
    PASS();
}

static void test_fwd_ffn_eval(void) {
    orion_ane_init();

    NSString *mil = orion_milgen_fwd_ffn(0, &kStories110M);
    int d = kStories110M.d_model;
    int h = kStories110M.hidden_dim;
    int s = kStories110M.max_seq;

    NSArray *paths = @[
        @"layer0/rms2.bin", @"layer0/w1.bin", @"layer0/w3.bin", @"layer0/w2.bin"
    ];
    NSArray *sizes = @[@(d), @(h*d), @(h*d), @(d*h)];

    NSDictionary *wdict = make_random_wdict(paths, sizes);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "fwdFFN_eval_test");
    ASSERT(prog != NULL, @"should compile");

    // Create input tensor [1, d, 1, s] fp16
    IOSurfaceRef input = orion_tensor_create(d, s);
    float *in_data = calloc(d * s, sizeof(float));
    for (int i = 0; i < d * s; i++) in_data[i] = 0.01f * sinf((float)i * 0.1f);
    orion_tensor_write_f32(input, in_data, d * s);
    free(in_data);

    // ANE requires all multi-output IOSurfaces to be the same allocation size.
    // Use max(d, h) = h for all outputs.
    IOSurfaceRef out_w2  = orion_tensor_create(h, s);   // w2_out [d, s] (padded)
    IOSurfaceRef out_w1  = orion_tensor_create(h, s);   // w1_out [h, s]
    IOSurfaceRef out_w3  = orion_tensor_create(h, s);   // w3_out [h, s]
    IOSurfaceRef out_gt  = orion_tensor_create(h, s);   // gate [h, s]
    IOSurfaceRef out_xn  = orion_tensor_create(h, s);   // rms2_out [d, s] (padded)

    IOSurfaceRef inputs[] = {input};
    IOSurfaceRef outputs[] = {out_w2, out_w1, out_w3, out_gt, out_xn};
    bool ok = orion_eval(prog, inputs, 1, outputs, 5);
    ASSERT(ok, @"fwdFFN eval should succeed");

    // Verify outputs are finite (read logical sizes, not padded sizes)
    float *buf = calloc(h * s, sizeof(float));
    int logical_sizes[] = {d*s, h*s, h*s, h*s, d*s};
    for (int o = 0; o < 5; o++) {
        orion_tensor_read_f32(outputs[o], buf, logical_sizes[o]);
        int finite_count = 0;
        for (int i = 0; i < logical_sizes[o]; i++) {
            if (isfinite(buf[i])) finite_count++;
        }
        NSString *msg = [NSString stringWithFormat:@"output %d should be all finite", o];
        ASSERT(finite_count == logical_sizes[o], msg);
    }

    free(buf);
    CFRelease(input);
    for (int i = 0; i < 5; i++) CFRelease(outputs[i]);
    orion_release_program(prog);
    PASS();
}

#pragma mark - T066: ffnBwd Tests

static void test_ffn_bwd_milgen(void) {
    NSString *mil = orion_milgen_ffn_bwd(0, &kStories110M);
    ASSERT(mil != nil, @"should generate MIL");
    ASSERT([mil containsString:@"sigmoid"], @"should contain sigmoid (SiLU backward)");
    ASSERT([mil containsString:@"slice_by_index"], @"should slice concatenated input");
    PASS();
}

static void test_ffn_bwd_compile(void) {
    orion_ane_init();

    NSString *mil = orion_milgen_ffn_bwd(0, &kStories110M);
    int d = kStories110M.d_model;
    int h = kStories110M.hidden_dim;

    NSArray *paths = @[
        @"layer0/w2t.bin", @"layer0/w1t.bin", @"layer0/w3t.bin"
    ];
    NSArray *sizes = @[@(h*d), @(d*h), @(d*h)];

    NSDictionary *wdict = make_random_wdict(paths, sizes);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, NULL);
    ASSERT(prog != NULL, @"ffnBwd should compile on ANE");

    orion_release_program(prog);
    PASS();
}

#pragma mark - T067: sdpaBwd1 Tests

static void test_sdpa_bwd1_milgen(void) {
    NSString *mil = orion_milgen_sdpa_bwd1(0, &kStories110M);
    ASSERT(mil != nil, @"should generate MIL");
    ASSERT([mil containsString:@"matmul"], @"should contain matmul");
    ASSERT([mil containsString:@"softmax"], @"should recompute softmax");
    PASS();
}

static void test_sdpa_bwd1_compile(void) {
    orion_ane_init();

    NSString *mil = orion_milgen_sdpa_bwd1(0, &kStories110M);
    int d = kStories110M.d_model;
    int s = kStories110M.max_seq;

    NSArray *paths = @[
        @"layer0/wot.bin",
        [NSString stringWithFormat:@"masks/causal_%d.bin", s]
    ];
    NSArray *sizes = @[@(d*d), @(s*s)];

    NSDictionary *wdict = make_random_wdict(paths, sizes);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, NULL);
    ASSERT(prog != NULL, @"sdpaBwd1 should compile on ANE");

    orion_release_program(prog);
    PASS();
}

#pragma mark - T068: sdpaBwd2 Tests

static void test_sdpa_bwd2_milgen(void) {
    NSString *mil = orion_milgen_sdpa_bwd2(0, &kStories110M);
    ASSERT(mil != nil, @"should generate MIL");
    ASSERT([mil containsString:@"reduce_sum"], @"should reduce_sum (softmax backward)");
    PASS();
}

static void test_sdpa_bwd2_compile(void) {
    orion_ane_init();

    NSString *mil = orion_milgen_sdpa_bwd2(0, &kStories110M);
    // No weight blobs (weight-free kernel)
    NSDictionary *wdict = @{};
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, NULL);
    ASSERT(prog != NULL, @"sdpaBwd2 should compile on ANE (weight-free)");

    orion_release_program(prog);
    PASS();
}

#pragma mark - T069: qkvBwd Tests

static void test_qkv_bwd_milgen(void) {
    NSString *mil = orion_milgen_qkv_bwd(0, &kStories110M);
    ASSERT(mil != nil, @"should generate MIL");
    ASSERT([mil containsString:@"conv"], @"should contain conv (transpose projections)");
    PASS();
}

static void test_qkv_bwd_compile(void) {
    orion_ane_init();

    NSString *mil = orion_milgen_qkv_bwd(0, &kStories110M);
    int d = kStories110M.d_model;

    NSArray *paths = @[@"layer0/wqt.bin", @"layer0/wkt.bin", @"layer0/wvt.bin"];
    NSArray *sizes = @[@(d*d), @(d*d), @(d*d)];

    NSDictionary *wdict = make_random_wdict(paths, sizes);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, NULL);
    ASSERT(prog != NULL, @"qkvBwd should compile on ANE");

    orion_release_program(prog);
    PASS();
}

#pragma mark - T070-T071: Classifier + Softmax Tests

static void test_classifier_milgen(void) {
    NSString *mil = orion_milgen_classifier_fwd(768, 32000);
    ASSERT(mil != nil, @"should generate MIL");
    ASSERT([mil containsString:@"conv"], @"should contain conv (classifier)");
    PASS();
}

static void test_softmax_milgen(void) {
    NSString *mil = orion_milgen_vocab_softmax(32000, 256);
    ASSERT(mil != nil, @"should generate MIL");
    ASSERT([mil containsString:@"softmax"], @"should contain softmax");
    PASS();
}

// Note: Classifier compile test omitted — 32000-channel conv may exceed
// ANE compile limits and is an optional offload anyway.

#pragma mark - Main

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        NSLog(@"=== T064: fwdAttn Tests ===");
        test_fwd_attn_milgen();
        test_fwd_attn_compile();
        test_fwd_attn_eval();

        NSLog(@"\n=== T065: fwdFFN Tests ===");
        test_fwd_ffn_milgen();
        test_fwd_ffn_compile();
        test_fwd_ffn_eval();

        NSLog(@"\n=== T066: ffnBwd Tests ===");
        test_ffn_bwd_milgen();
        test_ffn_bwd_compile();

        NSLog(@"\n=== T067: sdpaBwd1 Tests ===");
        test_sdpa_bwd1_milgen();
        test_sdpa_bwd1_compile();

        NSLog(@"\n=== T068: sdpaBwd2 Tests ===");
        test_sdpa_bwd2_milgen();
        test_sdpa_bwd2_compile();

        NSLog(@"\n=== T069: qkvBwd Tests ===");
        test_qkv_bwd_milgen();
        test_qkv_bwd_compile();

        NSLog(@"\n=== T070-T071: Classifier + Softmax Tests ===");
        test_classifier_milgen();
        test_softmax_milgen();

        NSLog(@"\n=== Results: %d passed, %d failed (compiles: %d) ===",
              g_pass, g_fail, orion_compile_count());
        return g_fail > 0 ? 1 : 0;
    }
}
