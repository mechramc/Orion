// tests/test_qwen35_9b_prefill_runtime_smoke.m
// Qwen3.5-9B ANE prefill runtime smoke: compile + eval q_proj / kv_proj / ffn.

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include "compiler/frontends/qwen35_prefill.h"
#include "compiler/model_config.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#import "compiler/codegen.h"
#include "core/ane_runtime.h"

static int g_passed = 0, g_failed = 0;

#define TEST(name) do { \
    printf("  %-40s ", #name); \
    if (test_##name(blob_dir)) { printf("PASS\n"); g_passed++; } \
    else { printf("FAIL\n"); g_failed++; } \
} while(0)

static const OrionModelConfig kQwen35_9B = {
    .n_layer = 32,
    .n_head = 16,
    .n_kv_head = 4,
    .d_model = 4096,
    .head_dim = 256,
    .hidden_dim = 12288,
    .vocab = 248320,
    .max_seq = 262144,
};

static IOSurfaceRef make_f32_surface(int count, float fill) {
    size_t bytes = (size_t)count * sizeof(float);
    IOSurfaceRef s = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
    IOSurfaceLock(s, 0, NULL);
    float *ptr = (float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) ptr[i] = fill;
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static IOSurfaceRef make_pattern_input_surface(int d_model, int seq) {
    int count = d_model * seq;
    IOSurfaceRef s = make_f32_surface(count, 0.0f);
    IOSurfaceLock(s, 0, NULL);
    float *ptr = (float *)IOSurfaceGetBaseAddress(s);
    for (int c = 0; c < d_model; c++) {
        for (int t = 0; t < seq; t++) {
            ptr[c * seq + t] = sinf((float)(c * 0.001 + t * 0.01)) * 0.1f;
        }
    }
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static bool surface_all_finite(IOSurfaceRef s, int count, float *max_abs_out) {
    bool ok = true;
    float max_abs = 0.0f;
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    const float *ptr = (const float *)IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) {
        float v = ptr[i];
        if (!isfinite(v)) ok = false;
        float a = fabsf(v);
        if (a > max_abs) max_abs = a;
    }
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
    if (max_abs_out) *max_abs_out = max_abs;
    return ok;
}

static void add_blob(NSMutableDictionary *dict, NSString *mil_path, NSString *file_path) {
    NSData *data = [NSData dataWithContentsOfFile:file_path];
    if (data) {
        dict[mil_path] = @{@"offset": @0, @"data": data};
    }
}

static NSDictionary *build_qproj_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/input_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"input_layernorm.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/self_attn_q_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"self_attn_q_proj.bin"]);
    return dict;
}

static NSDictionary *build_kv_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/input_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"input_layernorm.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/self_attn_k_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"self_attn_k_proj.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/self_attn_v_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"self_attn_v_proj.bin"]);
    return dict;
}

static NSDictionary *build_ffn_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/post_attention_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"post_attention_layernorm.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/mlp_gate_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"mlp_gate_proj.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/mlp_up_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"mlp_up_proj.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/mlp_down_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"mlp_down_proj.bin"]);
    return dict;
}

static NSString *compile_graph(OrionGraph *g) {
    if (!g) return nil;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) {
        NSLog(@"graph validate failed: %s", vr.message);
        orion_graph_free(g);
        return nil;
    }
    orion_pipeline_optimize(g);
    NSString *mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

static bool test_q_proj(NSString *blob_dir) {
    int layer = 3;
    int seq = 32;
    int d = kQwen35_9B.d_model;
    NSString *mil = compile_graph(orion_frontend_qwen35_prefill_q_proj(layer, seq, &kQwen35_9B));
    if (!mil || mil.length == 0) return false;
    NSDictionary *wdict = build_qproj_wdict(layer, blob_dir);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "qwen35_9b_qproj");
    if (!prog) return false;

    IOSurfaceRef ioIn = make_pattern_input_surface(d, seq);
    IOSurfaceRef ioOut = make_f32_surface(d * 2 * seq, 0.0f);
    IOSurfaceRef ins[] = {ioIn};
    IOSurfaceRef outs[] = {ioOut};
    bool ok = orion_eval(prog, ins, 1, outs, 1);
    float max_abs = 0.0f;
    ok = ok && surface_all_finite(ioOut, d * 2 * seq, &max_abs) && max_abs > 1e-4f;

    CFRelease(ioIn);
    CFRelease(ioOut);
    orion_release_program(prog);
    return ok;
}

static bool test_kv_proj(NSString *blob_dir) {
    int layer = 3;
    int seq = 32;
    int d = kQwen35_9B.d_model;
    int kv = kQwen35_9B.n_kv_head * kQwen35_9B.head_dim;
    NSString *mil = compile_graph(orion_frontend_qwen35_prefill_kv_proj(layer, seq, &kQwen35_9B));
    if (!mil || mil.length == 0) return false;
    NSDictionary *wdict = build_kv_wdict(layer, blob_dir);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "qwen35_9b_kvproj");
    if (!prog) return false;

    IOSurfaceRef ioIn = make_pattern_input_surface(d, seq);
    IOSurfaceRef ioK = make_f32_surface(kv * seq, 0.0f);
    IOSurfaceRef ioV = make_f32_surface(kv * seq, 0.0f);
    IOSurfaceRef ins[] = {ioIn};
    IOSurfaceRef outs[] = {ioK, ioV};
    bool ok = orion_eval(prog, ins, 1, outs, 2);
    float max_k = 0.0f, max_v = 0.0f;
    ok = ok &&
         surface_all_finite(ioK, kv * seq, &max_k) &&
         surface_all_finite(ioV, kv * seq, &max_v) &&
         max_k > 1e-4f && max_v > 1e-4f;

    CFRelease(ioIn);
    CFRelease(ioK);
    CFRelease(ioV);
    orion_release_program(prog);
    return ok;
}

static bool test_ffn(NSString *blob_dir) {
    int layer = 3;
    int seq = 32;
    int d = kQwen35_9B.d_model;
    NSString *mil = compile_graph(orion_frontend_qwen35_prefill_ffn(layer, seq, &kQwen35_9B));
    if (!mil || mil.length == 0) return false;
    NSDictionary *wdict = build_ffn_wdict(layer, blob_dir);
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "qwen35_9b_ffn");
    if (!prog) return false;

    IOSurfaceRef ioIn = make_pattern_input_surface(d, seq);
    IOSurfaceRef ioOut = make_f32_surface(d * seq, 0.0f);
    IOSurfaceRef ins[] = {ioIn};
    IOSurfaceRef outs[] = {ioOut};
    bool ok = orion_eval(prog, ins, 1, outs, 1);
    float max_abs = 0.0f;
    ok = ok && surface_all_finite(ioOut, d * seq, &max_abs) && max_abs > 1e-4f;

    CFRelease(ioIn);
    CFRelease(ioOut);
    orion_release_program(prog);
    return ok;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <blob_dir>\n", argv[0]);
            return 2;
        }
        if (!orion_ane_init()) {
            fprintf(stderr, "orion_ane_init failed\n");
            return 3;
        }
        NSString *blob_dir = @(argv[1]);
        printf("test_qwen35_9b_prefill_runtime_smoke:\n");
        TEST(q_proj);
        TEST(kv_proj);
        TEST(ffn);
        printf("\n%d/%d passed\n", g_passed, g_passed + g_failed);
        return g_failed > 0 ? 1 : 0;
    }
}
