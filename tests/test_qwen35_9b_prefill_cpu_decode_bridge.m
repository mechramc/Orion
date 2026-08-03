#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <math.h>
#include "compiler/frontends/qwen35_prefill.h"
#include "compiler/model_config.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#import "compiler/codegen.h"
#include "core/ane_runtime.h"
#include "model/weight_loader.h"
#include "kernels/inference/qwen_cpu_ops.h"

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

static IOSurfaceRef make_cpu_seq_input_surface(const float *x_seq, int seq_len, int bucket, int d_model) {
    IOSurfaceRef s = make_f32_surface(d_model * bucket, 0.0f);
    IOSurfaceLock(s, 0, NULL);
    float *ptr = (float *)IOSurfaceGetBaseAddress(s);
    for (int t = 0; t < seq_len; t++) {
        for (int c = 0; c < d_model; c++) {
            ptr[c * bucket + t] = x_seq[t * d_model + c];
        }
    }
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static void read_ane_surface_prefix(IOSurfaceRef s, int channels, int seq_len, int bucket, float *out_seq) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    const float *ptr = (const float *)IOSurfaceGetBaseAddress(s);
    for (int t = 0; t < seq_len; t++) {
        for (int c = 0; c < channels; c++) {
            out_seq[t * channels + c] = ptr[c * bucket + t];
        }
    }
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

static void add_blob(NSMutableDictionary *dict, NSString *mil_path, NSString *file_path) {
    NSData *data = [NSData dataWithContentsOfFile:file_path];
    if (data) dict[mil_path] = @{@"offset": @0, @"data": data};
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

static float *load_layer_exact(const char *blob_dir, int layer_idx, const char *name, int count) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/layer%d/%s", blob_dir, layer_idx, name);
    return orion_read_blob_f32_exact(path, count);
}

static double mean_abs_diff(const float *a, const float *b, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)a[i] - (double)b[i]);
    return total / (double)n;
}

static double max_abs_diff(const float *a, const float *b, int n) {
    double best = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > best) best = d;
    }
    return best;
}

static double abs_sum_vec(const float *x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)x[i]);
    return total;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <blob_dir>\n", argv[0]);
            return 2;
        }
        if (!orion_ane_init()) {
            fprintf(stderr, "FAIL: orion_ane_init failed\n");
            return 3;
        }

        const char *blob_dir = argv[1];
        NSString *blob_dir_ns = @(blob_dir);
        const int layer = 3;
        const int seq_len = 2;
        const int bucket = 32;
        const int d_model = kQwen35_9B.d_model;
        const int d_ff = kQwen35_9B.hidden_dim;
        const int n_head = kQwen35_9B.n_head;
        const int n_kv_head = kQwen35_9B.n_kv_head;
        const int head_dim = kQwen35_9B.head_dim;
        const int q_dim = n_head * head_dim;
        const int kv_dim = n_kv_head * head_dim;

        NSString *mil_q = compile_graph(orion_frontend_qwen35_prefill_q_proj(layer, bucket, &kQwen35_9B));
        NSString *mil_kv = compile_graph(orion_frontend_qwen35_prefill_kv_proj(layer, bucket, &kQwen35_9B));
        NSString *mil_ffn = compile_graph(orion_frontend_qwen35_prefill_ffn(layer, bucket, &kQwen35_9B));
        if (!mil_q || !mil_kv || !mil_ffn) {
            fprintf(stderr, "FAIL: compile_graph returned nil\n");
            return 4;
        }

        OrionProgram *prog_q = orion_compile_mil(mil_q.UTF8String, build_qproj_wdict(layer, blob_dir_ns), "qwen35_9b_bridge_q");
        OrionProgram *prog_kv = orion_compile_mil(mil_kv.UTF8String, build_kv_wdict(layer, blob_dir_ns), "qwen35_9b_bridge_kv");
        OrionProgram *prog_ffn = orion_compile_mil(mil_ffn.UTF8String, build_ffn_wdict(layer, blob_dir_ns), "qwen35_9b_bridge_ffn");
        if (!prog_q || !prog_kv || !prog_ffn) {
            fprintf(stderr, "FAIL: ANE program compile failed\n");
            if (prog_q) orion_release_program(prog_q);
            if (prog_kv) orion_release_program(prog_kv);
            if (prog_ffn) orion_release_program(prog_ffn);
            return 5;
        }

        float *x_seq = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        for (int t = 0; t < seq_len; t++) {
            for (int c = 0; c < d_model; c++) {
                x_seq[t * d_model + c] = sinf((float)(c * 0.001 + t * 0.01)) * 0.1f;
            }
        }

        float *input_ln = load_layer_exact(blob_dir, layer, "input_layernorm.bin", d_model);
        float *post_ln = load_layer_exact(blob_dir, layer, "post_attention_layernorm.bin", d_model);
        float *q_proj = load_layer_exact(blob_dir, layer, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
        float *k_proj = load_layer_exact(blob_dir, layer, "self_attn_k_proj.bin", kv_dim * d_model);
        float *v_proj = load_layer_exact(blob_dir, layer, "self_attn_v_proj.bin", kv_dim * d_model);
        float *o_proj = load_layer_exact(blob_dir, layer, "self_attn_o_proj.bin", d_model * q_dim);
        float *q_norm = load_layer_exact(blob_dir, layer, "self_attn_q_norm.bin", head_dim);
        float *k_norm = load_layer_exact(blob_dir, layer, "self_attn_k_norm.bin", head_dim);
        float *gate_proj = load_layer_exact(blob_dir, layer, "mlp_gate_proj.bin", d_ff * d_model);
        float *up_proj = load_layer_exact(blob_dir, layer, "mlp_up_proj.bin", d_ff * d_model);
        float *down_proj = load_layer_exact(blob_dir, layer, "mlp_down_proj.bin", d_model * d_ff);
        if (!input_ln || !post_ln || !q_proj || !k_proj || !v_proj || !o_proj || !q_norm || !k_norm ||
            !gate_proj || !up_proj || !down_proj) {
            fprintf(stderr, "FAIL: missing layer weights\n");
            return 6;
        }

        float *normed = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *cpu_attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *cpu_hidden_attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *cpu_normed_post = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *cpu_mlp = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *cpu_hidden_final = (float *)calloc((size_t)seq_len * d_model, sizeof(float));

        for (int t = 0; t < seq_len; t++) {
            orion_qwen_cpu_rmsnorm(x_seq + t * d_model, input_ln, d_model, 1e-6f, normed + t * d_model);
        }
        orion_qwen_cpu_full_attention_prefill_with_rope(normed, seq_len, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
                                                        d_model, n_head, n_kv_head, head_dim, 10000000.0f, 0.25f, cpu_attn);
        for (int i = 0; i < seq_len * d_model; i++) cpu_hidden_attn[i] = x_seq[i] + cpu_attn[i];
        for (int t = 0; t < seq_len; t++) {
            orion_qwen_cpu_rmsnorm(cpu_hidden_attn + t * d_model, post_ln, d_model, 1e-6f, cpu_normed_post + t * d_model);
            orion_qwen_cpu_swiglu_ffn(cpu_normed_post + t * d_model, gate_proj, up_proj, down_proj, d_model, d_ff,
                                      cpu_mlp + t * d_model);
        }
        for (int i = 0; i < seq_len * d_model; i++) cpu_hidden_final[i] = cpu_hidden_attn[i] + cpu_mlp[i];

        IOSurfaceRef ioIn = make_cpu_seq_input_surface(x_seq, seq_len, bucket, d_model);
        IOSurfaceRef ioQ = make_f32_surface((q_dim * 2) * bucket, 0.0f);
        IOSurfaceRef ioK = make_f32_surface(kv_dim * bucket, 0.0f);
        IOSurfaceRef ioV = make_f32_surface(kv_dim * bucket, 0.0f);
        IOSurfaceRef ioHidden = make_f32_surface(d_model * bucket, 0.0f);

        IOSurfaceRef insQ[] = {ioIn};
        IOSurfaceRef outsQ[] = {ioQ};
        IOSurfaceRef outsKV[] = {ioK, ioV};
        if (!orion_eval(prog_q, insQ, 1, outsQ, 1) || !orion_eval(prog_kv, insQ, 1, outsKV, 2)) {
            fprintf(stderr, "FAIL: ANE q/kv eval failed\n");
            return 7;
        }

        float *q_proj_seq = (float *)calloc((size_t)seq_len * (q_dim * 2), sizeof(float));
        float *k_proj_seq = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        float *v_proj_seq = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        float *bridge_attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *bridge_hidden_attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
        float *bridge_hidden_final = (float *)calloc((size_t)seq_len * d_model, sizeof(float));

        read_ane_surface_prefix(ioQ, q_dim * 2, seq_len, bucket, q_proj_seq);
        read_ane_surface_prefix(ioK, kv_dim, seq_len, bucket, k_proj_seq);
        read_ane_surface_prefix(ioV, kv_dim, seq_len, bucket, v_proj_seq);

        orion_qwen_cpu_full_attention_from_projections_with_rope(q_proj_seq, k_proj_seq, v_proj_seq, seq_len,
                                                                 o_proj, q_norm, k_norm,
                                                                 d_model, n_head, n_kv_head, head_dim,
                                                                 10000000.0f, 0.25f, bridge_attn);
        for (int i = 0; i < seq_len * d_model; i++) bridge_hidden_attn[i] = x_seq[i] + bridge_attn[i];

        IOSurfaceRef ioFfnIn = make_cpu_seq_input_surface(bridge_hidden_attn, seq_len, bucket, d_model);
        IOSurfaceRef insFFN[] = {ioFfnIn};
        IOSurfaceRef outsFFN[] = {ioHidden};
        if (!orion_eval(prog_ffn, insFFN, 1, outsFFN, 1)) {
            fprintf(stderr, "FAIL: ANE ffn eval failed\n");
            return 8;
        }
        read_ane_surface_prefix(ioHidden, d_model, seq_len, bucket, bridge_hidden_final);

        double mean_diff = mean_abs_diff(cpu_hidden_final, bridge_hidden_final, seq_len * d_model);
        double max_diff = max_abs_diff(cpu_hidden_final, bridge_hidden_final, seq_len * d_model);
        printf("qwen35_9b_prefill_cpu_decode_bridge:\n");
        printf("  seq_len=%d\n", seq_len);
        printf("  layer=%d\n", layer);
        printf("  mean_abs_diff=%.6f\n", mean_diff);
        printf("  max_abs_diff=%.6f\n", max_diff);
        printf("  bridge_hidden_abs_sum=%.6f\n", abs_sum_vec(bridge_hidden_final, seq_len * d_model));

        bool pass = isfinite(mean_diff) && isfinite(max_diff) && mean_diff < 0.05 && max_diff < 0.5;
        printf("  status=%s\n", pass ? "PASS" : "FAIL");

        CFRelease(ioIn);
        CFRelease(ioQ);
        CFRelease(ioK);
        CFRelease(ioV);
        CFRelease(ioHidden);
        CFRelease(ioFfnIn);
        orion_release_program(prog_q);
        orion_release_program(prog_kv);
        orion_release_program(prog_ffn);

        free(x_seq);
        free(input_ln); free(post_ln); free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
        free(gate_proj); free(up_proj); free(down_proj);
        free(normed); free(cpu_attn); free(cpu_hidden_attn); free(cpu_normed_post); free(cpu_mlp); free(cpu_hidden_final);
        free(q_proj_seq); free(k_proj_seq); free(v_proj_seq); free(bridge_attn); free(bridge_hidden_attn); free(bridge_hidden_final);

        return pass ? 0 : 1;
    }
}
