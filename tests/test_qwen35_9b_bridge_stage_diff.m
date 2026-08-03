#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "compiler/builder.h"
#include "compiler/frontends/qwen35_prefill.h"
#include "compiler/model_config.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#import "compiler/codegen.h"
#include "core/ane_runtime.h"
#import "../model/weight_loader.h"
#import "../tokenizer/gpt2_bpe.h"
#import "../kernels/inference/qwen_cpu_ops.h"

static NSDictionary* load_json(NSString* path) {
    NSData* data = [NSData dataWithContentsOfFile:path];
    if (!data) return nil;
    return [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
}

static int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

static float *load_exact(const char *blob_dir, int layer_idx, const char *suffix, int count) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/layer%d/%s", blob_dir, layer_idx, suffix);
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

typedef struct {
    long long lhs_size;
    long long rhs_size;
    long long first_diff_offset;
    int equal;
} OrionByteCompareResult;

static OrionByteCompareResult compare_nsdata(NSData *lhs, NSData *rhs) {
    OrionByteCompareResult result = {
        .lhs_size = lhs ? (long long)lhs.length : -1,
        .rhs_size = rhs ? (long long)rhs.length : -1,
        .first_diff_offset = -1,
        .equal = 0,
    };
    if (!lhs || !rhs) return result;
    if ([lhs isEqualToData:rhs]) {
        result.equal = 1;
        return result;
    }

    const unsigned char *lhs_bytes = (const unsigned char *)lhs.bytes;
    const unsigned char *rhs_bytes = (const unsigned char *)rhs.bytes;
    NSUInteger min_len = lhs.length < rhs.length ? lhs.length : rhs.length;
    for (NSUInteger i = 0; i < min_len; i++) {
        if (lhs_bytes[i] != rhs_bytes[i]) {
            result.first_diff_offset = (long long)i;
            return result;
        }
    }
    result.first_diff_offset = (long long)min_len;
    return result;
}

static NSData *load_blob_payload_data(NSString *path) {
    NSData *blob = [NSData dataWithContentsOfFile:path];
    if (!blob || blob.length < 128) return nil;
    return [blob subdataWithRange:NSMakeRange(128, blob.length - 128)];
}

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

static void read_ane_surface_prefix_token_major(IOSurfaceRef s, int channels, int seq_len, float *out_seq) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    const float *ptr = (const float *)IOSurfaceGetBaseAddress(s);
    for (int t = 0; t < seq_len; t++) {
        for (int c = 0; c < channels; c++) {
            out_seq[t * channels + c] = ptr[t * channels + c];
        }
    }
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

static double mean_abs_diff_q_half(const float *a, const float *b, int seq_len, int q_dim, int half_idx) {
    double total = 0.0;
    for (int s = 0; s < seq_len; s++) {
        const float *row_a = a + s * (q_dim * 2) + half_idx * q_dim;
        const float *row_b = b + s * (q_dim * 2) + half_idx * q_dim;
        for (int i = 0; i < q_dim; i++) total += fabs((double)row_a[i] - (double)row_b[i]);
    }
    return total / (double)(seq_len * q_dim);
}

static double max_abs_diff_q_half(const float *a, const float *b, int seq_len, int q_dim, int half_idx) {
    double best = 0.0;
    for (int s = 0; s < seq_len; s++) {
        const float *row_a = a + s * (q_dim * 2) + half_idx * q_dim;
        const float *row_b = b + s * (q_dim * 2) + half_idx * q_dim;
        for (int i = 0; i < q_dim; i++) {
            double d = fabs((double)row_a[i] - (double)row_b[i]);
            if (d > best) best = d;
        }
    }
    return best;
}

static double mean_abs_diff_q_half_swapped(const float *cpu_q, const float *ane_q, int seq_len, int q_dim) {
    double total = 0.0;
    for (int s = 0; s < seq_len; s++) {
        const float *cpu_query = cpu_q + s * (q_dim * 2);
        const float *cpu_gate = cpu_query + q_dim;
        const float *ane_gate = ane_q + s * (q_dim * 2);
        const float *ane_query = ane_gate + q_dim;
        for (int i = 0; i < q_dim; i++) total += fabs((double)cpu_query[i] - (double)ane_query[i]);
        for (int i = 0; i < q_dim; i++) total += fabs((double)cpu_gate[i] - (double)ane_gate[i]);
    }
    return total / (double)(seq_len * q_dim * 2);
}

static double max_abs_diff_q_half_swapped(const float *cpu_q, const float *ane_q, int seq_len, int q_dim) {
    double best = 0.0;
    for (int s = 0; s < seq_len; s++) {
        const float *cpu_query = cpu_q + s * (q_dim * 2);
        const float *cpu_gate = cpu_query + q_dim;
        const float *ane_gate = ane_q + s * (q_dim * 2);
        const float *ane_query = ane_gate + q_dim;
        for (int i = 0; i < q_dim; i++) {
            double d = fabs((double)cpu_query[i] - (double)ane_query[i]);
            if (d > best) best = d;
        }
        for (int i = 0; i < q_dim; i++) {
            double d = fabs((double)cpu_gate[i] - (double)ane_gate[i]);
            if (d > best) best = d;
        }
    }
    return best;
}

static double mean_abs_diff_kv_swapped(const float *cpu_k,
                                       const float *cpu_v,
                                       const float *ane_k,
                                       const float *ane_v,
                                       int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)cpu_k[i] - (double)ane_v[i]);
    for (int i = 0; i < n; i++) total += fabs((double)cpu_v[i] - (double)ane_k[i]);
    return total / (double)(n * 2);
}

static double max_abs_diff_kv_swapped(const float *cpu_k,
                                      const float *cpu_v,
                                      const float *ane_k,
                                      const float *ane_v,
                                      int n) {
    double best = 0.0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)cpu_k[i] - (double)ane_v[i]);
        if (d > best) best = d;
    }
    for (int i = 0; i < n; i++) {
        double d = fabs((double)cpu_v[i] - (double)ane_k[i]);
        if (d > best) best = d;
    }
    return best;
}

static inline float fp16_roundtrip(float x);

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

static NSDictionary *build_vproj_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/input_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"input_layernorm.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/self_attn_v_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"self_attn_v_proj.bin"]);
    return dict;
}

static NSDictionary *build_input_rms_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/input_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"input_layernorm.bin"]);
    return dict;
}

static NSDictionary *build_vproj_linear_only_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
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

static NSDictionary *build_ffn_rms_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/post_attention_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"post_attention_layernorm.bin"]);
    return dict;
}

static NSDictionary *build_ffn_gateup_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/post_attention_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"post_attention_layernorm.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/mlp_gate_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"mlp_gate_proj.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/mlp_up_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"mlp_up_proj.bin"]);
    return dict;
}

static NSDictionary *build_ffn_uponly_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/post_attention_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"post_attention_layernorm.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/mlp_up_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"mlp_up_proj.bin"]);
    return dict;
}

static NSDictionary *build_ffn_siluonly_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/post_attention_layernorm.bin", layer],
             [prefix stringByAppendingPathComponent:@"post_attention_layernorm.bin"]);
    add_blob(dict, [NSString stringWithFormat:@"@model_path/layer%d/mlp_gate_proj.bin", layer],
             [prefix stringByAppendingPathComponent:@"mlp_gate_proj.bin"]);
    return dict;
}

static NSDictionary *build_ffn_down_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
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

static int use_fp32_rrms_powchain_mode(void) {
    const char *mode = getenv("ORION_RMSNORM_RRMS_MODE");
    return mode && (strcmp(mode, "nr1") == 0 || strcmp(mode, "fp32") == 0);
}

static int qwen35_input_rmsnorm_local(OrionGraph *g, int input, int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int s = bucket;
    char path[256];
    int ln_shape[4] = {1, d, 1, 1};
    snprintf(path, sizeof(path), "@model_path/layer%d/input_layernorm.bin", layer);
    int rms_w = orion_gb_const_weight(g, "input_ln_w", ORION_DTYPE_FP16, ln_shape, path, 64);
    return orion_gb_rmsnorm(g, input, rms_w, 1e-6f, "input_rms", d, s);
}

static int qwen35_post_attn_rmsnorm_local(OrionGraph *g, int input, int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int s = bucket;
    char path[256];
    int ln_shape[4] = {1, d, 1, 1};
    snprintf(path, sizeof(path), "@model_path/layer%d/post_attention_layernorm.bin", layer);
    int rms_w = orion_gb_const_weight(g, "post_attn_ln_w", ORION_DTYPE_FP16, ln_shape, path, 64);
    return orion_gb_rmsnorm(g, input, rms_w, 1e-6f, "post_attn_rms", d, s);
}

static OrionGraph *build_ffn_rms_graph(int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int s = bucket;
    int shape[4] = {1, d, 1, s};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", shape);
    int rms = qwen35_post_attn_rmsnorm_local(g, x16, layer, bucket, cfg);
    int rms32 = orion_gb_cast(g, rms, ORION_DTYPE_FP32, "rms32", shape);
    orion_gb_output(g, rms32, "rms");
    return g;
}

static OrionGraph *build_ffn_gateup_graph(int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int h = cfg->hidden_dim;
    int s = bucket;
    int in_shape[4] = {1, d, 1, s};
    int hid_shape[4] = {1, h, 1, s};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", in_shape);
    int rms = qwen35_post_attn_rmsnorm_local(g, x16, layer, bucket, cfg);

    char gate_w[256], up_w[256];
    snprintf(gate_w, sizeof(gate_w), "@model_path/layer%d/mlp_gate_proj.bin", layer);
    snprintf(up_w, sizeof(up_w), "@model_path/layer%d/mlp_up_proj.bin", layer);

    int gate = orion_gb_linear(g, rms, "gate_proj", d, h, s, gate_w, NULL);
    int up = orion_gb_linear(g, rms, "up_proj", d, h, s, up_w, NULL);
    int silu = orion_gb_silu(g, gate, "gate_silu", h, s);
    int hidden = orion_gb_mul(g, silu, up, "ffn_hidden");

    int gate32 = orion_gb_cast(g, gate, ORION_DTYPE_FP32, "gate32", hid_shape);
    int up32 = orion_gb_cast(g, up, ORION_DTYPE_FP32, "up32", hid_shape);
    int silu32 = orion_gb_cast(g, silu, ORION_DTYPE_FP32, "silu32", hid_shape);
    int hidden32 = orion_gb_cast(g, hidden, ORION_DTYPE_FP32, "hidden32", hid_shape);
    orion_gb_output(g, gate32, "gate_proj");
    orion_gb_output(g, up32, "up_proj");
    orion_gb_output(g, silu32, "silu_gate");
    orion_gb_output(g, hidden32, "hidden_ff");
    return g;
}

static OrionGraph *build_ffn_uponly_graph(int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int h = cfg->hidden_dim;
    int s = bucket;
    int in_shape[4] = {1, d, 1, s};
    int hid_shape[4] = {1, h, 1, s};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", in_shape);
    int rms = qwen35_post_attn_rmsnorm_local(g, x16, layer, bucket, cfg);

    char up_w[256];
    snprintf(up_w, sizeof(up_w), "@model_path/layer%d/mlp_up_proj.bin", layer);
    int up = orion_gb_linear(g, rms, "up_proj", d, h, s, up_w, NULL);
    int up32 = orion_gb_cast(g, up, ORION_DTYPE_FP32, "up32", hid_shape);
    orion_gb_output(g, up32, "up_proj");
    return g;
}

static OrionGraph *build_ffn_siluonly_graph(int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int h = cfg->hidden_dim;
    int s = bucket;
    int in_shape[4] = {1, d, 1, s};
    int hid_shape[4] = {1, h, 1, s};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", in_shape);
    int rms = qwen35_post_attn_rmsnorm_local(g, x16, layer, bucket, cfg);

    char gate_w[256];
    snprintf(gate_w, sizeof(gate_w), "@model_path/layer%d/mlp_gate_proj.bin", layer);
    int gate = orion_gb_linear(g, rms, "gate_proj", d, h, s, gate_w, NULL);
    int silu = orion_gb_silu(g, gate, "gate_silu", h, s);
    int silu32 = orion_gb_cast(g, silu, ORION_DTYPE_FP32, "silu32", hid_shape);
    orion_gb_output(g, silu32, "silu_gate");
    return g;
}

static OrionGraph *build_ffn_hiddenonly_graph(int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int h = cfg->hidden_dim;
    int s = bucket;
    int in_shape[4] = {1, d, 1, s};
    int hid_shape[4] = {1, h, 1, s};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", in_shape);
    int rms = qwen35_post_attn_rmsnorm_local(g, x16, layer, bucket, cfg);

    char gate_w[256], up_w[256];
    snprintf(gate_w, sizeof(gate_w), "@model_path/layer%d/mlp_gate_proj.bin", layer);
    snprintf(up_w, sizeof(up_w), "@model_path/layer%d/mlp_up_proj.bin", layer);
    int gate = orion_gb_linear(g, rms, "gate_proj", d, h, s, gate_w, NULL);
    int up = orion_gb_linear(g, rms, "up_proj", d, h, s, up_w, NULL);
    int silu = orion_gb_silu(g, gate, "gate_silu", h, s);
    int hidden = orion_gb_mul(g, silu, up, "ffn_hidden");
    int hidden32 = orion_gb_cast(g, hidden, ORION_DTYPE_FP32, "hidden32", hid_shape);
    orion_gb_output(g, hidden32, "hidden_ff");
    return g;
}

static OrionGraph *build_ffn_down_graph(int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int h = cfg->hidden_dim;
    int s = bucket;
    int in_shape[4] = {1, h, 1, s};
    int out_shape[4] = {1, d, 1, s};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", in_shape);

    char down_w[256];
    snprintf(down_w, sizeof(down_w), "@model_path/layer%d/mlp_down_proj.bin", layer);
    int down = orion_gb_linear(g, x16, "down_proj", h, d, s, down_w, NULL);
    int down32 = orion_gb_cast(g, down, ORION_DTYPE_FP32, "down32", out_shape);
    orion_gb_output(g, down32, "down_proj");
    return g;
}

static OrionGraph *build_vproj_graph(int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int kv_heads = cfg->n_kv_head > 0 ? cfg->n_kv_head : cfg->n_head;
    int kv = kv_heads * cfg->head_dim;
    int s = bucket;
    int in_shape[4] = {1, d, 1, s};
    int out_shape[4] = {1, kv, 1, s};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", in_shape);
    int rms = qwen35_input_rmsnorm_local(g, x16, layer, bucket, cfg);

    char v_w[256];
    snprintf(v_w, sizeof(v_w), "@model_path/layer%d/self_attn_v_proj.bin", layer);
    int v_proj = orion_gb_linear(g, rms, "v_proj", d, kv, s, v_w, NULL);
    int v_proj32 = orion_gb_cast(g, v_proj, ORION_DTYPE_FP32, "v_proj32", out_shape);
    orion_gb_output(g, v_proj32, "v_proj");
    return g;
}

static OrionGraph *build_input_rms_graph(int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int s = bucket;
    int shape[4] = {1, d, 1, s};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", shape);
    int rms = qwen35_input_rmsnorm_local(g, x16, layer, bucket, cfg);
    int rms32 = orion_gb_cast(g, rms, ORION_DTYPE_FP32, "rms32", shape);
    orion_gb_output(g, rms32, "input_rms");
    return g;
}

typedef enum {
    ORION_INPUT_RMS_STAGE_MS = 0,
    ORION_INPUT_RMS_STAGE_RRMS = 1,
    ORION_INPUT_RMS_STAGE_XR = 2,
} OrionInputRmsStage;

static OrionGraph *build_input_rms_stage_graph(int layer, int bucket, const OrionModelConfig *cfg, OrionInputRmsStage stage) {
    int d = cfg->d_model;
    int s = bucket;
    int shape[4] = {1, d, 1, s};
    int scalar_shape[4] = {1, 1, 1, s};
    int ax_shape[4] = {1, 0, 0, 0};
    int axis_val = 1;
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", shape);
    (void)layer;

    int sq = orion_gb_mul(g, x16, x16, "input_rms_trace_sq");
    int axes = orion_gb_const_int32(g, "input_rms_trace_ax", ax_shape, &axis_val, 1);
    int sum = orion_gb_reduce_sum(g, sq, axes, true, "input_rms_trace_sum", scalar_shape);
    int ms = -1;
    int rrms = -1;
    int xr = -1;
    if (use_fp32_rrms_powchain_mode()) {
        int invd = orion_gb_const_scalar(g, "input_rms_trace_invd", ORION_DTYPE_FP16, 1.0f / (float)d);
        ms = orion_gb_mul(g, sum, invd, "input_rms_trace_ms");
        int eps = orion_gb_const_scalar(g, "input_rms_trace_eps", ORION_DTYPE_FP16, 1e-6f);
        int mse = orion_gb_add(g, ms, eps, "input_rms_trace_mse");
        int nhalf = orion_gb_const_scalar(g, "input_rms_trace_nhalf", ORION_DTYPE_FP16, -0.5f);
        int rrms0 = orion_gb_pow(g, mse, nhalf, "input_rms_trace_rrms");

        int rrms_sq = orion_gb_mul(g, rrms0, rrms0, "input_rms_trace_rrms_sq");
        int nr_term = orion_gb_mul(g, mse, rrms_sq, "input_rms_trace_nr_term");
        int half_nr = orion_gb_const_scalar(g, "input_rms_trace_half_nr", ORION_DTYPE_FP16, 0.5f);
        int nr_half = orion_gb_mul(g, nr_term, half_nr, "input_rms_trace_nr_half");
        int threehalves = orion_gb_const_scalar(g, "input_rms_trace_threehalves", ORION_DTYPE_FP16, 1.5f);
        int nr_corr = orion_gb_sub(g, threehalves, nr_half, "input_rms_trace_nr_corr");
        rrms = orion_gb_mul(g, rrms0, nr_corr, "input_rms_trace_rrms_refined");
        xr = orion_gb_mul(g, x16, rrms, "input_rms_trace_xr");
    } else {
        int invd = orion_gb_const_scalar(g, "input_rms_trace_invd", ORION_DTYPE_FP16, 1.0f / (float)d);
        ms = orion_gb_mul(g, sum, invd, "input_rms_trace_ms");
        int eps = orion_gb_const_scalar(g, "input_rms_trace_eps", ORION_DTYPE_FP16, 1e-6f);
        int mse = orion_gb_add(g, ms, eps, "input_rms_trace_mse");
        int nhalf = orion_gb_const_scalar(g, "input_rms_trace_nhalf", ORION_DTYPE_FP16, -0.5f);
        rrms = orion_gb_pow(g, mse, nhalf, "input_rms_trace_rrms");
        xr = orion_gb_mul(g, x16, rrms, "input_rms_trace_xr");
    }
    switch (stage) {
        case ORION_INPUT_RMS_STAGE_MS: {
            int ms32 = orion_gb_cast(g, ms, ORION_DTYPE_FP32, "input_rms_trace_ms_out32", scalar_shape);
            orion_gb_output(g, ms32, "input_rms_ms");
            break;
        }
        case ORION_INPUT_RMS_STAGE_RRMS: {
            int rrms32 = orion_gb_cast(g, rrms, ORION_DTYPE_FP32, "input_rms_trace_rrms_out32", scalar_shape);
            orion_gb_output(g, rrms32, "input_rms_rrms");
            break;
        }
        case ORION_INPUT_RMS_STAGE_XR: {
            int xr32 = orion_gb_cast(g, xr, ORION_DTYPE_FP32, "input_rms_trace_xr_out32", shape);
            orion_gb_output(g, xr32, "input_rms_xr");
            break;
        }
    }
    return g;
}

static OrionGraph *build_vproj_linear_only_graph(int layer, int bucket, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int kv_heads = cfg->n_kv_head > 0 ? cfg->n_kv_head : cfg->n_head;
    int kv = kv_heads * cfg->head_dim;
    int s = bucket;
    int in_shape[4] = {1, d, 1, s};
    int out_shape[4] = {1, kv, 1, s};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, in_shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", in_shape);

    char v_w[256];
    snprintf(v_w, sizeof(v_w), "@model_path/layer%d/self_attn_v_proj.bin", layer);
    int v_proj = orion_gb_linear(g, x16, "v_proj_linear_only", d, kv, s, v_w, NULL);
    int v_proj32 = orion_gb_cast(g, v_proj, ORION_DTYPE_FP32, "v_proj32", out_shape);
    orion_gb_output(g, v_proj32, "v_proj_linear_only");
    return g;
}

static OrionGraph *build_silu_micro_graph(int dim, int bucket) {
    int shape[4] = {1, dim, 1, bucket};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", shape);
    int silu = orion_gb_silu(g, x16, "micro_silu", dim, bucket);
    int silu32 = orion_gb_cast(g, silu, ORION_DTYPE_FP32, "silu32", shape);
    orion_gb_output(g, silu32, "silu");
    return g;
}

static OrionGraph *build_sigmoid_micro_graph(int dim, int bucket) {
    int shape[4] = {1, dim, 1, bucket};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", shape);
    int sig = orion_gb_sigmoid(g, x16, "micro_sigmoid");
    int sig32 = orion_gb_cast(g, sig, ORION_DTYPE_FP32, "sig32", shape);
    orion_gb_output(g, sig32, "sigmoid");
    return g;
}

static int build_sigmoid_tanh_node(OrionGraph *g, int input, const char *prefix) {
    char buf[ORION_MAX_NAME];

    snprintf(buf, sizeof(buf), "%s_half", prefix);
    int half = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, 0.5f);
    snprintf(buf, sizeof(buf), "%s_hx", prefix);
    int hx = orion_gb_mul(g, input, half, buf);

    snprintf(buf, sizeof(buf), "%s_th", prefix);
    int th = orion_gb_tanh(g, hx, buf);

    snprintf(buf, sizeof(buf), "%s_one", prefix);
    int one = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, 1.0f);
    snprintf(buf, sizeof(buf), "%s_onep", prefix);
    int onep = orion_gb_add(g, th, one, buf);

    snprintf(buf, sizeof(buf), "%s_out", prefix);
    return orion_gb_mul(g, onep, half, buf);
}

static OrionGraph *build_sigmoid_tanh_micro_graph(int dim, int bucket) {
    int shape[4] = {1, dim, 1, bucket};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", shape);
    int sig = build_sigmoid_tanh_node(g, x16, "micro_sigmoid_tanh");
    int sig32 = orion_gb_cast(g, sig, ORION_DTYPE_FP32, "sig32", shape);
    orion_gb_output(g, sig32, "sigmoid_tanh");
    return g;
}

static OrionGraph *build_silu_tanh_micro_graph(int dim, int bucket) {
    int shape[4] = {1, dim, 1, bucket};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", shape);
    int sig = build_sigmoid_tanh_node(g, x16, "micro_silu_tanh_sig");
    int silu = orion_gb_mul(g, x16, sig, "micro_silu_tanh");
    int silu32 = orion_gb_cast(g, silu, ORION_DTYPE_FP32, "silu32", shape);
    orion_gb_output(g, silu32, "silu_tanh");
    return g;
}

static OrionGraph *build_mul_micro_graph(int dim, int bucket) {
    int shape[4] = {1, dim, 1, bucket};
    OrionGraph *g = orion_graph_create();
    int a = orion_gb_input(g, "a", ORION_DTYPE_FP32, shape);
    int b = orion_gb_input(g, "b", ORION_DTYPE_FP32, shape);
    int a16 = orion_gb_cast(g, a, ORION_DTYPE_FP16, "a16", shape);
    int b16 = orion_gb_cast(g, b, ORION_DTYPE_FP16, "b16", shape);
    int mul = orion_gb_mul(g, a16, b16, "micro_mul");
    int mul32 = orion_gb_cast(g, mul, ORION_DTYPE_FP32, "mul32", shape);
    orion_gb_output(g, mul32, "mul");
    return g;
}

static OrionGraph *build_rrms_pow_micro_graph(int bucket) {
    int shape[4] = {1, 1, 1, bucket};
    OrionGraph *g = orion_graph_create();
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP32, shape);
    int x16 = orion_gb_cast(g, x, ORION_DTYPE_FP16, "x16", shape);
    int nhalf = orion_gb_const_scalar(g, "rrms_pow_nhalf", ORION_DTYPE_FP16, -0.5f);
    int out = orion_gb_pow(g, x16, nhalf, "rrms_pow");
    int out32 = orion_gb_cast(g, out, ORION_DTYPE_FP32, "rrms_pow32", shape);
    orion_gb_output(g, out32, "rrms_pow");
    return g;
}

typedef struct {
    OrionProgram *prog_q;
    OrionProgram *prog_kv;
    OrionProgram *prog_v_only;
    OrionProgram *prog_input_rms_only;
    OrionProgram *prog_input_rms_ms;
    OrionProgram *prog_input_rms_rrms;
    OrionProgram *prog_input_rms_xr;
    OrionProgram *prog_v_linear_only;
    OrionProgram *prog_ffn_rms;
    OrionProgram *prog_ffn_gateup;
    OrionProgram *prog_ffn_up_only;
    OrionProgram *prog_ffn_silu_only;
    OrionProgram *prog_ffn_hidden_only;
    OrionProgram *prog_ffn_down;
    OrionProgram *prog_ffn;
    OrionProgram *prog_sigmoid_micro;
    OrionProgram *prog_sigmoid_tanh_micro;
    OrionProgram *prog_silu_micro;
    OrionProgram *prog_silu_tanh_micro;
    OrionProgram *prog_mul_micro;
    OrionProgram *prog_rrms_pow_micro;
    int layer_idx;
    int bucket;
    int d_model;
    int q_dim;
    int kv_dim;
} OrionQwen35AneBridge;

static void bridge_release(OrionQwen35AneBridge *bridge) {
    if (!bridge) return;
    if (bridge->prog_q) orion_release_program(bridge->prog_q);
    if (bridge->prog_kv) orion_release_program(bridge->prog_kv);
    if (bridge->prog_v_only) orion_release_program(bridge->prog_v_only);
    if (bridge->prog_input_rms_only) orion_release_program(bridge->prog_input_rms_only);
    if (bridge->prog_input_rms_ms) orion_release_program(bridge->prog_input_rms_ms);
    if (bridge->prog_input_rms_rrms) orion_release_program(bridge->prog_input_rms_rrms);
    if (bridge->prog_input_rms_xr) orion_release_program(bridge->prog_input_rms_xr);
    if (bridge->prog_v_linear_only) orion_release_program(bridge->prog_v_linear_only);
    if (bridge->prog_ffn_rms) orion_release_program(bridge->prog_ffn_rms);
    if (bridge->prog_ffn_gateup) orion_release_program(bridge->prog_ffn_gateup);
    if (bridge->prog_ffn_up_only) orion_release_program(bridge->prog_ffn_up_only);
    if (bridge->prog_ffn_silu_only) orion_release_program(bridge->prog_ffn_silu_only);
    if (bridge->prog_ffn_hidden_only) orion_release_program(bridge->prog_ffn_hidden_only);
    if (bridge->prog_ffn_down) orion_release_program(bridge->prog_ffn_down);
    if (bridge->prog_ffn) orion_release_program(bridge->prog_ffn);
    if (bridge->prog_sigmoid_micro) orion_release_program(bridge->prog_sigmoid_micro);
    if (bridge->prog_sigmoid_tanh_micro) orion_release_program(bridge->prog_sigmoid_tanh_micro);
    if (bridge->prog_silu_micro) orion_release_program(bridge->prog_silu_micro);
    if (bridge->prog_silu_tanh_micro) orion_release_program(bridge->prog_silu_tanh_micro);
    if (bridge->prog_mul_micro) orion_release_program(bridge->prog_mul_micro);
    if (bridge->prog_rrms_pow_micro) orion_release_program(bridge->prog_rrms_pow_micro);
    memset(bridge, 0, sizeof(*bridge));
}

static int bridge_init(OrionQwen35AneBridge *bridge, NSString *blobDir, int layer, int bucket, OrionQwen35Manifest *manifest) {
    memset(bridge, 0, sizeof(*bridge));
    OrionModelConfig cfg = {
        .n_layer = manifest->n_layer,
        .n_head = manifest->n_head,
        .n_kv_head = manifest->n_kv_head,
        .d_model = manifest->d_model,
        .head_dim = manifest->head_dim,
        .hidden_dim = manifest->d_ff,
        .vocab = manifest->vocab,
        .max_seq = manifest->max_seq,
    };
    NSString *mil_q = compile_graph(orion_frontend_qwen35_prefill_q_proj(layer, bucket, &cfg));
    NSString *mil_kv = compile_graph(orion_frontend_qwen35_prefill_kv_proj(layer, bucket, &cfg));
    NSString *mil_v_only = compile_graph(build_vproj_graph(layer, bucket, &cfg));
    NSString *mil_input_rms_only = compile_graph(build_input_rms_graph(layer, bucket, &cfg));
    NSString *mil_input_rms_ms = compile_graph(build_input_rms_stage_graph(layer, bucket, &cfg, ORION_INPUT_RMS_STAGE_MS));
    NSString *mil_input_rms_rrms = compile_graph(build_input_rms_stage_graph(layer, bucket, &cfg, ORION_INPUT_RMS_STAGE_RRMS));
    NSString *mil_input_rms_xr = compile_graph(build_input_rms_stage_graph(layer, bucket, &cfg, ORION_INPUT_RMS_STAGE_XR));
    NSString *mil_v_linear_only = compile_graph(build_vproj_linear_only_graph(layer, bucket, &cfg));
    NSString *mil_ffn_rms = compile_graph(build_ffn_rms_graph(layer, bucket, &cfg));
    NSString *mil_ffn_gateup = compile_graph(build_ffn_gateup_graph(layer, bucket, &cfg));
    NSString *mil_ffn_up_only = compile_graph(build_ffn_uponly_graph(layer, bucket, &cfg));
    NSString *mil_ffn_silu_only = compile_graph(build_ffn_siluonly_graph(layer, bucket, &cfg));
    NSString *mil_ffn_hidden_only = compile_graph(build_ffn_hiddenonly_graph(layer, bucket, &cfg));
    NSString *mil_ffn_down = compile_graph(build_ffn_down_graph(layer, bucket, &cfg));
    NSString *mil_ffn = compile_graph(orion_frontend_qwen35_prefill_ffn(layer, bucket, &cfg));
    NSString *mil_sigmoid_micro = compile_graph(build_sigmoid_micro_graph(cfg.hidden_dim, bucket));
    NSString *mil_sigmoid_tanh_micro = compile_graph(build_sigmoid_tanh_micro_graph(cfg.hidden_dim, bucket));
    NSString *mil_silu_micro = compile_graph(build_silu_micro_graph(cfg.hidden_dim, bucket));
    NSString *mil_silu_tanh_micro = compile_graph(build_silu_tanh_micro_graph(cfg.hidden_dim, bucket));
    NSString *mil_mul_micro = compile_graph(build_mul_micro_graph(cfg.hidden_dim, bucket));
    NSString *mil_rrms_pow_micro = compile_graph(build_rrms_pow_micro_graph(bucket));
    if (!mil_q || !mil_kv || !mil_v_only || !mil_input_rms_only || !mil_input_rms_ms || !mil_input_rms_rrms ||
        !mil_input_rms_xr || !mil_v_linear_only ||
        !mil_ffn_rms || !mil_ffn_gateup || !mil_ffn_up_only ||
        !mil_ffn_silu_only || !mil_ffn_hidden_only || !mil_ffn_down || !mil_ffn ||
        !mil_sigmoid_micro || !mil_sigmoid_tanh_micro || !mil_silu_micro || !mil_silu_tanh_micro ||
        !mil_mul_micro || !mil_rrms_pow_micro) return 0;

    bridge->prog_q = orion_compile_mil(mil_q.UTF8String, build_qproj_wdict(layer, blobDir), "qwen35_9b_stage_q");
    bridge->prog_kv = orion_compile_mil(mil_kv.UTF8String, build_kv_wdict(layer, blobDir), "qwen35_9b_stage_kv");
    bridge->prog_v_only = orion_compile_mil(mil_v_only.UTF8String, build_vproj_wdict(layer, blobDir), "qwen35_9b_stage_v_only");
    bridge->prog_input_rms_only = orion_compile_mil(mil_input_rms_only.UTF8String, build_input_rms_wdict(layer, blobDir), "qwen35_9b_stage_input_rms_only");
    bridge->prog_input_rms_ms = orion_compile_mil(mil_input_rms_ms.UTF8String, build_input_rms_wdict(layer, blobDir), "qwen35_9b_stage_input_rms_ms");
    bridge->prog_input_rms_rrms = orion_compile_mil(mil_input_rms_rrms.UTF8String, build_input_rms_wdict(layer, blobDir), "qwen35_9b_stage_input_rms_rrms");
    bridge->prog_input_rms_xr = orion_compile_mil(mil_input_rms_xr.UTF8String, build_input_rms_wdict(layer, blobDir), "qwen35_9b_stage_input_rms_xr");
    bridge->prog_v_linear_only = orion_compile_mil(mil_v_linear_only.UTF8String, build_vproj_linear_only_wdict(layer, blobDir), "qwen35_9b_stage_v_linear_only");
    bridge->prog_ffn_rms = orion_compile_mil(mil_ffn_rms.UTF8String, build_ffn_rms_wdict(layer, blobDir), "qwen35_9b_stage_ffn_rms");
    bridge->prog_ffn_gateup = orion_compile_mil(mil_ffn_gateup.UTF8String, build_ffn_gateup_wdict(layer, blobDir), "qwen35_9b_stage_ffn_gateup");
    bridge->prog_ffn_up_only = orion_compile_mil(mil_ffn_up_only.UTF8String, build_ffn_uponly_wdict(layer, blobDir), "qwen35_9b_stage_ffn_up_only");
    bridge->prog_ffn_silu_only = orion_compile_mil(mil_ffn_silu_only.UTF8String, build_ffn_siluonly_wdict(layer, blobDir), "qwen35_9b_stage_ffn_silu_only");
    bridge->prog_ffn_hidden_only = orion_compile_mil(mil_ffn_hidden_only.UTF8String, build_ffn_gateup_wdict(layer, blobDir), "qwen35_9b_stage_ffn_hidden_only");
    bridge->prog_ffn_down = orion_compile_mil(mil_ffn_down.UTF8String, build_ffn_down_wdict(layer, blobDir), "qwen35_9b_stage_ffn_down");
    bridge->prog_ffn = orion_compile_mil(mil_ffn.UTF8String, build_ffn_wdict(layer, blobDir), "qwen35_9b_stage_ffn");
    bridge->prog_sigmoid_micro = orion_compile_mil(mil_sigmoid_micro.UTF8String, nil, "qwen35_9b_stage_sigmoid_micro");
    bridge->prog_sigmoid_tanh_micro = orion_compile_mil(mil_sigmoid_tanh_micro.UTF8String, nil, "qwen35_9b_stage_sigmoid_tanh_micro");
    bridge->prog_silu_micro = orion_compile_mil(mil_silu_micro.UTF8String, nil, "qwen35_9b_stage_silu_micro");
    bridge->prog_silu_tanh_micro = orion_compile_mil(mil_silu_tanh_micro.UTF8String, nil, "qwen35_9b_stage_silu_tanh_micro");
    bridge->prog_mul_micro = orion_compile_mil(mil_mul_micro.UTF8String, nil, "qwen35_9b_stage_mul_micro");
    bridge->prog_rrms_pow_micro = orion_compile_mil(mil_rrms_pow_micro.UTF8String, nil, "qwen35_9b_stage_rrms_pow_micro");
    if (!bridge->prog_q || !bridge->prog_kv || !bridge->prog_v_only ||
        !bridge->prog_input_rms_only || !bridge->prog_input_rms_ms || !bridge->prog_input_rms_rrms ||
        !bridge->prog_input_rms_xr || !bridge->prog_v_linear_only || !bridge->prog_ffn_rms ||
        !bridge->prog_ffn_gateup || !bridge->prog_ffn_up_only || !bridge->prog_ffn_silu_only ||
        !bridge->prog_ffn_hidden_only ||
        !bridge->prog_ffn_down || !bridge->prog_ffn || !bridge->prog_sigmoid_micro ||
        !bridge->prog_sigmoid_tanh_micro || !bridge->prog_silu_micro || !bridge->prog_silu_tanh_micro ||
        !bridge->prog_mul_micro || !bridge->prog_rrms_pow_micro) {
        bridge_release(bridge);
        return 0;
    }

    bridge->layer_idx = layer;
    bridge->bucket = bucket;
    bridge->d_model = manifest->d_model;
    bridge->q_dim = manifest->n_head * manifest->head_dim;
    bridge->kv_dim = manifest->n_kv_head * manifest->head_dim;
    return 1;
}

static int load_embeddings(const char *blob_dir,
                           OrionQwen35Manifest *manifest,
                           const int *token_ids,
                           int seq_len,
                           float *hidden_out) {
    const int d_model = manifest->d_model;
    char embed_path[2048];
    snprintf(embed_path, sizeof(embed_path), "%s/model/embed_tokens.bin", blob_dir);
    for (int s = 0; s < seq_len; s++) {
        if (!orion_read_blob_row_f32(embed_path, token_ids[s], d_model, hidden_out + s * d_model)) return 0;
    }
    return 1;
}

static int apply_cpu_ffn(const float *hidden_attn,
                         int seq_len,
                         int d_model,
                         int d_ff,
                         const float *post_ln,
                         const float *gate_proj,
                         const float *up_proj,
                         const float *down_proj,
                         float *hidden_final_out) {
    float *scratch = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *mlp_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    if (!scratch || !mlp_out) {
        free(scratch);
        free(mlp_out);
        return 0;
    }
    for (int s = 0; s < seq_len; s++) {
        orion_qwen_cpu_rmsnorm(hidden_attn + s * d_model, post_ln, d_model, 1e-6f, scratch + s * d_model);
        orion_qwen_cpu_swiglu_ffn(scratch + s * d_model, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out + s * d_model);
    }
    for (int i = 0; i < seq_len * d_model; i++) hidden_final_out[i] = hidden_attn[i] + mlp_out[i];
    free(scratch);
    free(mlp_out);
    return 1;
}

static int apply_cpu_layer(const char *blob_dir,
                           OrionQwen35Manifest *manifest,
                           int layer_idx,
                           const float *hidden_in,
                           int seq_len,
                           float *hidden_out) {
    const int d_model = manifest->d_model;
    const int d_ff = manifest->d_ff;
    const int n_head = manifest->n_head;
    const int n_kv_head = manifest->n_kv_head;
    const int head_dim = manifest->head_dim;
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_kv_head * head_dim;

    float *input_ln = load_exact(blob_dir, layer_idx, "input_layernorm.bin", d_model);
    float *post_ln = load_exact(blob_dir, layer_idx, "post_attention_layernorm.bin", d_model);
    float *gate_proj = load_exact(blob_dir, layer_idx, "mlp_gate_proj.bin", d_ff * d_model);
    float *up_proj = load_exact(blob_dir, layer_idx, "mlp_up_proj.bin", d_ff * d_model);
    float *down_proj = load_exact(blob_dir, layer_idx, "mlp_down_proj.bin", d_model * d_ff);
    if (!input_ln || !post_ln || !gate_proj || !up_proj || !down_proj) {
        free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
        return 0;
    }

    float *normed = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *attn_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *hidden_attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    if (!normed || !attn_out || !hidden_attn) {
        free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
        free(normed); free(attn_out); free(hidden_attn);
        return 0;
    }

    for (int s = 0; s < seq_len; s++) {
        orion_qwen_cpu_rmsnorm(hidden_in + s * d_model, input_ln, d_model, 1e-6f, normed + s * d_model);
    }

    char full_q_path[2048];
    snprintf(full_q_path, sizeof(full_q_path), "%s/layer%d/self_attn_q_proj.bin", blob_dir, layer_idx);
    if (file_exists(full_q_path)) {
        float *q_proj = load_exact(blob_dir, layer_idx, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
        float *k_proj = load_exact(blob_dir, layer_idx, "self_attn_k_proj.bin", kv_dim * d_model);
        float *v_proj = load_exact(blob_dir, layer_idx, "self_attn_v_proj.bin", kv_dim * d_model);
        float *o_proj = load_exact(blob_dir, layer_idx, "self_attn_o_proj.bin", d_model * q_dim);
        float *q_norm = load_exact(blob_dir, layer_idx, "self_attn_q_norm.bin", head_dim);
        float *k_norm = load_exact(blob_dir, layer_idx, "self_attn_k_norm.bin", head_dim);
        if (!q_proj || !k_proj || !v_proj || !o_proj || !q_norm || !k_norm) {
            free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
            free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
            free(normed); free(attn_out); free(hidden_attn);
            return 0;
        }
        orion_qwen_cpu_full_attention_prefill_with_rope(
            normed, seq_len, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn_out
        );
        free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
    } else {
        char path_qkv[2048], path_out[2048], path_dt[2048], path_norm[2048], path_conv[2048];
        snprintf(path_qkv, sizeof(path_qkv), "%s/layer%d/linear_attn_in_proj_qkv.bin", blob_dir, layer_idx);
        snprintf(path_out, sizeof(path_out), "%s/layer%d/linear_attn_out_proj.bin", blob_dir, layer_idx);
        snprintf(path_dt, sizeof(path_dt), "%s/layer%d/linear_attn_dt_bias.bin", blob_dir, layer_idx);
        snprintf(path_norm, sizeof(path_norm), "%s/layer%d/linear_attn_norm.bin", blob_dir, layer_idx);
        snprintf(path_conv, sizeof(path_conv), "%s/layer%d/linear_attn_conv1d.bin", blob_dir, layer_idx);

        int qkv_rows = orion_blob_element_count(path_qkv) / d_model;
        int value_dim = orion_blob_element_count(path_out) / d_model;
        int num_v_heads = orion_blob_element_count(path_dt);
        int head_v_dim = orion_blob_element_count(path_norm);
        int key_dim = (qkv_rows - value_dim) / 2;
        int num_k_heads = num_v_heads;
        int head_k_dim = key_dim / num_k_heads;
        int conv_kernel = orion_blob_element_count(path_conv) / qkv_rows;

        float *in_proj_qkv = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_qkv.bin", qkv_rows * d_model);
        float *in_proj_z = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_z.bin", value_dim * d_model);
        float *in_proj_a = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_a.bin", num_v_heads * d_model);
        float *in_proj_b = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_b.bin", num_v_heads * d_model);
        float *conv1d = load_exact(blob_dir, layer_idx, "linear_attn_conv1d.bin", qkv_rows * conv_kernel);
        float *dt_bias = load_exact(blob_dir, layer_idx, "linear_attn_dt_bias.bin", num_v_heads);
        float *a_log = load_exact(blob_dir, layer_idx, "linear_attn_a_log.bin", num_v_heads);
        float *norm_weight = load_exact(blob_dir, layer_idx, "linear_attn_norm.bin", head_v_dim);
        float *out_proj = load_exact(blob_dir, layer_idx, "linear_attn_out_proj.bin", d_model * value_dim);
        if (!in_proj_qkv || !in_proj_z || !in_proj_a || !in_proj_b || !conv1d ||
            !dt_bias || !a_log || !norm_weight || !out_proj) {
            free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
            free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
            free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
            free(normed); free(attn_out); free(hidden_attn);
            return 0;
        }
        orion_qwen_cpu_linear_attention_recurrent_prefill(
            normed, seq_len, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
            conv1d, dt_bias, a_log, norm_weight, out_proj,
            d_model, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel,
            attn_out
        );
        free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
        free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
    }

    for (int i = 0; i < seq_len * d_model; i++) hidden_attn[i] = hidden_in[i] + attn_out[i];
    int ok = apply_cpu_ffn(hidden_attn, seq_len, d_model, d_ff, post_ln, gate_proj, up_proj, down_proj, hidden_out);

    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
    free(normed); free(attn_out); free(hidden_attn);
    return ok;
}

static void cpu_linear_batch(const float *x_seq,
                             int seq_len,
                             const float *weight,
                             int in_dim,
                             int out_dim,
                             float *out_seq) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x_seq, in_dim, weight, in_dim,
                0.0f, out_seq, out_dim);
}

static void cpu_linear_batch_fp16_emulated(const float *x_seq,
                                           int seq_len,
                                           const float *weight,
                                           int in_dim,
                                           int out_dim,
                                           float *out_seq) {
    for (int s = 0; s < seq_len; s++) {
        const float *x = x_seq + (size_t)s * in_dim;
        float *out = out_seq + (size_t)s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w = weight + (size_t)o * in_dim;
            float acc = 0.0f;
            for (int i = 0; i < in_dim; i++) {
                float x16 = fp16_roundtrip(x[i]);
                float w16 = fp16_roundtrip(w[i]);
                float prod = fp16_roundtrip(x16 * w16);
                acc = fp16_roundtrip(acc + prod);
            }
            out[o] = acc;
        }
    }
}

static void cpu_linear_batch_fp16in_fp32acc_fp16out(const float *x_seq,
                                                    int seq_len,
                                                    const float *weight,
                                                    int in_dim,
                                                    int out_dim,
                                                    float *out_seq) {
    for (int s = 0; s < seq_len; s++) {
        const float *x = x_seq + (size_t)s * in_dim;
        float *out = out_seq + (size_t)s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w = weight + (size_t)o * in_dim;
            float acc = 0.0f;
            for (int i = 0; i < in_dim; i++) {
                float x16 = fp16_roundtrip(x[i]);
                float w16 = fp16_roundtrip(w[i]);
                acc += x16 * w16;
            }
            out[o] = fp16_roundtrip(acc);
        }
    }
}

static int selected_token_logits(const char *blob_dir,
                                 const char *lm_head_name,
                                 const float *hidden,
                                 int d_model,
                                 const int *token_ids,
                                 int token_count,
                                 float *out_logits) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/model/%s", blob_dir, lm_head_name);
    float *row = (float *)malloc((size_t)d_model * sizeof(float));
    if (!row) return 0;

    for (int tok_idx = 0; tok_idx < token_count; tok_idx++) {
        int tok = token_ids[tok_idx];
        if (tok < 0 || !orion_read_blob_row_f32(path, tok, d_model, row)) {
            free(row);
            return 0;
        }
        float dot = 0.0f;
        for (int i = 0; i < d_model; i++) dot += hidden[i] * row[i];
        out_logits[tok_idx] = dot;
    }

    free(row);
    return 1;
}

#define ORION_TRACE_DIM_MAX 16
#define ORION_TRACE_ATTR_TOPK 4

typedef struct {
    int channel;
    double abs_contrib;
    double contrib;
    double delta_input;
    double weight;
} OrionTraceChannelImpact;

static int compare_trace_channel_impact_desc(const void *lhs, const void *rhs) {
    const OrionTraceChannelImpact *a = (const OrionTraceChannelImpact *)lhs;
    const OrionTraceChannelImpact *b = (const OrionTraceChannelImpact *)rhs;
    if (a->abs_contrib > b->abs_contrib) return -1;
    if (a->abs_contrib < b->abs_contrib) return 1;
    if (a->channel < b->channel) return -1;
    if (a->channel > b->channel) return 1;
    return 0;
}

static int parse_dim_list(const char *csv, int *out_dims, int max_dims, int dim_limit) {
    if (!csv || !out_dims || max_dims <= 0) return 0;
    char *copy = strdup(csv);
    if (!copy) return 0;
    int count = 0;
    char *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok && count < max_dims; tok = strtok_r(NULL, ",", &save)) {
        while (*tok && isspace((unsigned char)*tok)) tok++;
        if (!*tok) continue;
        char *end = NULL;
        long dim = strtol(tok, &end, 10);
        if (end == tok) continue;
        while (*end && isspace((unsigned char)*end)) end++;
        if (*end != '\0') continue;
        if (dim < 0 || dim >= dim_limit) continue;
        int duplicate = 0;
        for (int i = 0; i < count; i++) {
            if (out_dims[i] == (int)dim) {
                duplicate = 1;
                break;
            }
        }
        if (duplicate) continue;
        out_dims[count++] = (int)dim;
    }
    free(copy);
    return count;
}

static int use_cpu_v_proj_override(void) {
    const char *source = getenv("ORION_V_PROJ_SOURCE");
    return source && strcmp(source, "cpu") == 0;
}

static int use_cpu_q_proj_override(void) {
    const char *source = getenv("ORION_Q_PROJ_SOURCE");
    return source && strcmp(source, "cpu") == 0;
}

static int use_cpu_q_query_override(void) {
    const char *source = getenv("ORION_Q_QUERY_SOURCE");
    return source && strcmp(source, "cpu") == 0;
}

static int use_cpu_q_gate_override(void) {
    const char *source = getenv("ORION_Q_GATE_SOURCE");
    return source && strcmp(source, "cpu") == 0;
}

static int load_v_proj_cpu_channel_override(int *out_channels, int max_channels, int q_dim) {
    const char *csv = getenv("ORION_V_PROJ_CPU_CHANNELS");
    if (!csv || !*csv) return 0;
    return parse_dim_list(csv, out_channels, max_channels, q_dim);
}

static const char *q_proj_source_label(int q_proj_uses_cpu) {
    return q_proj_uses_cpu ? "cpu_linear" : "ane_linear";
}

static const char *q_query_source_label(int q_query_uses_cpu) {
    return q_query_uses_cpu ? "cpu_linear" : "ane_linear";
}

static const char *q_gate_source_label(int q_gate_uses_cpu) {
    return q_gate_uses_cpu ? "cpu_linear" : "ane_linear";
}

static const char *v_proj_source_label(int v_proj_uses_cpu, int partial_channel_count) {
    if (v_proj_uses_cpu) return "cpu_linear";
    if (partial_channel_count > 0) return "ane_linear_cpu_channels";
    return "ane_linear";
}

static void apply_v_proj_cpu_channel_overrides(float *dst_v_proj_seq,
                                               const float *cpu_v_proj_seq,
                                               int seq_len,
                                               int kv_dim,
                                               int head_dim,
                                               int q_per_kv,
                                               const int *channels,
                                               int channel_count) {
    if (!dst_v_proj_seq || !cpu_v_proj_seq || !channels || channel_count <= 0 || seq_len <= 0 ||
        kv_dim <= 0 || head_dim <= 0 || q_per_kv <= 0) {
        return;
    }
    for (int i = 0; i < channel_count; i++) {
        int channel = channels[i];
        int head = channel / head_dim;
        int offset = channel % head_dim;
        int kv_head = head / q_per_kv;
        int kv_channel = kv_head * head_dim + offset;
        if (kv_channel < 0 || kv_channel >= kv_dim) continue;
        for (int s = 0; s < seq_len; s++) {
            dst_v_proj_seq[(size_t)s * kv_dim + kv_channel] =
                cpu_v_proj_seq[(size_t)s * kv_dim + kv_channel];
        }
    }
}

static void print_trace_stage_rows(const char *stage,
                                   const float *cpu_seq,
                                   const float *hybrid_seq,
                                   int seq_len,
                                   int d_model,
                                   const int *trace_dims,
                                   int trace_dim_count) {
    if (!stage || !cpu_seq || !hybrid_seq || !trace_dims || trace_dim_count <= 0 || seq_len <= 0) return;
    const float *cpu_last = cpu_seq + (seq_len - 1) * d_model;
    const float *hybrid_last = hybrid_seq + (seq_len - 1) * d_model;
    for (int i = 0; i < trace_dim_count; i++) {
        int dim = trace_dims[i];
        float cpu_val = cpu_last[dim];
        float hybrid_val = hybrid_last[dim];
        printf("trace_stage_dim stage=%s dim=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               stage, dim, cpu_val, hybrid_val, fabsf(cpu_val - hybrid_val));
    }
}

static void print_trace_attn_attr_rows(const float *cpu_in_seq,
                                       const float *hybrid_in_seq,
                                       const float *o_proj,
                                       int seq_len,
                                       int q_dim,
                                       int head_dim,
                                       const int *trace_dims,
                                       int trace_dim_count) {
    if (!cpu_in_seq || !hybrid_in_seq || !o_proj || !trace_dims || trace_dim_count <= 0 || seq_len <= 0) return;
    const float *cpu_last = cpu_in_seq + (seq_len - 1) * q_dim;
    const float *hybrid_last = hybrid_in_seq + (seq_len - 1) * q_dim;
    OrionTraceChannelImpact top[ORION_TRACE_ATTR_TOPK];
    for (int dim_idx = 0; dim_idx < trace_dim_count; dim_idx++) {
        int out_dim = trace_dims[dim_idx];
        int top_count = 0;
        const float *weight_row = o_proj + (size_t)out_dim * q_dim;
        for (int channel = 0; channel < q_dim; channel++) {
            double delta_input = (double)hybrid_last[channel] - (double)cpu_last[channel];
            double weight = (double)weight_row[channel];
            double contrib = delta_input * weight;
            OrionTraceChannelImpact cand = {
                .channel = channel,
                .abs_contrib = fabs(contrib),
                .contrib = contrib,
                .delta_input = delta_input,
                .weight = weight,
            };
            if (top_count < ORION_TRACE_ATTR_TOPK) {
                top[top_count++] = cand;
                continue;
            }
            int worst_idx = 0;
            for (int i = 1; i < top_count; i++) {
                if (top[i].abs_contrib < top[worst_idx].abs_contrib) worst_idx = i;
            }
            if (cand.abs_contrib > top[worst_idx].abs_contrib) top[worst_idx] = cand;
        }
        qsort(top, (size_t)top_count, sizeof(top[0]), compare_trace_channel_impact_desc);
        for (int rank = 0; rank < top_count; rank++) {
            int channel = top[rank].channel;
            int head = (head_dim > 0) ? (channel / head_dim) : -1;
            int offset = (head_dim > 0) ? (channel % head_dim) : channel;
            printf("trace_attn_attr dim=%d rank=%d channel=%d head=%d offset=%d delta_input=%.6f weight=%.6f contrib=%.6f\n",
                   out_dim, rank + 1, channel, head, offset,
                   top[rank].delta_input, top[rank].weight, top[rank].contrib);
        }
    }
}

static void print_trace_attn_channel_rows(const float *cpu_q_proj_seq,
                                          const float *hybrid_q_proj_seq,
                                          const float *cpu_v_proj_seq,
                                          const float *cpu_v_rms_only_proj_seq,
                                          const float *hybrid_v_proj_seq,
                                          const float *hybrid_v_single_proj_seq,
                                          const float *hybrid_v_linear_only_proj_seq,
                                          const float *cpu_q_normed_seq,
                                          const float *hybrid_q_normed_seq,
                                          const float *cpu_gate_sigmoid_seq,
                                          const float *hybrid_gate_sigmoid_seq,
                                          const float *cpu_q_rope_seq,
                                          const float *hybrid_q_rope_seq,
                                          const float *cpu_context_seq,
                                          const float *hybrid_context_seq,
                                          const float *cpu_gated_context_seq,
                                          const float *hybrid_gated_context_seq,
                                          int seq_len,
                                          int q_dim,
                                          int kv_dim,
                                          int head_dim,
                                          int q_per_kv,
                                          const int *trace_channels,
                                          int trace_channel_count) {
    if (!cpu_q_proj_seq || !hybrid_q_proj_seq || !cpu_v_proj_seq || !cpu_v_rms_only_proj_seq ||
        !hybrid_v_proj_seq ||
        !hybrid_v_single_proj_seq || !hybrid_v_linear_only_proj_seq ||
        !cpu_q_normed_seq || !hybrid_q_normed_seq || !cpu_gate_sigmoid_seq || !hybrid_gate_sigmoid_seq ||
        !cpu_q_rope_seq || !hybrid_q_rope_seq || !cpu_context_seq || !hybrid_context_seq ||
        !cpu_gated_context_seq || !hybrid_gated_context_seq || !trace_channels ||
        trace_channel_count <= 0 || seq_len <= 0 || head_dim <= 0 || q_per_kv <= 0) {
        return;
    }

    const float *cpu_q_last = cpu_q_proj_seq + (seq_len - 1) * (q_dim * 2);
    const float *hybrid_q_last = hybrid_q_proj_seq + (seq_len - 1) * (q_dim * 2);
    const float *cpu_v_last = cpu_v_proj_seq + (seq_len - 1) * kv_dim;
    const float *cpu_v_rms_only_last = cpu_v_rms_only_proj_seq + (seq_len - 1) * kv_dim;
    const float *hybrid_v_last = hybrid_v_proj_seq + (seq_len - 1) * kv_dim;
    const float *hybrid_v_single_last = hybrid_v_single_proj_seq + (seq_len - 1) * kv_dim;
    const float *hybrid_v_linear_only_last = hybrid_v_linear_only_proj_seq + (seq_len - 1) * kv_dim;
    const float *cpu_q_norm_last = cpu_q_normed_seq + (seq_len - 1) * q_dim;
    const float *hybrid_q_norm_last = hybrid_q_normed_seq + (seq_len - 1) * q_dim;
    const float *cpu_gate_sigmoid_last = cpu_gate_sigmoid_seq + (seq_len - 1) * q_dim;
    const float *hybrid_gate_sigmoid_last = hybrid_gate_sigmoid_seq + (seq_len - 1) * q_dim;
    const float *cpu_q_rope_last = cpu_q_rope_seq + (seq_len - 1) * q_dim;
    const float *hybrid_q_rope_last = hybrid_q_rope_seq + (seq_len - 1) * q_dim;
    const float *cpu_context_last = cpu_context_seq + (seq_len - 1) * q_dim;
    const float *hybrid_context_last = hybrid_context_seq + (seq_len - 1) * q_dim;
    const float *cpu_gated_context_last = cpu_gated_context_seq + (seq_len - 1) * q_dim;
    const float *hybrid_gated_context_last = hybrid_gated_context_seq + (seq_len - 1) * q_dim;

    for (int i = 0; i < trace_channel_count; i++) {
        int channel = trace_channels[i];
        int head = channel / head_dim;
        int offset = channel % head_dim;
        int kv_head = head / q_per_kv;
        int kv_channel = kv_head * head_dim + offset;
        if (kv_channel < 0 || kv_channel >= kv_dim) continue;

        float cpu_query = cpu_q_last[channel];
        float hybrid_query = hybrid_q_last[channel];
        float cpu_gate = cpu_q_last[q_dim + channel];
        float hybrid_gate = hybrid_q_last[q_dim + channel];
        float cpu_q_norm = cpu_q_norm_last[channel];
        float hybrid_q_norm = hybrid_q_norm_last[channel];
        float cpu_gate_sig = cpu_gate_sigmoid_last[channel];
        float hybrid_gate_sig = hybrid_gate_sigmoid_last[channel];
        float cpu_q_rope = cpu_q_rope_last[channel];
        float hybrid_q_rope = hybrid_q_rope_last[channel];
        float cpu_v = cpu_v_last[kv_channel];
        float hybrid_v = hybrid_v_last[kv_channel];
        float hybrid_v_single = hybrid_v_single_last[kv_channel];
        float cpu_context = cpu_context_last[channel];
        float hybrid_context = hybrid_context_last[channel];
        float cpu_gated_context = cpu_gated_context_last[channel];
        float hybrid_gated_context = hybrid_gated_context_last[channel];

        printf("trace_attn_channel stage=q_query_raw channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_query, hybrid_query, fabsf(cpu_query - hybrid_query));
        printf("trace_attn_channel stage=q_gate_raw channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_gate, hybrid_gate, fabsf(cpu_gate - hybrid_gate));
        printf("trace_attn_channel stage=q_normed channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_q_norm, hybrid_q_norm, fabsf(cpu_q_norm - hybrid_q_norm));
        printf("trace_attn_channel stage=gate_sigmoid channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_gate_sig, hybrid_gate_sig, fabsf(cpu_gate_sig - hybrid_gate_sig));
        printf("trace_attn_channel stage=q_rope channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_q_rope, hybrid_q_rope, fabsf(cpu_q_rope - hybrid_q_rope));
        printf("trace_attn_channel stage=v_proj_mapped channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_v, hybrid_v, fabsf(cpu_v - hybrid_v));
        printf("trace_attn_channel stage=v_proj_rms_only_mapped channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_v, cpu_v_rms_only_last[kv_channel],
               fabsf(cpu_v - cpu_v_rms_only_last[kv_channel]));
        printf("trace_attn_channel stage=v_proj_single_mapped channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_v, hybrid_v_single, fabsf(cpu_v - hybrid_v_single));
        printf("trace_attn_channel stage=v_proj_linear_only_mapped channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_v, hybrid_v_linear_only_last[kv_channel],
               fabsf(cpu_v - hybrid_v_linear_only_last[kv_channel]));
        printf("trace_attn_channel stage=context channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_context, hybrid_context, fabsf(cpu_context - hybrid_context));
        printf("trace_attn_channel stage=gated_context channel=%d head=%d offset=%d kv_head=%d kv_channel=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
               channel, head, offset, kv_head, kv_channel, cpu_gated_context, hybrid_gated_context, fabsf(cpu_gated_context - hybrid_gated_context));
    }
}

static void print_trace_v_neighbor_rows(const char *source,
                                        const float *cpu_v_proj_seq,
                                        const float *hybrid_v_proj_seq,
                                        int seq_len,
                                        int kv_dim,
                                        int head_dim,
                                        int q_per_kv,
                                        const int *trace_channels,
                                        int trace_channel_count) {
    if (!source || !cpu_v_proj_seq || !hybrid_v_proj_seq || !trace_channels ||
        trace_channel_count <= 0 || seq_len <= 0 || kv_dim <= 0 || head_dim <= 0 || q_per_kv <= 0) {
        return;
    }

    const float *cpu_v_last = cpu_v_proj_seq + (seq_len - 1) * kv_dim;
    const float *hybrid_v_last = hybrid_v_proj_seq + (seq_len - 1) * kv_dim;
    for (int i = 0; i < trace_channel_count; i++) {
        int channel = trace_channels[i];
        int head = channel / head_dim;
        int offset = channel % head_dim;
        int kv_head = head / q_per_kv;
        int kv_channel = kv_head * head_dim + offset;
        if (kv_channel < 0 || kv_channel >= kv_dim) continue;

        double self_diff = fabs((double)cpu_v_last[kv_channel] - (double)hybrid_v_last[kv_channel]);
        int global_best_cpu_channel = -1;
        double global_best_diff = INFINITY;
        int local_best_shift = 0;
        int local_best_cpu_channel = kv_channel;
        double local_best_diff = INFINITY;
        int radius = 8;
        for (int cpu_channel = 0; cpu_channel < kv_dim; cpu_channel++) {
            double diff = fabs((double)cpu_v_last[cpu_channel] - (double)hybrid_v_last[kv_channel]);
            if (diff < global_best_diff) {
                global_best_diff = diff;
                global_best_cpu_channel = cpu_channel;
            }
            int shift = cpu_channel - kv_channel;
            if (shift < -radius || shift > radius) continue;
            if (diff < local_best_diff) {
                local_best_diff = diff;
                local_best_shift = shift;
                local_best_cpu_channel = cpu_channel;
            }
        }
        printf("trace_v_neighbor source=%s channel=%d head=%d offset=%d kv_head=%d kv_channel=%d self_diff=%.6f local_best_shift=%d local_best_cpu_channel=%d local_best_diff=%.6f global_best_cpu_channel=%d global_best_diff=%.6f\n",
               source, channel, head, offset, kv_head, kv_channel, self_diff,
               local_best_shift, local_best_cpu_channel, local_best_diff,
               global_best_cpu_channel, global_best_diff);
    }
}

static inline float fp16_roundtrip(float x) {
    return (float)(_Float16)x;
}

static inline float sigmoid_scalar(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static inline float silu_scalar(float x) {
    return x * sigmoid_scalar(x);
}

static inline float sigmoid_scalar_fp16_emulated(float x) {
    float x16 = fp16_roundtrip(x);
    float neg = fp16_roundtrip(-x16);
    float expv = fp16_roundtrip(expf(neg));
    float denom = fp16_roundtrip(1.0f + expv);
    return fp16_roundtrip(1.0f / denom);
}

static inline float silu_scalar_fp16_emulated(float x) {
    float x16 = fp16_roundtrip(x);
    float sig = sigmoid_scalar_fp16_emulated(x16);
    return fp16_roundtrip(x16 * sig);
}

static inline float mul_scalar_fp16_emulated(float a, float b) {
    float a16 = fp16_roundtrip(a);
    float b16 = fp16_roundtrip(b);
    return fp16_roundtrip(a16 * b16);
}

static void apply_rope_text_inplace_local(float *q,
                                          float *k,
                                          int seq_len,
                                          int n_q_head,
                                          int n_kv_head,
                                          int head_dim,
                                          float rope_theta,
                                          float partial_rotary_factor) {
    int rotary_dim = (int)(head_dim * partial_rotary_factor);
    if (rotary_dim > head_dim) rotary_dim = head_dim;
    if (rotary_dim % 2 != 0) rotary_dim -= 1;
    if (rotary_dim <= 0) return;

    int half_rot = rotary_dim / 2;
    float *inv_freq = (float *)malloc((size_t)half_rot * sizeof(float));
    if (!inv_freq) return;

    for (int i = 0; i < half_rot; i++) {
        float exponent = (2.0f * (float)i) / (float)rotary_dim;
        inv_freq[i] = 1.0f / powf(rope_theta, exponent);
    }

    for (int pos = 0; pos < seq_len; pos++) {
        for (int h = 0; h < n_q_head; h++) {
            float *qh = q + pos * (n_q_head * head_dim) + h * head_dim;
            for (int i = 0; i < half_rot; i++) {
                float angle = (float)pos * inv_freq[i];
                float c = cosf(angle);
                float s = sinf(angle);
                float x0 = qh[i];
                float x1 = qh[i + half_rot];
                qh[i] = x0 * c - x1 * s;
                qh[i + half_rot] = x1 * c + x0 * s;
            }
        }
        for (int h = 0; h < n_kv_head; h++) {
            float *kh = k + pos * (n_kv_head * head_dim) + h * head_dim;
            for (int i = 0; i < half_rot; i++) {
                float angle = (float)pos * inv_freq[i];
                float c = cosf(angle);
                float s = sinf(angle);
                float x0 = kh[i];
                float x1 = kh[i + half_rot];
                kh[i] = x0 * c - x1 * s;
                kh[i + half_rot] = x1 * c + x0 * s;
            }
        }
    }

    free(inv_freq);
}

static int capture_attention_stages_from_projected_qkv(const float *q_proj_out_seq,
                                                       const float *k_proj_out_seq,
                                                       const float *v_proj_out_seq,
                                                       int seq_len,
                                                       const float *q_norm,
                                                       const float *k_norm,
                                                       int n_head,
                                                       int n_kv_head,
                                                       int head_dim,
                                                       float rope_theta,
                                                       float partial_rotary_factor,
                                                       float *q_normed_out,
                                                       float *gate_sigmoid_out,
                                                       float *k_normed_out,
                                                       float *q_rope_out,
                                                       float *k_rope_out,
                                                       float *scores_out,
                                                       float *probs_out,
                                                       float *context_out,
                                                       float *gated_context_out) {
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_kv_head * head_dim;
    const int q_per_kv = n_head / n_kv_head;
    const float scale = 1.0f / sqrtf((float)head_dim);

    float *gate_raw = (float *)calloc((size_t)seq_len * q_dim, sizeof(float));
    float *v_copy = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    float *scores = (float *)calloc((size_t)seq_len * seq_len, sizeof(float));
    float *probs = (float *)calloc((size_t)seq_len * seq_len, sizeof(float));
    float *qh = (float *)calloc((size_t)seq_len * head_dim, sizeof(float));
    float *kh = (float *)calloc((size_t)seq_len * head_dim, sizeof(float));
    float *vh = (float *)calloc((size_t)seq_len * head_dim, sizeof(float));
    float *context_h = (float *)calloc((size_t)seq_len * head_dim, sizeof(float));
    if (!gate_raw || !v_copy || !scores || !probs || !qh || !kh || !vh || !context_h) {
        free(gate_raw);
        free(v_copy);
        free(scores);
        free(probs);
        free(qh);
        free(kh);
        free(vh);
        free(context_h);
        return 0;
    }

    memcpy(v_copy, v_proj_out_seq, (size_t)seq_len * kv_dim * sizeof(float));
    memset(context_out, 0, (size_t)seq_len * q_dim * sizeof(float));
    memset(gated_context_out, 0, (size_t)seq_len * q_dim * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        memcpy(q_normed_out + s * q_dim,
               q_proj_out_seq + s * (q_dim * 2),
               (size_t)q_dim * sizeof(float));
        memcpy(gate_raw + s * q_dim,
               q_proj_out_seq + s * (q_dim * 2) + q_dim,
               (size_t)q_dim * sizeof(float));
        memcpy(k_normed_out + s * kv_dim,
               k_proj_out_seq + s * kv_dim,
               (size_t)kv_dim * sizeof(float));
    }

    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_head; h++) {
            orion_qwen_cpu_rmsnorm(q_normed_out + s * q_dim + h * head_dim,
                                   q_norm, head_dim, 1e-6f,
                                   q_normed_out + s * q_dim + h * head_dim);
        }
        for (int h = 0; h < n_kv_head; h++) {
            orion_qwen_cpu_rmsnorm(k_normed_out + s * kv_dim + h * head_dim,
                                   k_norm, head_dim, 1e-6f,
                                   k_normed_out + s * kv_dim + h * head_dim);
        }
        for (int i = 0; i < q_dim; i++) {
            gate_sigmoid_out[s * q_dim + i] = sigmoid_scalar(gate_raw[s * q_dim + i]);
        }
    }

    memcpy(q_rope_out, q_normed_out, (size_t)seq_len * q_dim * sizeof(float));
    memcpy(k_rope_out, k_normed_out, (size_t)seq_len * kv_dim * sizeof(float));
    apply_rope_text_inplace_local(q_rope_out, k_rope_out, seq_len, n_head, n_kv_head, head_dim, rope_theta, partial_rotary_factor);

    for (int h = 0; h < n_head; h++) {
        int kv_head = h / q_per_kv;
        memset(scores, 0, (size_t)seq_len * seq_len * sizeof(float));
        memset(probs, 0, (size_t)seq_len * seq_len * sizeof(float));
        memset(context_h, 0, (size_t)seq_len * head_dim * sizeof(float));

        for (int s = 0; s < seq_len; s++) {
            memcpy(qh + s * head_dim, q_rope_out + s * q_dim + h * head_dim, (size_t)head_dim * sizeof(float));
            memcpy(kh + s * head_dim, k_rope_out + s * kv_dim + kv_head * head_dim, (size_t)head_dim * sizeof(float));
            memcpy(vh + s * head_dim, v_copy + s * kv_dim + kv_head * head_dim, (size_t)head_dim * sizeof(float));
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim,
                    scale, qh, head_dim, kh, head_dim,
                    0.0f, scores, seq_len);

        for (int i = 0; i < seq_len; i++) {
            for (int j = i + 1; j < seq_len; j++) {
                scores[i * seq_len + j] = -INFINITY;
            }
            float max_val = scores[i * seq_len];
            for (int j = 1; j <= i; j++) {
                if (scores[i * seq_len + j] > max_val) max_val = scores[i * seq_len + j];
            }
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                probs[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                sum += probs[i * seq_len + j];
            }
            for (int j = 0; j <= i; j++) {
                probs[i * seq_len + j] /= sum;
            }
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len,
                    1.0f, probs, seq_len, vh, head_dim,
                    0.0f, context_h, head_dim);

        memcpy(scores_out + (size_t)h * seq_len * seq_len, scores, (size_t)seq_len * seq_len * sizeof(float));
        memcpy(probs_out + (size_t)h * seq_len * seq_len, probs, (size_t)seq_len * seq_len * sizeof(float));
        for (int s = 0; s < seq_len; s++) {
            memcpy(context_out + s * q_dim + h * head_dim,
                   context_h + s * head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }

    for (int i = 0; i < seq_len * q_dim; i++) {
        gated_context_out[i] = context_out[i] * gate_sigmoid_out[i];
    }

    free(gate_raw);
    free(v_copy);
    free(scores);
    free(probs);
    free(qh);
    free(kh);
    free(vh);
    free(context_h);
    return 1;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        OrionGPT2Tokenizer* tok = NULL;
        OrionQwen35Manifest* manifest = NULL;
        float *hidden = NULL;
        float *hidden_next = NULL;
        float *normed = NULL;
        float *normed_fp16input = NULL;
        float *cpu_q = NULL;
        float *cpu_k = NULL;
        float *cpu_v = NULL;
        float *cpu_v_rms_only = NULL;
        float *cpu_v_linear_fp16emu = NULL;
        float *cpu_v_linear_fp16acc32 = NULL;
        float *cpu_v_input_fp16 = NULL;
        float *cpu_input_rms_ms = NULL;
        float *cpu_input_rms_mse = NULL;
        float *cpu_input_rms_rrms = NULL;
        float *cpu_input_rms_xr = NULL;
        float *ane_q = NULL;
        float *ane_k = NULL;
        float *ane_v = NULL;
        float *ane_v_single = NULL;
        float *ane_input_rms = NULL;
        float *ane_input_rms_ms = NULL;
        float *ane_input_rms_rrms = NULL;
        float *ane_input_rms_rrms_pow_micro = NULL;
        float *ane_input_rms_xr = NULL;
        float *ane_v_linear_only = NULL;
        float *ane_v_linear_only_fp16input = NULL;
        float *cpu_attn = NULL;
        float *ane_attn = NULL;
        float *attn_gate_only = NULL;
        float *attn_v_only = NULL;
        float *attn_v_single_only = NULL;
        float *attn_v_linear_only = NULL;
        float *attn_v_linear_fp16emu = NULL;
        float *attn_v_linear_fp16acc32 = NULL;
        float *attn_v_input_fp16_cpu = NULL;
        float *attn_v_input_fp16_ane = NULL;
        float *attn_v_rms_only = NULL;
        float *sigmoid_only_gated_context = NULL;
        float *attn_sigmoid_only = NULL;
        float *ane_q_token_major = NULL;
        float *ane_k_token_major = NULL;
        float *ane_v_token_major = NULL;
        float *cpu_hidden_attn = NULL;
        float *ane_hidden_attn = NULL;
        float *gate_only_hidden_attn = NULL;
        float *sigmoid_only_hidden_attn = NULL;
        float *v_only_hidden_attn = NULL;
        float *cpu_attn_q_normed = NULL;
        float *ane_attn_q_normed = NULL;
        float *cpu_attn_gate_sigmoid = NULL;
        float *ane_attn_gate_sigmoid = NULL;
        float *cpu_attn_k_normed = NULL;
        float *ane_attn_k_normed = NULL;
        float *cpu_attn_q_rope = NULL;
        float *ane_attn_q_rope = NULL;
        float *cpu_attn_k_rope = NULL;
        float *ane_attn_k_rope = NULL;
        float *cpu_attn_scores = NULL;
        float *ane_attn_scores = NULL;
        float *cpu_attn_probs = NULL;
        float *ane_attn_probs = NULL;
        float *cpu_attn_context = NULL;
        float *ane_attn_context = NULL;
        float *cpu_attn_gated_context = NULL;
        float *ane_attn_gated_context = NULL;
        float *cpu_ffn_rms = NULL;
        float *ane_ffn_rms = NULL;
        float *cpu_ffn_gate = NULL;
        float *ane_ffn_gate = NULL;
        float *cpu_ffn_up = NULL;
        float *ane_ffn_up = NULL;
        float *ane_ffn_up_only = NULL;
        float *cpu_ffn_sigmoid = NULL;
        float *cpu_ffn_sigmoid_fp16emu = NULL;
        float *ane_ffn_sigmoid_micro = NULL;
        float *ane_ffn_sigmoid_tanh_micro = NULL;
        float *cpu_ffn_silu = NULL;
        float *cpu_ffn_silu_fp16emu = NULL;
        float *ane_ffn_silu = NULL;
        float *ane_ffn_silu_only = NULL;
        float *ane_ffn_silu_micro = NULL;
        float *ane_ffn_silu_tanh_micro = NULL;
        float *cpu_ffn_hidden = NULL;
        float *cpu_ffn_hidden_fp16emu = NULL;
        float *ane_ffn_hidden = NULL;
        float *ane_ffn_hidden_only = NULL;
        float *ane_ffn_mul_micro = NULL;
        float *cpu_ffn_down = NULL;
        float *ane_ffn_down_same_input = NULL;
        float *cpu_ffn_final = NULL;
        float *ane_ffn_final_same_input = NULL;
        float *hybrid_layer_final = NULL;
        float *input_ln = NULL;
        float *post_ln = NULL;
        float *q_proj = NULL;
        float *k_proj = NULL;
        float *v_proj = NULL;
        float *o_proj = NULL;
        float *q_norm = NULL;
        float *k_norm = NULL;
        float *gate_proj = NULL;
        float *up_proj = NULL;
        float *down_proj = NULL;
        float *final_norm = NULL;
        float *cpu_stage_last = NULL;
        float *hybrid_stage_last = NULL;
        OrionQwen35AneBridge bridge;
        memset(&bridge, 0, sizeof(bridge));
        IOSurfaceRef ioIn = NULL;
        IOSurfaceRef ioQ = NULL;
        IOSurfaceRef ioK = NULL;
        IOSurfaceRef ioV = NULL;
        IOSurfaceRef ioVSingle = NULL;
        IOSurfaceRef ioNormedIn = NULL;
        IOSurfaceRef ioNormedFp16In = NULL;
        IOSurfaceRef ioInputRmsOnly = NULL;
        IOSurfaceRef ioInputRmsMs = NULL;
        IOSurfaceRef ioInputRmsRrms = NULL;
        IOSurfaceRef ioInputRmsMseIn = NULL;
        IOSurfaceRef ioInputRmsRrmsPowMicro = NULL;
        IOSurfaceRef ioInputRmsXr = NULL;
        IOSurfaceRef ioVLinearOnly = NULL;
        IOSurfaceRef ioVLinearOnlyFp16In = NULL;
        IOSurfaceRef ioFfnIn = NULL;
        IOSurfaceRef ioFfnRms = NULL;
        IOSurfaceRef ioFfnGate = NULL;
        IOSurfaceRef ioFfnUp = NULL;
        IOSurfaceRef ioFfnUpOnly = NULL;
        IOSurfaceRef ioFfnSigmoidMicroIn = NULL;
        IOSurfaceRef ioFfnSigmoidMicroOut = NULL;
        IOSurfaceRef ioFfnSigmoidTanhMicroOut = NULL;
        IOSurfaceRef ioFfnSilu = NULL;
        IOSurfaceRef ioFfnSiluOnly = NULL;
        IOSurfaceRef ioFfnSiluMicroIn = NULL;
        IOSurfaceRef ioFfnSiluMicroOut = NULL;
        IOSurfaceRef ioFfnSiluTanhMicroOut = NULL;
        IOSurfaceRef ioFfnHidden = NULL;
        IOSurfaceRef ioFfnHiddenOnly = NULL;
        IOSurfaceRef ioFfnMulMicroA = NULL;
        IOSurfaceRef ioFfnMulMicroB = NULL;
        IOSurfaceRef ioFfnMulMicroOut = NULL;
        IOSurfaceRef ioFfnDownIn = NULL;
        IOSurfaceRef ioFfnDown = NULL;
        IOSurfaceRef ioHidden = NULL;
        IOSurfaceRef ioFfnInMixed = NULL;
        IOSurfaceRef ioHiddenMixed = NULL;

        if (argc < 3) {
            fprintf(stderr, "usage: %s <blob_dir> <tokenizer_dir> [prompt] [layer] [candidate_a] [candidate_b] [trace_dims] [trace_channels]\n", argv[0]);
            return 2;
        }
        if (!orion_ane_init()) {
            fprintf(stderr, "FAIL: orion_ane_init failed\n");
            return 3;
        }

        const char *blob_dir = argv[1];
        NSString* tokDir = [NSString stringWithUTF8String:argv[2]];
        NSDictionary* meta = load_json([tokDir stringByAppendingPathComponent:@"meta.json"]);
        if (!meta) {
            fprintf(stderr, "FAIL: missing tokenizer meta.json\n");
            return 1;
        }
        NSString* regex = meta[@"regex_pattern"];
        NSString* vocabPath = [tokDir stringByAppendingPathComponent:@"vocab.json"];
        NSString* mergesPath = [tokDir stringByAppendingPathComponent:@"merges.txt"];
        tok = orion_gpt2_tokenizer_load_with_regex(vocabPath.UTF8String, mergesPath.UTF8String, regex.UTF8String);
        if (!tok) {
            fprintf(stderr, "FAIL: tokenizer load failed\n");
            return 1;
        }

        manifest = orion_qwen35_manifest_load(blob_dir);
        if (!manifest) {
            fprintf(stderr, "FAIL: manifest loader returned NULL\n");
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        const char *prompt = (argc >= 4) ? argv[3] : "사진";
        int layer = (argc >= 5) ? atoi(argv[4]) : 3;
        int candidate_a = (argc >= 6) ? atoi(argv[5]) : -1;
        int candidate_b = (argc >= 7) ? atoi(argv[6]) : -1;
        int pair_enabled = (candidate_a >= 0 && candidate_b >= 0);
        if (layer < 0 || layer >= manifest->n_layer) {
            fprintf(stderr, "FAIL: invalid layer %d\n", layer);
            orion_qwen35_manifest_free(manifest);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        int token_ids[128] = {0};
        int seq_len = orion_gpt2_encode(tok, prompt, token_ids, 128);
        if (seq_len <= 0) {
            fprintf(stderr, "FAIL: prompt encode failed\n");
            orion_qwen35_manifest_free(manifest);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        const int d_model = manifest->d_model;
        const int d_ff = manifest->d_ff;
        const int n_head = manifest->n_head;
        const int n_kv_head = manifest->n_kv_head;
        const int q_per_kv = n_head / n_kv_head;
        const int head_dim = manifest->head_dim;
        const int q_dim = n_head * head_dim;
        const int kv_dim = n_kv_head * head_dim;
        const int bucket = 32;
        const int total = seq_len * d_model;
        const int total_score = n_head * seq_len * seq_len;
        const int total_q = seq_len * q_dim;
        const int total_kv = seq_len * kv_dim;
        const int total_ff = seq_len * d_ff;
        const char *lm_head_name = manifest->tie_word_embeddings ? "embed_tokens.bin" : "lm_head.bin";
        const char *trace_dims_csv = (argc >= 8) ? argv[7] : "";
        const char *trace_channels_csv = (argc >= 9) ? argv[8] : "";
        int trace_dims[ORION_TRACE_DIM_MAX] = {0};
        int trace_dim_count = parse_dim_list(trace_dims_csv, trace_dims, ORION_TRACE_DIM_MAX, d_model);
        int trace_channels[ORION_TRACE_DIM_MAX] = {0};
        int trace_channel_count = parse_dim_list(trace_channels_csv, trace_channels, ORION_TRACE_DIM_MAX, q_dim);
        const int q_proj_uses_cpu = use_cpu_q_proj_override();
        const int q_query_uses_cpu = use_cpu_q_query_override();
        const int q_gate_uses_cpu = use_cpu_q_gate_override();
        const int v_proj_uses_cpu = use_cpu_v_proj_override();
        int v_proj_cpu_channels[ORION_TRACE_DIM_MAX] = {0};
        int v_proj_cpu_channel_count = 0;
        if (!v_proj_uses_cpu) {
            v_proj_cpu_channel_count = load_v_proj_cpu_channel_override(v_proj_cpu_channels, ORION_TRACE_DIM_MAX, q_dim);
        }
        NSString *v_linear_only_tmp_dir = nil;
        NSString *v_linear_only_source_blob_path = nil;
        NSString *v_linear_only_data_path = nil;
        NSString *v_linear_only_weight_path = nil;
        NSData *v_linear_only_source_blob = nil;
        NSData *v_linear_only_source_payload = nil;
        NSData *v_linear_only_data_blob = nil;
        NSData *v_linear_only_weight_blob = nil;
        OrionByteCompareResult v_linear_only_data_vs_source = {0};
        OrionByteCompareResult v_linear_only_weight_vs_source = {0};
        OrionByteCompareResult v_linear_only_data_vs_weight = {0};
        OrionByteCompareResult v_linear_only_data_vs_source_payload = {0};

        hidden = (float *)calloc((size_t)total, sizeof(float));
        hidden_next = (float *)calloc((size_t)total, sizeof(float));
        normed = (float *)calloc((size_t)total, sizeof(float));
        normed_fp16input = (float *)calloc((size_t)total, sizeof(float));
        cpu_q = (float *)calloc((size_t)seq_len * (q_dim * 2), sizeof(float));
        cpu_k = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        cpu_v = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        cpu_v_rms_only = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        cpu_v_linear_fp16emu = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        cpu_v_linear_fp16acc32 = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        cpu_v_input_fp16 = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        cpu_input_rms_ms = (float *)calloc((size_t)seq_len, sizeof(float));
        cpu_input_rms_mse = (float *)calloc((size_t)seq_len, sizeof(float));
        cpu_input_rms_rrms = (float *)calloc((size_t)seq_len, sizeof(float));
        cpu_input_rms_xr = (float *)calloc((size_t)total, sizeof(float));
        ane_q = (float *)calloc((size_t)seq_len * (q_dim * 2), sizeof(float));
        ane_k = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        ane_v = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        ane_v_single = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        ane_input_rms = (float *)calloc((size_t)total, sizeof(float));
        ane_input_rms_ms = (float *)calloc((size_t)seq_len, sizeof(float));
        ane_input_rms_rrms = (float *)calloc((size_t)seq_len, sizeof(float));
        ane_input_rms_rrms_pow_micro = (float *)calloc((size_t)seq_len, sizeof(float));
        ane_input_rms_xr = (float *)calloc((size_t)total, sizeof(float));
        ane_v_linear_only = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        ane_v_linear_only_fp16input = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        ane_q_token_major = (float *)calloc((size_t)seq_len * (q_dim * 2), sizeof(float));
        ane_k_token_major = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        ane_v_token_major = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
        cpu_attn = (float *)calloc((size_t)total, sizeof(float));
        ane_attn = (float *)calloc((size_t)total, sizeof(float));
        attn_gate_only = (float *)calloc((size_t)total, sizeof(float));
        attn_v_only = (float *)calloc((size_t)total, sizeof(float));
        attn_v_single_only = (float *)calloc((size_t)total, sizeof(float));
        attn_v_linear_only = (float *)calloc((size_t)total, sizeof(float));
        attn_v_linear_fp16emu = (float *)calloc((size_t)total, sizeof(float));
        attn_v_linear_fp16acc32 = (float *)calloc((size_t)total, sizeof(float));
        attn_v_input_fp16_cpu = (float *)calloc((size_t)total, sizeof(float));
        attn_v_input_fp16_ane = (float *)calloc((size_t)total, sizeof(float));
        attn_v_rms_only = (float *)calloc((size_t)total, sizeof(float));
        sigmoid_only_gated_context = (float *)calloc((size_t)total_q, sizeof(float));
        attn_sigmoid_only = (float *)calloc((size_t)total, sizeof(float));
        cpu_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
        ane_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
        gate_only_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
        sigmoid_only_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
        v_only_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
        cpu_attn_q_normed = (float *)calloc((size_t)total_q, sizeof(float));
        ane_attn_q_normed = (float *)calloc((size_t)total_q, sizeof(float));
        cpu_attn_gate_sigmoid = (float *)calloc((size_t)total_q, sizeof(float));
        ane_attn_gate_sigmoid = (float *)calloc((size_t)total_q, sizeof(float));
        cpu_attn_k_normed = (float *)calloc((size_t)total_kv, sizeof(float));
        ane_attn_k_normed = (float *)calloc((size_t)total_kv, sizeof(float));
        cpu_attn_q_rope = (float *)calloc((size_t)total_q, sizeof(float));
        ane_attn_q_rope = (float *)calloc((size_t)total_q, sizeof(float));
        cpu_attn_k_rope = (float *)calloc((size_t)total_kv, sizeof(float));
        ane_attn_k_rope = (float *)calloc((size_t)total_kv, sizeof(float));
        cpu_attn_scores = (float *)calloc((size_t)total_score, sizeof(float));
        ane_attn_scores = (float *)calloc((size_t)total_score, sizeof(float));
        cpu_attn_probs = (float *)calloc((size_t)total_score, sizeof(float));
        ane_attn_probs = (float *)calloc((size_t)total_score, sizeof(float));
        cpu_attn_context = (float *)calloc((size_t)total_q, sizeof(float));
        ane_attn_context = (float *)calloc((size_t)total_q, sizeof(float));
        cpu_attn_gated_context = (float *)calloc((size_t)total_q, sizeof(float));
        ane_attn_gated_context = (float *)calloc((size_t)total_q, sizeof(float));
        cpu_ffn_rms = (float *)calloc((size_t)total, sizeof(float));
        ane_ffn_rms = (float *)calloc((size_t)total, sizeof(float));
        cpu_ffn_gate = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_gate = (float *)calloc((size_t)total_ff, sizeof(float));
        cpu_ffn_up = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_up = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_up_only = (float *)calloc((size_t)total_ff, sizeof(float));
        cpu_ffn_sigmoid = (float *)calloc((size_t)total_ff, sizeof(float));
        cpu_ffn_sigmoid_fp16emu = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_sigmoid_micro = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_sigmoid_tanh_micro = (float *)calloc((size_t)total_ff, sizeof(float));
        cpu_ffn_silu = (float *)calloc((size_t)total_ff, sizeof(float));
        cpu_ffn_silu_fp16emu = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_silu = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_silu_only = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_silu_micro = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_silu_tanh_micro = (float *)calloc((size_t)total_ff, sizeof(float));
        cpu_ffn_hidden = (float *)calloc((size_t)total_ff, sizeof(float));
        cpu_ffn_hidden_fp16emu = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_hidden = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_hidden_only = (float *)calloc((size_t)total_ff, sizeof(float));
        ane_ffn_mul_micro = (float *)calloc((size_t)total_ff, sizeof(float));
        cpu_ffn_down = (float *)calloc((size_t)total, sizeof(float));
        ane_ffn_down_same_input = (float *)calloc((size_t)total, sizeof(float));
        cpu_ffn_final = (float *)calloc((size_t)total, sizeof(float));
        ane_ffn_final_same_input = (float *)calloc((size_t)total, sizeof(float));
        hybrid_layer_final = (float *)calloc((size_t)total, sizeof(float));
        cpu_stage_last = (float *)calloc((size_t)d_model, sizeof(float));
        hybrid_stage_last = (float *)calloc((size_t)d_model, sizeof(float));
        if (!hidden || !hidden_next || !normed || !normed_fp16input || !cpu_q || !cpu_k || !cpu_v || !cpu_v_rms_only || !cpu_v_linear_fp16emu || !cpu_v_linear_fp16acc32 || !cpu_v_input_fp16 ||
            !cpu_input_rms_ms || !cpu_input_rms_mse || !cpu_input_rms_rrms || !cpu_input_rms_xr ||
            !ane_q || !ane_k || !ane_v || !ane_v_single || !ane_input_rms ||
            !ane_input_rms_ms || !ane_input_rms_rrms || !ane_input_rms_rrms_pow_micro || !ane_input_rms_xr ||
            !ane_v_linear_only || !ane_v_linear_only_fp16input ||
            !ane_q_token_major || !ane_k_token_major || !ane_v_token_major ||
            !cpu_attn || !ane_attn || !attn_gate_only || !attn_v_only || !attn_v_single_only || !attn_v_linear_only || !attn_v_linear_fp16emu || !attn_v_linear_fp16acc32 || !attn_v_input_fp16_cpu || !attn_v_input_fp16_ane || !attn_v_rms_only ||
            !sigmoid_only_gated_context || !attn_sigmoid_only ||
            !cpu_hidden_attn || !ane_hidden_attn || !gate_only_hidden_attn || !sigmoid_only_hidden_attn || !v_only_hidden_attn ||
            !cpu_attn_q_normed || !ane_attn_q_normed || !cpu_attn_gate_sigmoid || !ane_attn_gate_sigmoid ||
            !cpu_attn_k_normed || !ane_attn_k_normed || !cpu_attn_q_rope || !ane_attn_q_rope ||
            !cpu_attn_k_rope || !ane_attn_k_rope || !cpu_attn_scores || !ane_attn_scores ||
            !cpu_attn_probs || !ane_attn_probs || !cpu_attn_context || !ane_attn_context ||
            !cpu_attn_gated_context || !ane_attn_gated_context ||
            !cpu_ffn_rms || !ane_ffn_rms || !cpu_ffn_gate || !ane_ffn_gate ||
            !cpu_ffn_up || !ane_ffn_up || !ane_ffn_up_only ||
            !cpu_ffn_sigmoid || !cpu_ffn_sigmoid_fp16emu || !ane_ffn_sigmoid_micro || !ane_ffn_sigmoid_tanh_micro ||
            !cpu_ffn_silu || !cpu_ffn_silu_fp16emu || !ane_ffn_silu || !ane_ffn_silu_only || !ane_ffn_silu_micro || !ane_ffn_silu_tanh_micro ||
            !cpu_ffn_hidden || !cpu_ffn_hidden_fp16emu || !ane_ffn_hidden || !ane_ffn_hidden_only || !ane_ffn_mul_micro ||
            !cpu_ffn_down || !ane_ffn_down_same_input ||
            !cpu_ffn_final || !ane_ffn_final_same_input || !hybrid_layer_final ||
            !cpu_stage_last || !hybrid_stage_last) {
            fprintf(stderr, "FAIL: allocation failed\n");
            goto fail;
        }

        if (!load_embeddings(blob_dir, manifest, token_ids, seq_len, hidden)) {
            fprintf(stderr, "FAIL: embedding load failed\n");
            goto fail;
        }
        for (int i = 0; i < layer; i++) {
            if (!apply_cpu_layer(blob_dir, manifest, i, hidden, seq_len, hidden_next)) {
                fprintf(stderr, "FAIL: pre-layer cpu apply failed at layer %d\n", i);
                goto fail;
            }
            float *tmp = hidden;
            hidden = hidden_next;
            hidden_next = tmp;
        }

        input_ln = load_exact(blob_dir, layer, "input_layernorm.bin", d_model);
        post_ln = load_exact(blob_dir, layer, "post_attention_layernorm.bin", d_model);
        q_proj = load_exact(blob_dir, layer, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
        k_proj = load_exact(blob_dir, layer, "self_attn_k_proj.bin", kv_dim * d_model);
        v_proj = load_exact(blob_dir, layer, "self_attn_v_proj.bin", kv_dim * d_model);
        o_proj = load_exact(blob_dir, layer, "self_attn_o_proj.bin", d_model * q_dim);
        q_norm = load_exact(blob_dir, layer, "self_attn_q_norm.bin", head_dim);
        k_norm = load_exact(blob_dir, layer, "self_attn_k_norm.bin", head_dim);
        gate_proj = load_exact(blob_dir, layer, "mlp_gate_proj.bin", d_ff * d_model);
        up_proj = load_exact(blob_dir, layer, "mlp_up_proj.bin", d_ff * d_model);
        down_proj = load_exact(blob_dir, layer, "mlp_down_proj.bin", d_model * d_ff);
        char final_norm_path[2048];
        snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
        final_norm = orion_read_blob_f32_exact(final_norm_path, d_model);
        if (!input_ln || !post_ln || !q_proj || !k_proj || !v_proj || !o_proj || !q_norm || !k_norm ||
            !gate_proj || !up_proj || !down_proj || !final_norm) {
            fprintf(stderr, "FAIL: missing layer weights\n");
            goto fail;
        }

        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(hidden + s * d_model, input_ln, d_model, 1e-6f, normed + s * d_model);
            double sumsq = 0.0;
            for (int i = 0; i < d_model; i++) {
                double x = (double)hidden[s * d_model + i];
                sumsq += x * x;
            }
            cpu_input_rms_ms[s] = (float)(sumsq / (double)d_model);
            cpu_input_rms_mse[s] = cpu_input_rms_ms[s] + 1e-6f;
            cpu_input_rms_rrms[s] = powf(cpu_input_rms_mse[s], -0.5f);
            for (int i = 0; i < d_model; i++) {
                cpu_input_rms_xr[s * d_model + i] = hidden[s * d_model + i] * cpu_input_rms_rrms[s];
            }
        }
        for (int i = 0; i < total; i++) {
            normed_fp16input[i] = fp16_roundtrip(normed[i]);
        }

        cpu_linear_batch(normed, seq_len, q_proj, d_model, q_dim * 2, cpu_q);
        cpu_linear_batch(normed, seq_len, k_proj, d_model, kv_dim, cpu_k);
        cpu_linear_batch(normed, seq_len, v_proj, d_model, kv_dim, cpu_v);
        cpu_linear_batch_fp16_emulated(normed, seq_len, v_proj, d_model, kv_dim, cpu_v_linear_fp16emu);
        cpu_linear_batch_fp16in_fp32acc_fp16out(normed, seq_len, v_proj, d_model, kv_dim, cpu_v_linear_fp16acc32);
        cpu_linear_batch(normed_fp16input, seq_len, v_proj, d_model, kv_dim, cpu_v_input_fp16);

        if (!bridge_init(&bridge, [NSString stringWithUTF8String:blob_dir], layer, bucket, manifest)) {
            fprintf(stderr, "FAIL: bridge_init failed\n");
            goto fail;
        }

        v_linear_only_tmp_dir = orion_program_tmp_dir(bridge.prog_v_linear_only);
        v_linear_only_source_blob_path = [NSString stringWithFormat:@"%s/layer%d/self_attn_v_proj.bin", blob_dir, layer];
        if (v_linear_only_tmp_dir) {
            v_linear_only_data_path = [v_linear_only_tmp_dir stringByAppendingPathComponent:@"data"];
            v_linear_only_weight_path = [v_linear_only_tmp_dir stringByAppendingPathComponent:
                                         [NSString stringWithFormat:@"layer%d/self_attn_v_proj.bin", layer]];
        }
        v_linear_only_source_blob = [NSData dataWithContentsOfFile:v_linear_only_source_blob_path];
        v_linear_only_source_payload = load_blob_payload_data(v_linear_only_source_blob_path);
        v_linear_only_data_blob = v_linear_only_data_path ? [NSData dataWithContentsOfFile:v_linear_only_data_path] : nil;
        v_linear_only_weight_blob = v_linear_only_weight_path ? [NSData dataWithContentsOfFile:v_linear_only_weight_path] : nil;
        v_linear_only_data_vs_source = compare_nsdata(v_linear_only_data_blob, v_linear_only_source_blob);
        v_linear_only_weight_vs_source = compare_nsdata(v_linear_only_weight_blob, v_linear_only_source_blob);
        v_linear_only_data_vs_weight = compare_nsdata(v_linear_only_data_blob, v_linear_only_weight_blob);
        v_linear_only_data_vs_source_payload = compare_nsdata(v_linear_only_data_blob, v_linear_only_source_payload);

        ioIn = make_cpu_seq_input_surface(hidden, seq_len, bucket, d_model);
        ioQ = make_f32_surface((q_dim * 2) * bucket, 0.0f);
        ioK = make_f32_surface(kv_dim * bucket, 0.0f);
        ioV = make_f32_surface(kv_dim * bucket, 0.0f);
        ioVSingle = make_f32_surface(kv_dim * bucket, 0.0f);
        ioNormedIn = make_cpu_seq_input_surface(normed, seq_len, bucket, d_model);
        ioNormedFp16In = make_cpu_seq_input_surface(normed_fp16input, seq_len, bucket, d_model);
        ioInputRmsOnly = make_f32_surface(d_model * bucket, 0.0f);
        ioInputRmsMs = make_f32_surface(bucket, 0.0f);
        ioInputRmsRrms = make_f32_surface(bucket, 0.0f);
        ioInputRmsMseIn = make_cpu_seq_input_surface(cpu_input_rms_mse, seq_len, bucket, 1);
        ioInputRmsRrmsPowMicro = make_f32_surface(bucket, 0.0f);
        ioInputRmsXr = make_f32_surface(d_model * bucket, 0.0f);
        ioVLinearOnly = make_f32_surface(kv_dim * bucket, 0.0f);
        ioVLinearOnlyFp16In = make_f32_surface(kv_dim * bucket, 0.0f);
        IOSurfaceRef insQ[] = {ioIn};
        IOSurfaceRef insNormed[] = {ioNormedIn};
        IOSurfaceRef insNormedFp16[] = {ioNormedFp16In};
        IOSurfaceRef outsQ[] = {ioQ};
        IOSurfaceRef outsKV[] = {ioK, ioV};
        IOSurfaceRef outsVSingle[] = {ioVSingle};
        IOSurfaceRef outsInputRmsOnly[] = {ioInputRmsOnly};
        IOSurfaceRef outsInputRmsMs[] = {ioInputRmsMs};
        IOSurfaceRef outsInputRmsRrms[] = {ioInputRmsRrms};
        IOSurfaceRef outsInputRmsXr[] = {ioInputRmsXr};
        IOSurfaceRef insInputRmsMse[] = {ioInputRmsMseIn};
        IOSurfaceRef outsInputRmsRrmsPowMicro[] = {ioInputRmsRrmsPowMicro};
        IOSurfaceRef outsVLinearOnly[] = {ioVLinearOnly};
        IOSurfaceRef outsVLinearOnlyFp16In[] = {ioVLinearOnlyFp16In};
        if (!orion_eval(bridge.prog_q, insQ, 1, outsQ, 1) || !orion_eval(bridge.prog_kv, insQ, 1, outsKV, 2)) {
            fprintf(stderr, "FAIL: ANE q/kv eval failed\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_v_only, insQ, 1, outsVSingle, 1)) {
            fprintf(stderr, "FAIL: ANE v-only eval failed\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_input_rms_only, insQ, 1, outsInputRmsOnly, 1)) {
            fprintf(stderr, "FAIL: ANE input-rms-only eval failed\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_input_rms_ms, insQ, 1, outsInputRmsMs, 1)) {
            fprintf(stderr, "FAIL: ANE input-rms-ms eval failed\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_input_rms_rrms, insQ, 1, outsInputRmsRrms, 1)) {
            fprintf(stderr, "FAIL: ANE input-rms-rrms eval failed\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_input_rms_xr, insQ, 1, outsInputRmsXr, 1)) {
            fprintf(stderr, "FAIL: ANE input-rms-xr eval failed\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_rrms_pow_micro, insInputRmsMse, 1, outsInputRmsRrmsPowMicro, 1)) {
            fprintf(stderr, "FAIL: ANE rrms-pow-micro eval failed\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_v_linear_only, insNormed, 1, outsVLinearOnly, 1)) {
            fprintf(stderr, "FAIL: ANE v-linear-only eval failed\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_v_linear_only, insNormedFp16, 1, outsVLinearOnlyFp16In, 1)) {
            fprintf(stderr, "FAIL: ANE v-linear-only eval failed on fp16-rounded input\n");
            goto fail;
        }
        read_ane_surface_prefix(ioQ, q_dim * 2, seq_len, bucket, ane_q);
        read_ane_surface_prefix(ioK, kv_dim, seq_len, bucket, ane_k);
        read_ane_surface_prefix(ioV, kv_dim, seq_len, bucket, ane_v);
        read_ane_surface_prefix(ioVSingle, kv_dim, seq_len, bucket, ane_v_single);
        read_ane_surface_prefix(ioInputRmsOnly, d_model, seq_len, bucket, ane_input_rms);
        read_ane_surface_prefix(ioInputRmsMs, 1, seq_len, bucket, ane_input_rms_ms);
        read_ane_surface_prefix(ioInputRmsRrms, 1, seq_len, bucket, ane_input_rms_rrms);
        read_ane_surface_prefix(ioInputRmsRrmsPowMicro, 1, seq_len, bucket, ane_input_rms_rrms_pow_micro);
        read_ane_surface_prefix(ioInputRmsXr, d_model, seq_len, bucket, ane_input_rms_xr);
        read_ane_surface_prefix(ioVLinearOnly, kv_dim, seq_len, bucket, ane_v_linear_only);
        read_ane_surface_prefix(ioVLinearOnlyFp16In, kv_dim, seq_len, bucket, ane_v_linear_only_fp16input);
        read_ane_surface_prefix_token_major(ioQ, q_dim * 2, seq_len, ane_q_token_major);
        read_ane_surface_prefix_token_major(ioK, kv_dim, seq_len, ane_k_token_major);
        read_ane_surface_prefix_token_major(ioV, kv_dim, seq_len, ane_v_token_major);
        if (q_proj_uses_cpu || q_query_uses_cpu || q_gate_uses_cpu) {
            for (int s = 0; s < seq_len; s++) {
                float *dst = ane_q + (size_t)s * (q_dim * 2);
                float *dst_token_major = ane_q_token_major + (size_t)s * (q_dim * 2);
                const float *src = cpu_q + (size_t)s * (q_dim * 2);
                if (q_proj_uses_cpu || q_query_uses_cpu) {
                    memcpy(dst, src, (size_t)q_dim * sizeof(float));
                    memcpy(dst_token_major, src, (size_t)q_dim * sizeof(float));
                }
                if (q_proj_uses_cpu || q_gate_uses_cpu) {
                    memcpy(dst + q_dim, src + q_dim, (size_t)q_dim * sizeof(float));
                    memcpy(dst_token_major + q_dim, src + q_dim, (size_t)q_dim * sizeof(float));
                }
            }
        }
        if (v_proj_uses_cpu) {
            memcpy(ane_v, cpu_v, (size_t)seq_len * kv_dim * sizeof(float));
            memcpy(ane_v_single, cpu_v, (size_t)seq_len * kv_dim * sizeof(float));
            memcpy(ane_v_linear_only, cpu_v, (size_t)seq_len * kv_dim * sizeof(float));
            memcpy(ane_v_linear_only_fp16input, cpu_v, (size_t)seq_len * kv_dim * sizeof(float));
            memcpy(ane_v_token_major, cpu_v, (size_t)seq_len * kv_dim * sizeof(float));
        } else if (v_proj_cpu_channel_count > 0) {
            apply_v_proj_cpu_channel_overrides(ane_v, cpu_v, seq_len, kv_dim, head_dim, n_head / n_kv_head,
                                               v_proj_cpu_channels, v_proj_cpu_channel_count);
            apply_v_proj_cpu_channel_overrides(ane_v_single, cpu_v, seq_len, kv_dim, head_dim, n_head / n_kv_head,
                                               v_proj_cpu_channels, v_proj_cpu_channel_count);
            apply_v_proj_cpu_channel_overrides(ane_v_linear_only, cpu_v, seq_len, kv_dim, head_dim, n_head / n_kv_head,
                                               v_proj_cpu_channels, v_proj_cpu_channel_count);
            apply_v_proj_cpu_channel_overrides(ane_v_linear_only_fp16input, cpu_v, seq_len, kv_dim, head_dim, n_head / n_kv_head,
                                               v_proj_cpu_channels, v_proj_cpu_channel_count);
            apply_v_proj_cpu_channel_overrides(ane_v_token_major, cpu_v, seq_len, kv_dim, head_dim, n_head / n_kv_head,
                                               v_proj_cpu_channels, v_proj_cpu_channel_count);
        }
        cpu_linear_batch(ane_input_rms, seq_len, v_proj, d_model, kv_dim, cpu_v_rms_only);

        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q, cpu_k, cpu_v, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            cpu_attn
        );
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            ane_q, ane_k, ane_v, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            ane_attn
        );
        float *cpu_q_with_ane_gate = (float *)malloc((size_t)seq_len * (q_dim * 2) * sizeof(float));
        if (!cpu_q_with_ane_gate) {
            fprintf(stderr, "FAIL: allocation failed for cpu_q_with_ane_gate\n");
            goto fail;
        }
        memcpy(cpu_q_with_ane_gate, cpu_q, (size_t)seq_len * (q_dim * 2) * sizeof(float));
        for (int s = 0; s < seq_len; s++) {
            memcpy(cpu_q_with_ane_gate + s * (q_dim * 2) + q_dim,
                   ane_q + s * (q_dim * 2) + q_dim,
                   (size_t)q_dim * sizeof(float));
        }
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q_with_ane_gate, cpu_k, cpu_v, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn_gate_only
        );
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q, cpu_k, ane_v, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn_v_only
        );
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q, cpu_k, ane_v_single, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn_v_single_only
        );
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q, cpu_k, ane_v_linear_only, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn_v_linear_only
        );
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q, cpu_k, cpu_v_linear_fp16emu, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn_v_linear_fp16emu
        );
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q, cpu_k, cpu_v_linear_fp16acc32, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn_v_linear_fp16acc32
        );
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q, cpu_k, cpu_v_input_fp16, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn_v_input_fp16_cpu
        );
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q, cpu_k, ane_v_linear_only_fp16input, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn_v_input_fp16_ane
        );
        orion_qwen_cpu_full_attention_from_projections_with_rope(
            cpu_q, cpu_k, cpu_v_rms_only, seq_len, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn_v_rms_only
        );
        free(cpu_q_with_ane_gate);
        if (!capture_attention_stages_from_projected_qkv(cpu_q, cpu_k, cpu_v, seq_len,
                                                         q_norm, k_norm, n_head, n_kv_head, head_dim,
                                                         manifest->rope_theta, manifest->partial_rotary_factor,
                                                         cpu_attn_q_normed, cpu_attn_gate_sigmoid, cpu_attn_k_normed,
                                                         cpu_attn_q_rope, cpu_attn_k_rope, cpu_attn_scores, cpu_attn_probs,
                                                         cpu_attn_context, cpu_attn_gated_context)) {
            fprintf(stderr, "FAIL: cpu attention stage capture failed\n");
            goto fail;
        }
        if (!capture_attention_stages_from_projected_qkv(ane_q, ane_k, ane_v, seq_len,
                                                         q_norm, k_norm, n_head, n_kv_head, head_dim,
                                                         manifest->rope_theta, manifest->partial_rotary_factor,
                                                         ane_attn_q_normed, ane_attn_gate_sigmoid, ane_attn_k_normed,
                                                         ane_attn_q_rope, ane_attn_k_rope, ane_attn_scores, ane_attn_probs,
                                                         ane_attn_context, ane_attn_gated_context)) {
            fprintf(stderr, "FAIL: ane attention stage capture failed\n");
            goto fail;
        }
        for (int i = 0; i < total_q; i++) {
            sigmoid_only_gated_context[i] = cpu_attn_context[i] * ane_attn_gate_sigmoid[i];
        }
        cpu_linear_batch(sigmoid_only_gated_context, seq_len, o_proj, q_dim, d_model, attn_sigmoid_only);

        for (int i = 0; i < total; i++) {
            cpu_hidden_attn[i] = hidden[i] + cpu_attn[i];
            ane_hidden_attn[i] = hidden[i] + ane_attn[i];
            gate_only_hidden_attn[i] = hidden[i] + attn_gate_only[i];
            sigmoid_only_hidden_attn[i] = hidden[i] + attn_sigmoid_only[i];
            v_only_hidden_attn[i] = hidden[i] + attn_v_only[i];
        }

        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(cpu_hidden_attn + s * d_model, post_ln, d_model, 1e-6f, cpu_ffn_rms + s * d_model);
        }
        cpu_linear_batch(cpu_ffn_rms, seq_len, gate_proj, d_model, d_ff, cpu_ffn_gate);
        cpu_linear_batch(cpu_ffn_rms, seq_len, up_proj, d_model, d_ff, cpu_ffn_up);
        for (int i = 0; i < total_ff; i++) {
            cpu_ffn_sigmoid[i] = sigmoid_scalar(cpu_ffn_gate[i]);
            cpu_ffn_sigmoid_fp16emu[i] = sigmoid_scalar_fp16_emulated(cpu_ffn_gate[i]);
            cpu_ffn_silu[i] = silu_scalar(cpu_ffn_gate[i]);
            cpu_ffn_silu_fp16emu[i] = silu_scalar_fp16_emulated(cpu_ffn_gate[i]);
            cpu_ffn_hidden[i] = cpu_ffn_silu[i] * cpu_ffn_up[i];
            cpu_ffn_hidden_fp16emu[i] = mul_scalar_fp16_emulated(cpu_ffn_silu_fp16emu[i], cpu_ffn_up[i]);
        }
        cpu_linear_batch(cpu_ffn_hidden, seq_len, down_proj, d_ff, d_model, cpu_ffn_down);
        for (int i = 0; i < total; i++) cpu_ffn_final[i] = cpu_hidden_attn[i] + cpu_ffn_down[i];

        ioFfnIn = make_cpu_seq_input_surface(cpu_hidden_attn, seq_len, bucket, d_model);
        ioFfnRms = make_f32_surface(d_model * bucket, 0.0f);
        ioFfnGate = make_f32_surface(d_ff * bucket, 0.0f);
        ioFfnUp = make_f32_surface(d_ff * bucket, 0.0f);
        ioFfnUpOnly = make_f32_surface(d_ff * bucket, 0.0f);
        ioFfnSilu = make_f32_surface(d_ff * bucket, 0.0f);
        ioFfnSiluOnly = make_f32_surface(d_ff * bucket, 0.0f);
        ioFfnHidden = make_f32_surface(d_ff * bucket, 0.0f);
        ioFfnHiddenOnly = make_f32_surface(d_ff * bucket, 0.0f);
        ioHidden = make_f32_surface(d_model * bucket, 0.0f);
        IOSurfaceRef insFFN[] = {ioFfnIn};
        IOSurfaceRef outsFFNRms[] = {ioFfnRms};
        IOSurfaceRef outsFFNGateUp[] = {ioFfnGate, ioFfnUp, ioFfnSilu, ioFfnHidden};
        IOSurfaceRef outsFFNUpOnly[] = {ioFfnUpOnly};
        IOSurfaceRef outsFFNSiluOnly[] = {ioFfnSiluOnly};
        IOSurfaceRef outsFFNHiddenOnly[] = {ioFfnHiddenOnly};
        IOSurfaceRef outsFFN[] = {ioHidden};
        if (!orion_eval(bridge.prog_ffn_rms, insFFN, 1, outsFFNRms, 1)) {
            fprintf(stderr, "FAIL: ANE ffn rms eval failed on cpu_hidden_attn\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_ffn_gateup, insFFN, 1, outsFFNGateUp, 4)) {
            fprintf(stderr, "FAIL: ANE ffn gate/up eval failed on cpu_hidden_attn\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_ffn_up_only, insFFN, 1, outsFFNUpOnly, 1)) {
            fprintf(stderr, "FAIL: ANE ffn up-only eval failed on cpu_hidden_attn\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_ffn_silu_only, insFFN, 1, outsFFNSiluOnly, 1)) {
            fprintf(stderr, "FAIL: ANE ffn silu-only eval failed on cpu_hidden_attn\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_ffn_hidden_only, insFFN, 1, outsFFNHiddenOnly, 1)) {
            fprintf(stderr, "FAIL: ANE ffn hidden-only eval failed on cpu_hidden_attn\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_ffn, insFFN, 1, outsFFN, 1)) {
            fprintf(stderr, "FAIL: ANE ffn eval failed on cpu_hidden_attn\n");
            goto fail;
        }
        read_ane_surface_prefix(ioFfnRms, d_model, seq_len, bucket, ane_ffn_rms);
        read_ane_surface_prefix(ioFfnGate, d_ff, seq_len, bucket, ane_ffn_gate);
        read_ane_surface_prefix(ioFfnUp, d_ff, seq_len, bucket, ane_ffn_up);
        read_ane_surface_prefix(ioFfnUpOnly, d_ff, seq_len, bucket, ane_ffn_up_only);
        read_ane_surface_prefix(ioFfnSilu, d_ff, seq_len, bucket, ane_ffn_silu);
        read_ane_surface_prefix(ioFfnSiluOnly, d_ff, seq_len, bucket, ane_ffn_silu_only);
        read_ane_surface_prefix(ioFfnHidden, d_ff, seq_len, bucket, ane_ffn_hidden);
        read_ane_surface_prefix(ioFfnHiddenOnly, d_ff, seq_len, bucket, ane_ffn_hidden_only);
        read_ane_surface_prefix(ioHidden, d_model, seq_len, bucket, ane_ffn_final_same_input);

        ioFfnSigmoidMicroIn = make_cpu_seq_input_surface(cpu_ffn_gate, seq_len, bucket, d_ff);
        ioFfnSigmoidMicroOut = make_f32_surface(d_ff * bucket, 0.0f);
        ioFfnSigmoidTanhMicroOut = make_f32_surface(d_ff * bucket, 0.0f);
        IOSurfaceRef insSigmoidMicro[] = {ioFfnSigmoidMicroIn};
        IOSurfaceRef outsSigmoidMicro[] = {ioFfnSigmoidMicroOut};
        IOSurfaceRef outsSigmoidTanhMicro[] = {ioFfnSigmoidTanhMicroOut};
        if (!orion_eval(bridge.prog_sigmoid_micro, insSigmoidMicro, 1, outsSigmoidMicro, 1)) {
            fprintf(stderr, "FAIL: ANE sigmoid micro eval failed on cpu_ffn_gate\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_sigmoid_tanh_micro, insSigmoidMicro, 1, outsSigmoidTanhMicro, 1)) {
            fprintf(stderr, "FAIL: ANE sigmoid tanh micro eval failed on cpu_ffn_gate\n");
            goto fail;
        }
        read_ane_surface_prefix(ioFfnSigmoidMicroOut, d_ff, seq_len, bucket, ane_ffn_sigmoid_micro);
        read_ane_surface_prefix(ioFfnSigmoidTanhMicroOut, d_ff, seq_len, bucket, ane_ffn_sigmoid_tanh_micro);

        ioFfnSiluMicroIn = make_cpu_seq_input_surface(cpu_ffn_gate, seq_len, bucket, d_ff);
        ioFfnSiluMicroOut = make_f32_surface(d_ff * bucket, 0.0f);
        ioFfnSiluTanhMicroOut = make_f32_surface(d_ff * bucket, 0.0f);
        IOSurfaceRef insSiluMicro[] = {ioFfnSiluMicroIn};
        IOSurfaceRef outsSiluMicro[] = {ioFfnSiluMicroOut};
        IOSurfaceRef outsSiluTanhMicro[] = {ioFfnSiluTanhMicroOut};
        if (!orion_eval(bridge.prog_silu_micro, insSiluMicro, 1, outsSiluMicro, 1)) {
            fprintf(stderr, "FAIL: ANE silu micro eval failed on cpu_ffn_gate\n");
            goto fail;
        }
        if (!orion_eval(bridge.prog_silu_tanh_micro, insSiluMicro, 1, outsSiluTanhMicro, 1)) {
            fprintf(stderr, "FAIL: ANE silu tanh micro eval failed on cpu_ffn_gate\n");
            goto fail;
        }
        read_ane_surface_prefix(ioFfnSiluMicroOut, d_ff, seq_len, bucket, ane_ffn_silu_micro);
        read_ane_surface_prefix(ioFfnSiluTanhMicroOut, d_ff, seq_len, bucket, ane_ffn_silu_tanh_micro);

        ioFfnMulMicroA = make_cpu_seq_input_surface(cpu_ffn_silu, seq_len, bucket, d_ff);
        ioFfnMulMicroB = make_cpu_seq_input_surface(cpu_ffn_up, seq_len, bucket, d_ff);
        ioFfnMulMicroOut = make_f32_surface(d_ff * bucket, 0.0f);
        IOSurfaceRef insMulMicro[] = {ioFfnMulMicroA, ioFfnMulMicroB};
        IOSurfaceRef outsMulMicro[] = {ioFfnMulMicroOut};
        if (!orion_eval(bridge.prog_mul_micro, insMulMicro, 2, outsMulMicro, 1)) {
            fprintf(stderr, "FAIL: ANE mul micro eval failed on cpu_ffn_silu/cpu_ffn_up\n");
            goto fail;
        }
        read_ane_surface_prefix(ioFfnMulMicroOut, d_ff, seq_len, bucket, ane_ffn_mul_micro);

        ioFfnDownIn = make_cpu_seq_input_surface(cpu_ffn_hidden, seq_len, bucket, d_ff);
        ioFfnDown = make_f32_surface(d_model * bucket, 0.0f);
        IOSurfaceRef insFFNDown[] = {ioFfnDownIn};
        IOSurfaceRef outsFFNDown[] = {ioFfnDown};
        if (!orion_eval(bridge.prog_ffn_down, insFFNDown, 1, outsFFNDown, 1)) {
            fprintf(stderr, "FAIL: ANE ffn down eval failed on cpu_ffn_hidden\n");
            goto fail;
        }
        read_ane_surface_prefix(ioFfnDown, d_model, seq_len, bucket, ane_ffn_down_same_input);

        ioFfnInMixed = make_cpu_seq_input_surface(ane_hidden_attn, seq_len, bucket, d_model);
        ioHiddenMixed = make_f32_surface(d_model * bucket, 0.0f);
        IOSurfaceRef insFFNMixed[] = {ioFfnInMixed};
        IOSurfaceRef outsFFNMixed[] = {ioHiddenMixed};
        if (!orion_eval(bridge.prog_ffn, insFFNMixed, 1, outsFFNMixed, 1)) {
            fprintf(stderr, "FAIL: ANE ffn eval failed on ane_hidden_attn\n");
            goto fail;
        }
        read_ane_surface_prefix(ioHiddenMixed, d_model, seq_len, bucket, hybrid_layer_final);

        double q_mean = mean_abs_diff(cpu_q, ane_q, seq_len * (q_dim * 2));
        double q_max = max_abs_diff(cpu_q, ane_q, seq_len * (q_dim * 2));
        double query_mean = mean_abs_diff_q_half(cpu_q, ane_q, seq_len, q_dim, 0);
        double query_max = max_abs_diff_q_half(cpu_q, ane_q, seq_len, q_dim, 0);
        double gate_mean = mean_abs_diff_q_half(cpu_q, ane_q, seq_len, q_dim, 1);
        double gate_max = max_abs_diff_q_half(cpu_q, ane_q, seq_len, q_dim, 1);
        double q_half_swap_mean = mean_abs_diff_q_half_swapped(cpu_q, ane_q, seq_len, q_dim);
        double q_half_swap_max = max_abs_diff_q_half_swapped(cpu_q, ane_q, seq_len, q_dim);
        double k_mean = mean_abs_diff(cpu_k, ane_k, seq_len * kv_dim);
        double k_max = max_abs_diff(cpu_k, ane_k, seq_len * kv_dim);
        double input_rms_ms_mean = mean_abs_diff(cpu_input_rms_ms, ane_input_rms_ms, seq_len);
        double input_rms_ms_max = max_abs_diff(cpu_input_rms_ms, ane_input_rms_ms, seq_len);
        double input_rms_rrms_mean = mean_abs_diff(cpu_input_rms_rrms, ane_input_rms_rrms, seq_len);
        double input_rms_rrms_max = max_abs_diff(cpu_input_rms_rrms, ane_input_rms_rrms, seq_len);
        double input_rms_rrms_pow_micro_mean = mean_abs_diff(cpu_input_rms_rrms, ane_input_rms_rrms_pow_micro, seq_len);
        double input_rms_rrms_pow_micro_max = max_abs_diff(cpu_input_rms_rrms, ane_input_rms_rrms_pow_micro, seq_len);
        double input_rms_rrms_stage_vs_pow_mean = mean_abs_diff(ane_input_rms_rrms, ane_input_rms_rrms_pow_micro, seq_len);
        double input_rms_rrms_stage_vs_pow_max = max_abs_diff(ane_input_rms_rrms, ane_input_rms_rrms_pow_micro, seq_len);
        double input_rms_xr_mean = mean_abs_diff(cpu_input_rms_xr, ane_input_rms_xr, total);
        double input_rms_xr_max = max_abs_diff(cpu_input_rms_xr, ane_input_rms_xr, total);
        double input_rms_mean = mean_abs_diff(normed, ane_input_rms, total);
        double input_rms_max = max_abs_diff(normed, ane_input_rms, total);
        double v_mean = mean_abs_diff(cpu_v, ane_v, seq_len * kv_dim);
        double v_max = max_abs_diff(cpu_v, ane_v, seq_len * kv_dim);
        double v_rms_only_mean = mean_abs_diff(cpu_v, cpu_v_rms_only, seq_len * kv_dim);
        double v_rms_only_max = max_abs_diff(cpu_v, cpu_v_rms_only, seq_len * kv_dim);
        double v_single_mean = mean_abs_diff(cpu_v, ane_v_single, seq_len * kv_dim);
        double v_single_max = max_abs_diff(cpu_v, ane_v_single, seq_len * kv_dim);
        double v_linear_only_mean = mean_abs_diff(cpu_v, ane_v_linear_only, seq_len * kv_dim);
        double v_linear_only_max = max_abs_diff(cpu_v, ane_v_linear_only, seq_len * kv_dim);
        double v_input_fp16_cpu_mean = mean_abs_diff(cpu_v, cpu_v_input_fp16, seq_len * kv_dim);
        double v_input_fp16_cpu_max = max_abs_diff(cpu_v, cpu_v_input_fp16, seq_len * kv_dim);
        double v_input_fp16_ane_mean = mean_abs_diff(cpu_v_input_fp16, ane_v_linear_only_fp16input, seq_len * kv_dim);
        double v_input_fp16_ane_max = max_abs_diff(cpu_v_input_fp16, ane_v_linear_only_fp16input, seq_len * kv_dim);
        double v_input_fp16_ane_self_mean = mean_abs_diff(ane_v_linear_only, ane_v_linear_only_fp16input, seq_len * kv_dim);
        double v_input_fp16_ane_self_max = max_abs_diff(ane_v_linear_only, ane_v_linear_only_fp16input, seq_len * kv_dim);
        double v_fp16emu_cpu_mean = mean_abs_diff(cpu_v, cpu_v_linear_fp16emu, seq_len * kv_dim);
        double v_fp16emu_cpu_max = max_abs_diff(cpu_v, cpu_v_linear_fp16emu, seq_len * kv_dim);
        double v_fp16emu_ane_mean = mean_abs_diff(cpu_v_linear_fp16emu, ane_v_linear_only, seq_len * kv_dim);
        double v_fp16emu_ane_max = max_abs_diff(cpu_v_linear_fp16emu, ane_v_linear_only, seq_len * kv_dim);
        double v_fp16acc32_cpu_mean = mean_abs_diff(cpu_v, cpu_v_linear_fp16acc32, seq_len * kv_dim);
        double v_fp16acc32_cpu_max = max_abs_diff(cpu_v, cpu_v_linear_fp16acc32, seq_len * kv_dim);
        double v_fp16acc32_ane_mean = mean_abs_diff(cpu_v_linear_fp16acc32, ane_v_linear_only, seq_len * kv_dim);
        double v_fp16acc32_ane_max = max_abs_diff(cpu_v_linear_fp16acc32, ane_v_linear_only, seq_len * kv_dim);
        double v_multi_vs_single_mean = mean_abs_diff(ane_v, ane_v_single, seq_len * kv_dim);
        double v_multi_vs_single_max = max_abs_diff(ane_v, ane_v_single, seq_len * kv_dim);
        double kv_swap_mean = mean_abs_diff_kv_swapped(cpu_k, cpu_v, ane_k, ane_v, seq_len * kv_dim);
        double kv_swap_max = max_abs_diff_kv_swapped(cpu_k, cpu_v, ane_k, ane_v, seq_len * kv_dim);
        double q_token_major_mean = mean_abs_diff(cpu_q, ane_q_token_major, seq_len * (q_dim * 2));
        double q_token_major_max = max_abs_diff(cpu_q, ane_q_token_major, seq_len * (q_dim * 2));
        double k_token_major_mean = mean_abs_diff(cpu_k, ane_k_token_major, seq_len * kv_dim);
        double k_token_major_max = max_abs_diff(cpu_k, ane_k_token_major, seq_len * kv_dim);
        double v_token_major_mean = mean_abs_diff(cpu_v, ane_v_token_major, seq_len * kv_dim);
        double v_token_major_max = max_abs_diff(cpu_v, ane_v_token_major, seq_len * kv_dim);
        double attn_q_norm_mean = mean_abs_diff(cpu_attn_q_normed, ane_attn_q_normed, total_q);
        double attn_q_norm_max = max_abs_diff(cpu_attn_q_normed, ane_attn_q_normed, total_q);
        double attn_gate_sigmoid_mean = mean_abs_diff(cpu_attn_gate_sigmoid, ane_attn_gate_sigmoid, total_q);
        double attn_gate_sigmoid_max = max_abs_diff(cpu_attn_gate_sigmoid, ane_attn_gate_sigmoid, total_q);
        double attn_k_norm_mean = mean_abs_diff(cpu_attn_k_normed, ane_attn_k_normed, total_kv);
        double attn_k_norm_max = max_abs_diff(cpu_attn_k_normed, ane_attn_k_normed, total_kv);
        double attn_q_rope_mean = mean_abs_diff(cpu_attn_q_rope, ane_attn_q_rope, total_q);
        double attn_q_rope_max = max_abs_diff(cpu_attn_q_rope, ane_attn_q_rope, total_q);
        double attn_k_rope_mean = mean_abs_diff(cpu_attn_k_rope, ane_attn_k_rope, total_kv);
        double attn_k_rope_max = max_abs_diff(cpu_attn_k_rope, ane_attn_k_rope, total_kv);
        double attn_score_mean = mean_abs_diff(cpu_attn_scores, ane_attn_scores, total_score);
        double attn_score_max = max_abs_diff(cpu_attn_scores, ane_attn_scores, total_score);
        double attn_softmax_mean = mean_abs_diff(cpu_attn_probs, ane_attn_probs, total_score);
        double attn_softmax_max = max_abs_diff(cpu_attn_probs, ane_attn_probs, total_score);
        double attn_context_mean = mean_abs_diff(cpu_attn_context, ane_attn_context, total_q);
        double attn_context_max = max_abs_diff(cpu_attn_context, ane_attn_context, total_q);
        double attn_gated_context_mean = mean_abs_diff(cpu_attn_gated_context, ane_attn_gated_context, total_q);
        double attn_gated_context_max = max_abs_diff(cpu_attn_gated_context, ane_attn_gated_context, total_q);
        double attn_mean = mean_abs_diff(cpu_attn, ane_attn, total);
        double attn_max = max_abs_diff(cpu_attn, ane_attn, total);
        double attn_gate_only_mean = mean_abs_diff(cpu_attn, attn_gate_only, total);
        double attn_gate_only_max = max_abs_diff(cpu_attn, attn_gate_only, total);
        double attn_v_only_mean = mean_abs_diff(cpu_attn, attn_v_only, total);
        double attn_v_only_max = max_abs_diff(cpu_attn, attn_v_only, total);
        double attn_v_single_only_mean = mean_abs_diff(cpu_attn, attn_v_single_only, total);
        double attn_v_single_only_max = max_abs_diff(cpu_attn, attn_v_single_only, total);
        double attn_v_linear_only_mean = mean_abs_diff(cpu_attn, attn_v_linear_only, total);
        double attn_v_linear_only_max = max_abs_diff(cpu_attn, attn_v_linear_only, total);
        double attn_sigmoid_only_mean = mean_abs_diff(cpu_attn, attn_sigmoid_only, total);
        double attn_sigmoid_only_max = max_abs_diff(cpu_attn, attn_sigmoid_only, total);
        double attn_v_input_fp16_cpu_mean = mean_abs_diff(cpu_attn, attn_v_input_fp16_cpu, total);
        double attn_v_input_fp16_cpu_max = max_abs_diff(cpu_attn, attn_v_input_fp16_cpu, total);
        double attn_v_input_fp16_ane_mean = mean_abs_diff(attn_v_input_fp16_cpu, attn_v_input_fp16_ane, total);
        double attn_v_input_fp16_ane_max = max_abs_diff(attn_v_input_fp16_cpu, attn_v_input_fp16_ane, total);
        double attn_v_input_fp16_ane_self_mean = mean_abs_diff(attn_v_linear_only, attn_v_input_fp16_ane, total);
        double attn_v_input_fp16_ane_self_max = max_abs_diff(attn_v_linear_only, attn_v_input_fp16_ane, total);
        double attn_v_fp16emu_cpu_mean = mean_abs_diff(cpu_attn, attn_v_linear_fp16emu, total);
        double attn_v_fp16emu_cpu_max = max_abs_diff(cpu_attn, attn_v_linear_fp16emu, total);
        double attn_v_fp16emu_ane_mean = mean_abs_diff(attn_v_linear_fp16emu, attn_v_linear_only, total);
        double attn_v_fp16emu_ane_max = max_abs_diff(attn_v_linear_fp16emu, attn_v_linear_only, total);
        double attn_v_fp16acc32_cpu_mean = mean_abs_diff(cpu_attn, attn_v_linear_fp16acc32, total);
        double attn_v_fp16acc32_cpu_max = max_abs_diff(cpu_attn, attn_v_linear_fp16acc32, total);
        double attn_v_fp16acc32_ane_mean = mean_abs_diff(attn_v_linear_fp16acc32, attn_v_linear_only, total);
        double attn_v_fp16acc32_ane_max = max_abs_diff(attn_v_linear_fp16acc32, attn_v_linear_only, total);
        double attn_v_rms_only_mean = mean_abs_diff(cpu_attn, attn_v_rms_only, total);
        double attn_v_rms_only_max = max_abs_diff(cpu_attn, attn_v_rms_only, total);
        double hidden_attn_mean = mean_abs_diff(cpu_hidden_attn, ane_hidden_attn, total);
        double hidden_attn_max = max_abs_diff(cpu_hidden_attn, ane_hidden_attn, total);
        double ffn_rms_mean = mean_abs_diff(cpu_ffn_rms, ane_ffn_rms, total);
        double ffn_rms_max = max_abs_diff(cpu_ffn_rms, ane_ffn_rms, total);
        double ffn_gate_mean = mean_abs_diff(cpu_ffn_gate, ane_ffn_gate, total_ff);
        double ffn_gate_max = max_abs_diff(cpu_ffn_gate, ane_ffn_gate, total_ff);
        double ffn_up_mean = mean_abs_diff(cpu_ffn_up, ane_ffn_up, total_ff);
        double ffn_up_max = max_abs_diff(cpu_ffn_up, ane_ffn_up, total_ff);
        double ffn_up_only_mean = mean_abs_diff(cpu_ffn_up, ane_ffn_up_only, total_ff);
        double ffn_up_only_max = max_abs_diff(cpu_ffn_up, ane_ffn_up_only, total_ff);
        double ffn_sigmoid_micro_mean = mean_abs_diff(cpu_ffn_sigmoid, ane_ffn_sigmoid_micro, total_ff);
        double ffn_sigmoid_micro_max = max_abs_diff(cpu_ffn_sigmoid, ane_ffn_sigmoid_micro, total_ff);
        double ffn_sigmoid_tanh_micro_mean = mean_abs_diff(cpu_ffn_sigmoid, ane_ffn_sigmoid_tanh_micro, total_ff);
        double ffn_sigmoid_tanh_micro_max = max_abs_diff(cpu_ffn_sigmoid, ane_ffn_sigmoid_tanh_micro, total_ff);
        double ffn_sigmoid_fp16emu_cpu_mean = mean_abs_diff(cpu_ffn_sigmoid, cpu_ffn_sigmoid_fp16emu, total_ff);
        double ffn_sigmoid_fp16emu_cpu_max = max_abs_diff(cpu_ffn_sigmoid, cpu_ffn_sigmoid_fp16emu, total_ff);
        double ffn_sigmoid_fp16emu_ane_mean = mean_abs_diff(cpu_ffn_sigmoid_fp16emu, ane_ffn_sigmoid_micro, total_ff);
        double ffn_sigmoid_fp16emu_ane_max = max_abs_diff(cpu_ffn_sigmoid_fp16emu, ane_ffn_sigmoid_micro, total_ff);
        double ffn_silu_mean = mean_abs_diff(cpu_ffn_silu, ane_ffn_silu, total_ff);
        double ffn_silu_max = max_abs_diff(cpu_ffn_silu, ane_ffn_silu, total_ff);
        double ffn_silu_only_mean = mean_abs_diff(cpu_ffn_silu, ane_ffn_silu_only, total_ff);
        double ffn_silu_only_max = max_abs_diff(cpu_ffn_silu, ane_ffn_silu_only, total_ff);
        double ffn_silu_micro_mean = mean_abs_diff(cpu_ffn_silu, ane_ffn_silu_micro, total_ff);
        double ffn_silu_micro_max = max_abs_diff(cpu_ffn_silu, ane_ffn_silu_micro, total_ff);
        double ffn_silu_tanh_micro_mean = mean_abs_diff(cpu_ffn_silu, ane_ffn_silu_tanh_micro, total_ff);
        double ffn_silu_tanh_micro_max = max_abs_diff(cpu_ffn_silu, ane_ffn_silu_tanh_micro, total_ff);
        double ffn_silu_fp16emu_cpu_mean = mean_abs_diff(cpu_ffn_silu, cpu_ffn_silu_fp16emu, total_ff);
        double ffn_silu_fp16emu_cpu_max = max_abs_diff(cpu_ffn_silu, cpu_ffn_silu_fp16emu, total_ff);
        double ffn_silu_fp16emu_ane_mean = mean_abs_diff(cpu_ffn_silu_fp16emu, ane_ffn_silu_micro, total_ff);
        double ffn_silu_fp16emu_ane_max = max_abs_diff(cpu_ffn_silu_fp16emu, ane_ffn_silu_micro, total_ff);
        double ffn_hidden_mean = mean_abs_diff(cpu_ffn_hidden, ane_ffn_hidden, total_ff);
        double ffn_hidden_max = max_abs_diff(cpu_ffn_hidden, ane_ffn_hidden, total_ff);
        double ffn_hidden_only_mean = mean_abs_diff(cpu_ffn_hidden, ane_ffn_hidden_only, total_ff);
        double ffn_hidden_only_max = max_abs_diff(cpu_ffn_hidden, ane_ffn_hidden_only, total_ff);
        double ffn_mul_micro_mean = mean_abs_diff(cpu_ffn_hidden, ane_ffn_mul_micro, total_ff);
        double ffn_mul_micro_max = max_abs_diff(cpu_ffn_hidden, ane_ffn_mul_micro, total_ff);
        double ffn_hidden_fp16emu_cpu_mean = mean_abs_diff(cpu_ffn_hidden, cpu_ffn_hidden_fp16emu, total_ff);
        double ffn_hidden_fp16emu_cpu_max = max_abs_diff(cpu_ffn_hidden, cpu_ffn_hidden_fp16emu, total_ff);
        double ffn_hidden_fp16emu_ane_mean = mean_abs_diff(cpu_ffn_hidden_fp16emu, ane_ffn_hidden_only, total_ff);
        double ffn_hidden_fp16emu_ane_max = max_abs_diff(cpu_ffn_hidden_fp16emu, ane_ffn_hidden_only, total_ff);
        double ffn_down_same_input_mean = mean_abs_diff(cpu_ffn_down, ane_ffn_down_same_input, total);
        double ffn_down_same_input_max = max_abs_diff(cpu_ffn_down, ane_ffn_down_same_input, total);
        double ffn_same_input_mean = mean_abs_diff(cpu_ffn_final, ane_ffn_final_same_input, total);
        double ffn_same_input_max = max_abs_diff(cpu_ffn_final, ane_ffn_final_same_input, total);
        double full_layer_mean = mean_abs_diff(cpu_ffn_final, hybrid_layer_final, total);
        double full_layer_max = max_abs_diff(cpu_ffn_final, hybrid_layer_final, total);
        float cpu_hidden_attn_pair_logits[2] = {0.0f, 0.0f};
        float hybrid_hidden_attn_pair_logits[2] = {0.0f, 0.0f};
        float gate_only_hidden_attn_pair_logits[2] = {0.0f, 0.0f};
        float sigmoid_only_hidden_attn_pair_logits[2] = {0.0f, 0.0f};
        float v_only_hidden_attn_pair_logits[2] = {0.0f, 0.0f};
        float cpu_ffn_final_pair_logits[2] = {0.0f, 0.0f};
        float ane_ffn_same_input_pair_logits[2] = {0.0f, 0.0f};
        float hybrid_layer_final_pair_logits[2] = {0.0f, 0.0f};
        double cpu_hidden_attn_pair_gap = 0.0;
        double hybrid_hidden_attn_pair_gap = 0.0;
        double gate_only_hidden_attn_pair_gap = 0.0;
        double sigmoid_only_hidden_attn_pair_gap = 0.0;
        double v_only_hidden_attn_pair_gap = 0.0;
        double cpu_ffn_final_pair_gap = 0.0;
        double ane_ffn_same_input_pair_gap = 0.0;
        double hybrid_layer_final_pair_gap = 0.0;
        int cpu_hidden_attn_pref_token = -1;
        int hybrid_hidden_attn_pref_token = -1;
        int gate_only_hidden_attn_pref_token = -1;
        int sigmoid_only_hidden_attn_pref_token = -1;
        int v_only_hidden_attn_pref_token = -1;
        int cpu_ffn_final_pref_token = -1;
        int ane_ffn_same_input_pref_token = -1;
        int hybrid_layer_final_pref_token = -1;

        if (pair_enabled) {
            const int pair_ids[2] = {candidate_a, candidate_b};
            const float *cpu_hidden_attn_last = cpu_hidden_attn + (seq_len - 1) * d_model;
            const float *ane_hidden_attn_last = ane_hidden_attn + (seq_len - 1) * d_model;
            const float *gate_only_hidden_attn_last = gate_only_hidden_attn + (seq_len - 1) * d_model;
            const float *sigmoid_only_hidden_attn_last = sigmoid_only_hidden_attn + (seq_len - 1) * d_model;
            const float *v_only_hidden_attn_last = v_only_hidden_attn + (seq_len - 1) * d_model;
            const float *cpu_ffn_final_last = cpu_ffn_final + (seq_len - 1) * d_model;
            const float *ane_ffn_final_same_input_last = ane_ffn_final_same_input + (seq_len - 1) * d_model;
            const float *hybrid_layer_final_last = hybrid_layer_final + (seq_len - 1) * d_model;

            orion_qwen_cpu_rmsnorm(cpu_hidden_attn_last, final_norm, d_model, 1e-6f, cpu_stage_last);
            orion_qwen_cpu_rmsnorm(ane_hidden_attn_last, final_norm, d_model, 1e-6f, hybrid_stage_last);
            if (!selected_token_logits(blob_dir, lm_head_name, cpu_stage_last, d_model, pair_ids, 2, cpu_hidden_attn_pair_logits) ||
                !selected_token_logits(blob_dir, lm_head_name, hybrid_stage_last, d_model, pair_ids, 2, hybrid_hidden_attn_pair_logits)) {
                fprintf(stderr, "FAIL: pair logits failed at hidden_attn stage\n");
                goto fail;
            }
            orion_qwen_cpu_rmsnorm(gate_only_hidden_attn_last, final_norm, d_model, 1e-6f, cpu_stage_last);
            if (!selected_token_logits(blob_dir, lm_head_name, cpu_stage_last, d_model, pair_ids, 2, gate_only_hidden_attn_pair_logits)) {
                fprintf(stderr, "FAIL: pair logits failed at gate-only hidden_attn stage\n");
                goto fail;
            }
            orion_qwen_cpu_rmsnorm(sigmoid_only_hidden_attn_last, final_norm, d_model, 1e-6f, cpu_stage_last);
            if (!selected_token_logits(blob_dir, lm_head_name, cpu_stage_last, d_model, pair_ids, 2, sigmoid_only_hidden_attn_pair_logits)) {
                fprintf(stderr, "FAIL: pair logits failed at sigmoid-only hidden_attn stage\n");
                goto fail;
            }
            orion_qwen_cpu_rmsnorm(v_only_hidden_attn_last, final_norm, d_model, 1e-6f, cpu_stage_last);
            if (!selected_token_logits(blob_dir, lm_head_name, cpu_stage_last, d_model, pair_ids, 2, v_only_hidden_attn_pair_logits)) {
                fprintf(stderr, "FAIL: pair logits failed at v-only hidden_attn stage\n");
                goto fail;
            }

            orion_qwen_cpu_rmsnorm(cpu_ffn_final_last, final_norm, d_model, 1e-6f, cpu_stage_last);
            orion_qwen_cpu_rmsnorm(ane_ffn_final_same_input_last, final_norm, d_model, 1e-6f, hybrid_stage_last);
            if (!selected_token_logits(blob_dir, lm_head_name, cpu_stage_last, d_model, pair_ids, 2, cpu_ffn_final_pair_logits) ||
                !selected_token_logits(blob_dir, lm_head_name, hybrid_stage_last, d_model, pair_ids, 2, ane_ffn_same_input_pair_logits)) {
                fprintf(stderr, "FAIL: pair logits failed at ffn same-input stage\n");
                goto fail;
            }

            orion_qwen_cpu_rmsnorm(hybrid_layer_final_last, final_norm, d_model, 1e-6f, hybrid_stage_last);
            if (!selected_token_logits(blob_dir, lm_head_name, hybrid_stage_last, d_model, pair_ids, 2, hybrid_layer_final_pair_logits)) {
                fprintf(stderr, "FAIL: pair logits failed at hybrid final stage\n");
                goto fail;
            }

            cpu_hidden_attn_pair_gap = (double)cpu_hidden_attn_pair_logits[0] - (double)cpu_hidden_attn_pair_logits[1];
            hybrid_hidden_attn_pair_gap = (double)hybrid_hidden_attn_pair_logits[0] - (double)hybrid_hidden_attn_pair_logits[1];
            gate_only_hidden_attn_pair_gap = (double)gate_only_hidden_attn_pair_logits[0] - (double)gate_only_hidden_attn_pair_logits[1];
            sigmoid_only_hidden_attn_pair_gap = (double)sigmoid_only_hidden_attn_pair_logits[0] - (double)sigmoid_only_hidden_attn_pair_logits[1];
            v_only_hidden_attn_pair_gap = (double)v_only_hidden_attn_pair_logits[0] - (double)v_only_hidden_attn_pair_logits[1];
            cpu_ffn_final_pair_gap = (double)cpu_ffn_final_pair_logits[0] - (double)cpu_ffn_final_pair_logits[1];
            ane_ffn_same_input_pair_gap = (double)ane_ffn_same_input_pair_logits[0] - (double)ane_ffn_same_input_pair_logits[1];
            hybrid_layer_final_pair_gap = (double)hybrid_layer_final_pair_logits[0] - (double)hybrid_layer_final_pair_logits[1];
            cpu_hidden_attn_pref_token = (cpu_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
            hybrid_hidden_attn_pref_token = (hybrid_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
            gate_only_hidden_attn_pref_token = (gate_only_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
            sigmoid_only_hidden_attn_pref_token = (sigmoid_only_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
            v_only_hidden_attn_pref_token = (v_only_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
            cpu_ffn_final_pref_token = (cpu_ffn_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
            ane_ffn_same_input_pref_token = (ane_ffn_same_input_pair_gap >= 0.0) ? candidate_a : candidate_b;
            hybrid_layer_final_pref_token = (hybrid_layer_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
        }

        printf("PASS: qwen35 9b bridge stage diff trace\n");
        printf("  prompt=%s\n", prompt);
        printf("  prompt_len=%d\n", seq_len);
        printf("  target_layer=%d\n", layer);
        printf("  candidate_a=%d\n", candidate_a);
        printf("  candidate_b=%d\n", candidate_b);
        printf("  q_proj_source=%s\n", q_proj_source_label(q_proj_uses_cpu));
        printf("  q_query_source=%s\n", q_query_source_label(q_query_uses_cpu));
        printf("  q_gate_source=%s\n", q_gate_source_label(q_gate_uses_cpu));
        printf("  v_proj_source=%s\n", v_proj_source_label(v_proj_uses_cpu, v_proj_cpu_channel_count));
        printf("  v_proj_cpu_channel_count=%d\n", v_proj_cpu_channel_count);
        printf("  trace_dim_count=%d\n", trace_dim_count);
        printf("  trace_channel_count=%d\n", trace_channel_count);
        for (int i = 0; i < trace_dim_count; i++) {
            printf("  trace_dim_config_rank=%d dim=%d\n", i + 1, trace_dims[i]);
        }
        for (int i = 0; i < trace_channel_count; i++) {
            printf("  trace_channel_config_rank=%d channel=%d\n", i + 1, trace_channels[i]);
        }
        printf("  q_proj_mean_abs_diff=%.6f\n", q_mean);
        printf("  q_proj_max_abs_diff=%.6f\n", q_max);
        printf("  q_query_mean_abs_diff=%.6f\n", query_mean);
        printf("  q_query_max_abs_diff=%.6f\n", query_max);
        printf("  q_gate_mean_abs_diff=%.6f\n", gate_mean);
        printf("  q_gate_max_abs_diff=%.6f\n", gate_max);
        printf("  q_half_swap_mean_abs_diff=%.6f\n", q_half_swap_mean);
        printf("  q_half_swap_max_abs_diff=%.6f\n", q_half_swap_max);
        printf("  k_proj_mean_abs_diff=%.6f\n", k_mean);
        printf("  k_proj_max_abs_diff=%.6f\n", k_max);
        printf("  input_rms_ms_mean_abs_diff=%.6f\n", input_rms_ms_mean);
        printf("  input_rms_ms_max_abs_diff=%.6f\n", input_rms_ms_max);
        printf("  input_rms_rrms_mean_abs_diff=%.6f\n", input_rms_rrms_mean);
        printf("  input_rms_rrms_max_abs_diff=%.6f\n", input_rms_rrms_max);
        printf("  input_rms_rrms_pow_micro_mean_abs_diff=%.6f\n", input_rms_rrms_pow_micro_mean);
        printf("  input_rms_rrms_pow_micro_max_abs_diff=%.6f\n", input_rms_rrms_pow_micro_max);
        printf("  input_rms_rrms_stage_vs_pow_mean_abs_diff=%.6f\n", input_rms_rrms_stage_vs_pow_mean);
        printf("  input_rms_rrms_stage_vs_pow_max_abs_diff=%.6f\n", input_rms_rrms_stage_vs_pow_max);
        printf("  input_rms_xr_mean_abs_diff=%.6f\n", input_rms_xr_mean);
        printf("  input_rms_xr_max_abs_diff=%.6f\n", input_rms_xr_max);
        printf("  input_rms_mean_abs_diff=%.6f\n", input_rms_mean);
        printf("  input_rms_max_abs_diff=%.6f\n", input_rms_max);
        printf("  v_proj_mean_abs_diff=%.6f\n", v_mean);
        printf("  v_proj_max_abs_diff=%.6f\n", v_max);
        printf("  v_proj_rms_only_mean_abs_diff=%.6f\n", v_rms_only_mean);
        printf("  v_proj_rms_only_max_abs_diff=%.6f\n", v_rms_only_max);
        printf("  v_proj_single_mean_abs_diff=%.6f\n", v_single_mean);
        printf("  v_proj_single_max_abs_diff=%.6f\n", v_single_max);
        printf("  v_proj_linear_only_mean_abs_diff=%.6f\n", v_linear_only_mean);
        printf("  v_proj_linear_only_max_abs_diff=%.6f\n", v_linear_only_max);
        printf("  v_proj_input_fp16_cpu_mean_abs_diff=%.6f\n", v_input_fp16_cpu_mean);
        printf("  v_proj_input_fp16_cpu_max_abs_diff=%.6f\n", v_input_fp16_cpu_max);
        printf("  v_proj_input_fp16_ane_mean_abs_diff=%.6f\n", v_input_fp16_ane_mean);
        printf("  v_proj_input_fp16_ane_max_abs_diff=%.6f\n", v_input_fp16_ane_max);
        printf("  v_proj_input_fp16_ane_self_mean_abs_diff=%.6f\n", v_input_fp16_ane_self_mean);
        printf("  v_proj_input_fp16_ane_self_max_abs_diff=%.6f\n", v_input_fp16_ane_self_max);
        printf("  v_proj_fp16emu_cpu_mean_abs_diff=%.6f\n", v_fp16emu_cpu_mean);
        printf("  v_proj_fp16emu_cpu_max_abs_diff=%.6f\n", v_fp16emu_cpu_max);
        printf("  v_proj_fp16emu_ane_mean_abs_diff=%.6f\n", v_fp16emu_ane_mean);
        printf("  v_proj_fp16emu_ane_max_abs_diff=%.6f\n", v_fp16emu_ane_max);
        printf("  v_proj_fp16acc32_cpu_mean_abs_diff=%.6f\n", v_fp16acc32_cpu_mean);
        printf("  v_proj_fp16acc32_cpu_max_abs_diff=%.6f\n", v_fp16acc32_cpu_max);
        printf("  v_proj_fp16acc32_ane_mean_abs_diff=%.6f\n", v_fp16acc32_ane_mean);
        printf("  v_proj_fp16acc32_ane_max_abs_diff=%.6f\n", v_fp16acc32_ane_max);
        printf("  v_proj_multi_vs_single_mean_abs_diff=%.6f\n", v_multi_vs_single_mean);
        printf("  v_proj_multi_vs_single_max_abs_diff=%.6f\n", v_multi_vs_single_max);
        printf("  v_linear_only_tmp_dir=%s\n", v_linear_only_tmp_dir ? v_linear_only_tmp_dir.UTF8String : "");
        printf("  v_linear_only_source_blob_path=%s\n", v_linear_only_source_blob_path ? v_linear_only_source_blob_path.UTF8String : "");
        printf("  v_linear_only_data_path=%s\n", v_linear_only_data_path ? v_linear_only_data_path.UTF8String : "");
        printf("  v_linear_only_weight_path=%s\n", v_linear_only_weight_path ? v_linear_only_weight_path.UTF8String : "");
        printf("  v_linear_only_source_blob_bytes=%lld\n", v_linear_only_data_vs_source.rhs_size);
        printf("  v_linear_only_source_payload_bytes=%lld\n", v_linear_only_data_vs_source_payload.rhs_size);
        printf("  v_linear_only_data_bytes=%lld\n", v_linear_only_data_vs_source.lhs_size);
        printf("  v_linear_only_weight_bytes=%lld\n", v_linear_only_weight_vs_source.lhs_size);
        printf("  v_linear_only_data_matches_source_blob=%d\n", v_linear_only_data_vs_source.equal);
        printf("  v_linear_only_weight_matches_source_blob=%d\n", v_linear_only_weight_vs_source.equal);
        printf("  v_linear_only_data_matches_weight_blob=%d\n", v_linear_only_data_vs_weight.equal);
        printf("  v_linear_only_data_matches_source_payload=%d\n", v_linear_only_data_vs_source_payload.equal);
        printf("  v_linear_only_data_vs_source_first_diff=%lld\n", v_linear_only_data_vs_source.first_diff_offset);
        printf("  v_linear_only_weight_vs_source_first_diff=%lld\n", v_linear_only_weight_vs_source.first_diff_offset);
        printf("  v_linear_only_data_vs_weight_first_diff=%lld\n", v_linear_only_data_vs_weight.first_diff_offset);
        printf("  v_linear_only_data_vs_source_payload_first_diff=%lld\n", v_linear_only_data_vs_source_payload.first_diff_offset);
        printf("  kv_swap_mean_abs_diff=%.6f\n", kv_swap_mean);
        printf("  kv_swap_max_abs_diff=%.6f\n", kv_swap_max);
        printf("  q_token_major_mean_abs_diff=%.6f\n", q_token_major_mean);
        printf("  q_token_major_max_abs_diff=%.6f\n", q_token_major_max);
        printf("  k_token_major_mean_abs_diff=%.6f\n", k_token_major_mean);
        printf("  k_token_major_max_abs_diff=%.6f\n", k_token_major_max);
        printf("  v_token_major_mean_abs_diff=%.6f\n", v_token_major_mean);
        printf("  v_token_major_max_abs_diff=%.6f\n", v_token_major_max);
        printf("  attn_q_norm_mean_abs_diff=%.6f\n", attn_q_norm_mean);
        printf("  attn_q_norm_max_abs_diff=%.6f\n", attn_q_norm_max);
        printf("  attn_gate_sigmoid_mean_abs_diff=%.6f\n", attn_gate_sigmoid_mean);
        printf("  attn_gate_sigmoid_max_abs_diff=%.6f\n", attn_gate_sigmoid_max);
        printf("  attn_k_norm_mean_abs_diff=%.6f\n", attn_k_norm_mean);
        printf("  attn_k_norm_max_abs_diff=%.6f\n", attn_k_norm_max);
        printf("  attn_q_rope_mean_abs_diff=%.6f\n", attn_q_rope_mean);
        printf("  attn_q_rope_max_abs_diff=%.6f\n", attn_q_rope_max);
        printf("  attn_k_rope_mean_abs_diff=%.6f\n", attn_k_rope_mean);
        printf("  attn_k_rope_max_abs_diff=%.6f\n", attn_k_rope_max);
        printf("  attn_score_mean_abs_diff=%.6f\n", attn_score_mean);
        printf("  attn_score_max_abs_diff=%.6f\n", attn_score_max);
        printf("  attn_softmax_mean_abs_diff=%.6f\n", attn_softmax_mean);
        printf("  attn_softmax_max_abs_diff=%.6f\n", attn_softmax_max);
        printf("  attn_context_mean_abs_diff=%.6f\n", attn_context_mean);
        printf("  attn_context_max_abs_diff=%.6f\n", attn_context_max);
        printf("  attn_gated_context_mean_abs_diff=%.6f\n", attn_gated_context_mean);
        printf("  attn_gated_context_max_abs_diff=%.6f\n", attn_gated_context_max);
        printf("  attn_out_mean_abs_diff=%.6f\n", attn_mean);
        printf("  attn_out_max_abs_diff=%.6f\n", attn_max);
        printf("  attn_gate_only_mean_abs_diff=%.6f\n", attn_gate_only_mean);
        printf("  attn_gate_only_max_abs_diff=%.6f\n", attn_gate_only_max);
        printf("  attn_v_only_mean_abs_diff=%.6f\n", attn_v_only_mean);
        printf("  attn_v_only_max_abs_diff=%.6f\n", attn_v_only_max);
        printf("  attn_v_single_only_mean_abs_diff=%.6f\n", attn_v_single_only_mean);
        printf("  attn_v_single_only_max_abs_diff=%.6f\n", attn_v_single_only_max);
        printf("  attn_v_linear_only_mean_abs_diff=%.6f\n", attn_v_linear_only_mean);
        printf("  attn_v_linear_only_max_abs_diff=%.6f\n", attn_v_linear_only_max);
        printf("  attn_sigmoid_only_mean_abs_diff=%.6f\n", attn_sigmoid_only_mean);
        printf("  attn_sigmoid_only_max_abs_diff=%.6f\n", attn_sigmoid_only_max);
        printf("  attn_v_input_fp16_cpu_mean_abs_diff=%.6f\n", attn_v_input_fp16_cpu_mean);
        printf("  attn_v_input_fp16_cpu_max_abs_diff=%.6f\n", attn_v_input_fp16_cpu_max);
        printf("  attn_v_input_fp16_ane_mean_abs_diff=%.6f\n", attn_v_input_fp16_ane_mean);
        printf("  attn_v_input_fp16_ane_max_abs_diff=%.6f\n", attn_v_input_fp16_ane_max);
        printf("  attn_v_input_fp16_ane_self_mean_abs_diff=%.6f\n", attn_v_input_fp16_ane_self_mean);
        printf("  attn_v_input_fp16_ane_self_max_abs_diff=%.6f\n", attn_v_input_fp16_ane_self_max);
        printf("  attn_v_fp16emu_cpu_mean_abs_diff=%.6f\n", attn_v_fp16emu_cpu_mean);
        printf("  attn_v_fp16emu_cpu_max_abs_diff=%.6f\n", attn_v_fp16emu_cpu_max);
        printf("  attn_v_fp16emu_ane_mean_abs_diff=%.6f\n", attn_v_fp16emu_ane_mean);
        printf("  attn_v_fp16emu_ane_max_abs_diff=%.6f\n", attn_v_fp16emu_ane_max);
        printf("  attn_v_fp16acc32_cpu_mean_abs_diff=%.6f\n", attn_v_fp16acc32_cpu_mean);
        printf("  attn_v_fp16acc32_cpu_max_abs_diff=%.6f\n", attn_v_fp16acc32_cpu_max);
        printf("  attn_v_fp16acc32_ane_mean_abs_diff=%.6f\n", attn_v_fp16acc32_ane_mean);
        printf("  attn_v_fp16acc32_ane_max_abs_diff=%.6f\n", attn_v_fp16acc32_ane_max);
        printf("  attn_v_rms_only_mean_abs_diff=%.6f\n", attn_v_rms_only_mean);
        printf("  attn_v_rms_only_max_abs_diff=%.6f\n", attn_v_rms_only_max);
        printf("  hidden_attn_mean_abs_diff=%.6f\n", hidden_attn_mean);
        printf("  hidden_attn_max_abs_diff=%.6f\n", hidden_attn_max);
        printf("  ffn_rms_mean_abs_diff=%.6f\n", ffn_rms_mean);
        printf("  ffn_rms_max_abs_diff=%.6f\n", ffn_rms_max);
        printf("  ffn_gate_mean_abs_diff=%.6f\n", ffn_gate_mean);
        printf("  ffn_gate_max_abs_diff=%.6f\n", ffn_gate_max);
        printf("  ffn_up_mean_abs_diff=%.6f\n", ffn_up_mean);
        printf("  ffn_up_max_abs_diff=%.6f\n", ffn_up_max);
        printf("  ffn_up_only_mean_abs_diff=%.6f\n", ffn_up_only_mean);
        printf("  ffn_up_only_max_abs_diff=%.6f\n", ffn_up_only_max);
        printf("  ffn_sigmoid_micro_mean_abs_diff=%.6f\n", ffn_sigmoid_micro_mean);
        printf("  ffn_sigmoid_micro_max_abs_diff=%.6f\n", ffn_sigmoid_micro_max);
        printf("  ffn_sigmoid_tanh_micro_mean_abs_diff=%.6f\n", ffn_sigmoid_tanh_micro_mean);
        printf("  ffn_sigmoid_tanh_micro_max_abs_diff=%.6f\n", ffn_sigmoid_tanh_micro_max);
        printf("  ffn_sigmoid_fp16emu_cpu_mean_abs_diff=%.6f\n", ffn_sigmoid_fp16emu_cpu_mean);
        printf("  ffn_sigmoid_fp16emu_cpu_max_abs_diff=%.6f\n", ffn_sigmoid_fp16emu_cpu_max);
        printf("  ffn_sigmoid_fp16emu_ane_mean_abs_diff=%.6f\n", ffn_sigmoid_fp16emu_ane_mean);
        printf("  ffn_sigmoid_fp16emu_ane_max_abs_diff=%.6f\n", ffn_sigmoid_fp16emu_ane_max);
        printf("  ffn_silu_mean_abs_diff=%.6f\n", ffn_silu_mean);
        printf("  ffn_silu_max_abs_diff=%.6f\n", ffn_silu_max);
        printf("  ffn_silu_only_mean_abs_diff=%.6f\n", ffn_silu_only_mean);
        printf("  ffn_silu_only_max_abs_diff=%.6f\n", ffn_silu_only_max);
        printf("  ffn_silu_micro_mean_abs_diff=%.6f\n", ffn_silu_micro_mean);
        printf("  ffn_silu_micro_max_abs_diff=%.6f\n", ffn_silu_micro_max);
        printf("  ffn_silu_tanh_micro_mean_abs_diff=%.6f\n", ffn_silu_tanh_micro_mean);
        printf("  ffn_silu_tanh_micro_max_abs_diff=%.6f\n", ffn_silu_tanh_micro_max);
        printf("  ffn_silu_fp16emu_cpu_mean_abs_diff=%.6f\n", ffn_silu_fp16emu_cpu_mean);
        printf("  ffn_silu_fp16emu_cpu_max_abs_diff=%.6f\n", ffn_silu_fp16emu_cpu_max);
        printf("  ffn_silu_fp16emu_ane_mean_abs_diff=%.6f\n", ffn_silu_fp16emu_ane_mean);
        printf("  ffn_silu_fp16emu_ane_max_abs_diff=%.6f\n", ffn_silu_fp16emu_ane_max);
        printf("  ffn_hidden_mean_abs_diff=%.6f\n", ffn_hidden_mean);
        printf("  ffn_hidden_max_abs_diff=%.6f\n", ffn_hidden_max);
        printf("  ffn_hidden_only_mean_abs_diff=%.6f\n", ffn_hidden_only_mean);
        printf("  ffn_hidden_only_max_abs_diff=%.6f\n", ffn_hidden_only_max);
        printf("  ffn_mul_micro_mean_abs_diff=%.6f\n", ffn_mul_micro_mean);
        printf("  ffn_mul_micro_max_abs_diff=%.6f\n", ffn_mul_micro_max);
        printf("  ffn_hidden_fp16emu_cpu_mean_abs_diff=%.6f\n", ffn_hidden_fp16emu_cpu_mean);
        printf("  ffn_hidden_fp16emu_cpu_max_abs_diff=%.6f\n", ffn_hidden_fp16emu_cpu_max);
        printf("  ffn_hidden_fp16emu_ane_mean_abs_diff=%.6f\n", ffn_hidden_fp16emu_ane_mean);
        printf("  ffn_hidden_fp16emu_ane_max_abs_diff=%.6f\n", ffn_hidden_fp16emu_ane_max);
        printf("  ffn_down_same_input_mean_abs_diff=%.6f\n", ffn_down_same_input_mean);
        printf("  ffn_down_same_input_max_abs_diff=%.6f\n", ffn_down_same_input_max);
        printf("  ffn_same_input_mean_abs_diff=%.6f\n", ffn_same_input_mean);
        printf("  ffn_same_input_max_abs_diff=%.6f\n", ffn_same_input_max);
        printf("  full_layer_mean_abs_diff=%.6f\n", full_layer_mean);
        printf("  full_layer_max_abs_diff=%.6f\n", full_layer_max);
        if (pair_enabled) {
            printf("  cpu_hidden_attn_pair_gap=%.6f\n", cpu_hidden_attn_pair_gap);
            printf("  hybrid_hidden_attn_pair_gap=%.6f\n", hybrid_hidden_attn_pair_gap);
            printf("  gate_only_hidden_attn_pair_gap=%.6f\n", gate_only_hidden_attn_pair_gap);
            printf("  sigmoid_only_hidden_attn_pair_gap=%.6f\n", sigmoid_only_hidden_attn_pair_gap);
            printf("  v_only_hidden_attn_pair_gap=%.6f\n", v_only_hidden_attn_pair_gap);
            printf("  cpu_hidden_attn_pref_token=%d\n", cpu_hidden_attn_pref_token);
            printf("  hybrid_hidden_attn_pref_token=%d\n", hybrid_hidden_attn_pref_token);
            printf("  gate_only_hidden_attn_pref_token=%d\n", gate_only_hidden_attn_pref_token);
            printf("  sigmoid_only_hidden_attn_pref_token=%d\n", sigmoid_only_hidden_attn_pref_token);
            printf("  v_only_hidden_attn_pref_token=%d\n", v_only_hidden_attn_pref_token);
            printf("  cpu_ffn_final_pair_gap=%.6f\n", cpu_ffn_final_pair_gap);
            printf("  ane_ffn_same_input_pair_gap=%.6f\n", ane_ffn_same_input_pair_gap);
            printf("  hybrid_layer_final_pair_gap=%.6f\n", hybrid_layer_final_pair_gap);
            printf("  cpu_ffn_final_pref_token=%d\n", cpu_ffn_final_pref_token);
            printf("  ane_ffn_same_input_pref_token=%d\n", ane_ffn_same_input_pref_token);
            printf("  hybrid_layer_final_pref_token=%d\n", hybrid_layer_final_pref_token);
        }
        print_trace_stage_rows("input_rms_xr", cpu_input_rms_xr, ane_input_rms_xr, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("input_rms_out", normed, ane_input_rms, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("attn_out", cpu_attn, ane_attn, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("hidden_attn", cpu_hidden_attn, ane_hidden_attn, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("ffn_rms", cpu_ffn_rms, ane_ffn_rms, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("ffn_down_same_input", cpu_ffn_down, ane_ffn_down_same_input, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("ffn_final_same_input", cpu_ffn_final, ane_ffn_final_same_input, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("full_layer_final", cpu_ffn_final, hybrid_layer_final, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("attn_gate_only", cpu_attn, attn_gate_only, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("attn_v_only", cpu_attn, attn_v_only, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("attn_v_single_only", cpu_attn, attn_v_single_only, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("attn_v_linear_only", cpu_attn, attn_v_linear_only, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_stage_rows("attn_v_rms_only", cpu_attn, attn_v_rms_only, seq_len, d_model, trace_dims, trace_dim_count);
        print_trace_attn_attr_rows(cpu_attn_gated_context, ane_attn_gated_context, o_proj,
                                   seq_len, q_dim, head_dim, trace_dims, trace_dim_count);
        print_trace_attn_channel_rows(cpu_q, ane_q, cpu_v, cpu_v_rms_only, ane_v, ane_v_single, ane_v_linear_only,
                                      cpu_attn_q_normed, ane_attn_q_normed,
                                      cpu_attn_gate_sigmoid, ane_attn_gate_sigmoid,
                                      cpu_attn_q_rope, ane_attn_q_rope,
                                      cpu_attn_context, ane_attn_context,
                                      cpu_attn_gated_context, ane_attn_gated_context,
                                      seq_len, q_dim, kv_dim, head_dim, q_per_kv,
                                      trace_channels, trace_channel_count);
        print_trace_v_neighbor_rows("v_proj", cpu_v, ane_v, seq_len, kv_dim, head_dim, q_per_kv, trace_channels, trace_channel_count);
        print_trace_v_neighbor_rows("v_proj_linear_only", cpu_v, ane_v_linear_only, seq_len, kv_dim, head_dim, q_per_kv, trace_channels, trace_channel_count);

        if (ioIn) CFRelease(ioIn);
        if (ioQ) CFRelease(ioQ);
        if (ioK) CFRelease(ioK);
        if (ioV) CFRelease(ioV);
        if (ioVSingle) CFRelease(ioVSingle);
        if (ioNormedIn) CFRelease(ioNormedIn);
        if (ioNormedFp16In) CFRelease(ioNormedFp16In);
        if (ioInputRmsOnly) CFRelease(ioInputRmsOnly);
        if (ioInputRmsMs) CFRelease(ioInputRmsMs);
        if (ioInputRmsRrms) CFRelease(ioInputRmsRrms);
        if (ioInputRmsMseIn) CFRelease(ioInputRmsMseIn);
        if (ioInputRmsRrmsPowMicro) CFRelease(ioInputRmsRrmsPowMicro);
        if (ioInputRmsXr) CFRelease(ioInputRmsXr);
        if (ioVLinearOnly) CFRelease(ioVLinearOnly);
        if (ioVLinearOnlyFp16In) CFRelease(ioVLinearOnlyFp16In);
        if (ioFfnIn) CFRelease(ioFfnIn);
        if (ioFfnRms) CFRelease(ioFfnRms);
        if (ioFfnGate) CFRelease(ioFfnGate);
        if (ioFfnUp) CFRelease(ioFfnUp);
        if (ioFfnUpOnly) CFRelease(ioFfnUpOnly);
        if (ioFfnSigmoidMicroIn) CFRelease(ioFfnSigmoidMicroIn);
        if (ioFfnSigmoidMicroOut) CFRelease(ioFfnSigmoidMicroOut);
        if (ioFfnSigmoidTanhMicroOut) CFRelease(ioFfnSigmoidTanhMicroOut);
        if (ioFfnSilu) CFRelease(ioFfnSilu);
        if (ioFfnSiluOnly) CFRelease(ioFfnSiluOnly);
        if (ioFfnSiluMicroIn) CFRelease(ioFfnSiluMicroIn);
        if (ioFfnSiluMicroOut) CFRelease(ioFfnSiluMicroOut);
        if (ioFfnSiluTanhMicroOut) CFRelease(ioFfnSiluTanhMicroOut);
        if (ioFfnHidden) CFRelease(ioFfnHidden);
        if (ioFfnHiddenOnly) CFRelease(ioFfnHiddenOnly);
        if (ioFfnMulMicroA) CFRelease(ioFfnMulMicroA);
        if (ioFfnMulMicroB) CFRelease(ioFfnMulMicroB);
        if (ioFfnMulMicroOut) CFRelease(ioFfnMulMicroOut);
        if (ioFfnDownIn) CFRelease(ioFfnDownIn);
        if (ioFfnDown) CFRelease(ioFfnDown);
        if (ioHidden) CFRelease(ioHidden);
        if (ioFfnInMixed) CFRelease(ioFfnInMixed);
        if (ioHiddenMixed) CFRelease(ioHiddenMixed);
        bridge_release(&bridge);
        free(input_ln); free(post_ln); free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
        free(gate_proj); free(up_proj); free(down_proj);
        free(final_norm); free(cpu_stage_last); free(hybrid_stage_last);
        free(hidden); free(hidden_next); free(normed); free(normed_fp16input); free(cpu_q); free(cpu_k); free(cpu_v); free(cpu_v_rms_only); free(cpu_v_linear_fp16emu); free(cpu_v_linear_fp16acc32); free(cpu_v_input_fp16);
        free(cpu_input_rms_ms); free(cpu_input_rms_mse); free(cpu_input_rms_rrms); free(cpu_input_rms_xr);
        free(ane_q); free(ane_k); free(ane_v); free(ane_v_single); free(ane_input_rms);
        free(ane_input_rms_ms); free(ane_input_rms_rrms); free(ane_input_rms_rrms_pow_micro); free(ane_input_rms_xr);
        free(ane_v_linear_only); free(ane_v_linear_only_fp16input);
        free(ane_q_token_major); free(ane_k_token_major); free(ane_v_token_major);
        free(cpu_attn); free(ane_attn); free(attn_gate_only); free(attn_v_only); free(attn_v_single_only);
        free(attn_v_linear_only); free(attn_v_linear_fp16emu); free(attn_v_linear_fp16acc32); free(attn_v_input_fp16_cpu); free(attn_v_input_fp16_ane); free(attn_v_rms_only); free(sigmoid_only_gated_context); free(attn_sigmoid_only); free(cpu_hidden_attn); free(ane_hidden_attn); free(gate_only_hidden_attn); free(sigmoid_only_hidden_attn); free(v_only_hidden_attn);
        free(cpu_attn_q_normed); free(ane_attn_q_normed); free(cpu_attn_gate_sigmoid); free(ane_attn_gate_sigmoid);
        free(cpu_attn_k_normed); free(ane_attn_k_normed); free(cpu_attn_q_rope); free(ane_attn_q_rope);
        free(cpu_attn_k_rope); free(ane_attn_k_rope); free(cpu_attn_scores); free(ane_attn_scores);
        free(cpu_attn_probs); free(ane_attn_probs); free(cpu_attn_context); free(ane_attn_context);
        free(cpu_attn_gated_context); free(ane_attn_gated_context);
        free(cpu_ffn_rms); free(ane_ffn_rms); free(cpu_ffn_gate); free(ane_ffn_gate);
        free(cpu_ffn_up); free(ane_ffn_up); free(ane_ffn_up_only);
        free(cpu_ffn_sigmoid); free(cpu_ffn_sigmoid_fp16emu); free(ane_ffn_sigmoid_micro); free(ane_ffn_sigmoid_tanh_micro);
        free(cpu_ffn_silu); free(cpu_ffn_silu_fp16emu); free(ane_ffn_silu); free(ane_ffn_silu_only); free(ane_ffn_silu_micro); free(ane_ffn_silu_tanh_micro);
        free(cpu_ffn_hidden); free(cpu_ffn_hidden_fp16emu); free(ane_ffn_hidden); free(ane_ffn_hidden_only); free(ane_ffn_mul_micro);
        free(cpu_ffn_down); free(ane_ffn_down_same_input);
        free(cpu_ffn_final); free(ane_ffn_final_same_input); free(hybrid_layer_final);
        orion_qwen35_manifest_free(manifest);
        orion_gpt2_tokenizer_free(tok);
        return 0;

fail:
        if (ioIn) CFRelease(ioIn);
        if (ioQ) CFRelease(ioQ);
        if (ioK) CFRelease(ioK);
        if (ioV) CFRelease(ioV);
        if (ioVSingle) CFRelease(ioVSingle);
        if (ioNormedIn) CFRelease(ioNormedIn);
        if (ioNormedFp16In) CFRelease(ioNormedFp16In);
        if (ioInputRmsOnly) CFRelease(ioInputRmsOnly);
        if (ioInputRmsMs) CFRelease(ioInputRmsMs);
        if (ioInputRmsRrms) CFRelease(ioInputRmsRrms);
        if (ioInputRmsMseIn) CFRelease(ioInputRmsMseIn);
        if (ioInputRmsRrmsPowMicro) CFRelease(ioInputRmsRrmsPowMicro);
        if (ioInputRmsXr) CFRelease(ioInputRmsXr);
        if (ioVLinearOnly) CFRelease(ioVLinearOnly);
        if (ioVLinearOnlyFp16In) CFRelease(ioVLinearOnlyFp16In);
        if (ioFfnIn) CFRelease(ioFfnIn);
        if (ioFfnRms) CFRelease(ioFfnRms);
        if (ioFfnGate) CFRelease(ioFfnGate);
        if (ioFfnUp) CFRelease(ioFfnUp);
        if (ioFfnUpOnly) CFRelease(ioFfnUpOnly);
        if (ioFfnSigmoidMicroIn) CFRelease(ioFfnSigmoidMicroIn);
        if (ioFfnSigmoidMicroOut) CFRelease(ioFfnSigmoidMicroOut);
        if (ioFfnSigmoidTanhMicroOut) CFRelease(ioFfnSigmoidTanhMicroOut);
        if (ioFfnSilu) CFRelease(ioFfnSilu);
        if (ioFfnSiluOnly) CFRelease(ioFfnSiluOnly);
        if (ioFfnSiluMicroIn) CFRelease(ioFfnSiluMicroIn);
        if (ioFfnSiluMicroOut) CFRelease(ioFfnSiluMicroOut);
        if (ioFfnSiluTanhMicroOut) CFRelease(ioFfnSiluTanhMicroOut);
        if (ioFfnHidden) CFRelease(ioFfnHidden);
        if (ioFfnHiddenOnly) CFRelease(ioFfnHiddenOnly);
        if (ioFfnMulMicroA) CFRelease(ioFfnMulMicroA);
        if (ioFfnMulMicroB) CFRelease(ioFfnMulMicroB);
        if (ioFfnMulMicroOut) CFRelease(ioFfnMulMicroOut);
        if (ioFfnDownIn) CFRelease(ioFfnDownIn);
        if (ioFfnDown) CFRelease(ioFfnDown);
        if (ioHidden) CFRelease(ioHidden);
        if (ioFfnInMixed) CFRelease(ioFfnInMixed);
        if (ioHiddenMixed) CFRelease(ioHiddenMixed);
        bridge_release(&bridge);
        free(input_ln); free(post_ln); free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
        free(gate_proj); free(up_proj); free(down_proj);
        free(final_norm); free(cpu_stage_last); free(hybrid_stage_last);
        free(hidden); free(hidden_next); free(normed); free(normed_fp16input); free(cpu_q); free(cpu_k); free(cpu_v); free(cpu_v_rms_only); free(cpu_v_linear_fp16emu); free(cpu_v_linear_fp16acc32); free(cpu_v_input_fp16);
        free(cpu_input_rms_ms); free(cpu_input_rms_mse); free(cpu_input_rms_rrms); free(cpu_input_rms_xr);
        free(ane_q); free(ane_k); free(ane_v); free(ane_v_single); free(ane_input_rms);
        free(ane_input_rms_ms); free(ane_input_rms_rrms); free(ane_input_rms_rrms_pow_micro); free(ane_input_rms_xr);
        free(ane_v_linear_only); free(ane_v_linear_only_fp16input);
        free(ane_q_token_major); free(ane_k_token_major); free(ane_v_token_major);
        free(cpu_attn); free(ane_attn); free(attn_gate_only); free(attn_v_only); free(attn_v_single_only);
        free(attn_v_linear_only); free(attn_v_linear_fp16emu); free(attn_v_linear_fp16acc32); free(attn_v_input_fp16_cpu); free(attn_v_input_fp16_ane); free(attn_v_rms_only); free(sigmoid_only_gated_context); free(attn_sigmoid_only); free(cpu_hidden_attn); free(ane_hidden_attn); free(gate_only_hidden_attn); free(sigmoid_only_hidden_attn); free(v_only_hidden_attn);
        free(cpu_attn_q_normed); free(ane_attn_q_normed); free(cpu_attn_gate_sigmoid); free(ane_attn_gate_sigmoid);
        free(cpu_attn_k_normed); free(ane_attn_k_normed); free(cpu_attn_q_rope); free(ane_attn_q_rope);
        free(cpu_attn_k_rope); free(ane_attn_k_rope); free(cpu_attn_scores); free(ane_attn_scores);
        free(cpu_attn_probs); free(ane_attn_probs); free(cpu_attn_context); free(ane_attn_context);
        free(cpu_attn_gated_context); free(ane_attn_gated_context);
        free(cpu_ffn_rms); free(ane_ffn_rms); free(cpu_ffn_gate); free(ane_ffn_gate);
        free(cpu_ffn_up); free(ane_ffn_up); free(ane_ffn_up_only);
        free(cpu_ffn_sigmoid); free(cpu_ffn_sigmoid_fp16emu); free(ane_ffn_sigmoid_micro); free(ane_ffn_sigmoid_tanh_micro);
        free(cpu_ffn_silu); free(cpu_ffn_silu_fp16emu); free(ane_ffn_silu); free(ane_ffn_silu_only); free(ane_ffn_silu_micro); free(ane_ffn_silu_tanh_micro);
        free(cpu_ffn_hidden); free(cpu_ffn_hidden_fp16emu); free(ane_ffn_hidden); free(ane_ffn_hidden_only); free(ane_ffn_mul_micro);
        free(cpu_ffn_down); free(ane_ffn_down_same_input);
        free(cpu_ffn_final); free(ane_ffn_final_same_input); free(hybrid_layer_final);
        if (manifest) orion_qwen35_manifest_free(manifest);
        if (tok) orion_gpt2_tokenizer_free(tok);
        return 1;
    }
}
