#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "compiler/frontends/qwen35_prefill.h"
#include "compiler/model_config.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#import "compiler/codegen.h"
#include "core/ane_runtime.h"
#include "core/bucket.h"
#import "../model/weight_loader.h"
#import "../tokenizer/gpt2_bpe.h"
#import "../kernels/inference/qwen_cpu_ops.h"

static const int kQwenPrefillBuckets[] = {32, 64, 128, 256, 512, 1024};
static const int kQwenPrefillNumBuckets = (int)(sizeof(kQwenPrefillBuckets) / sizeof(kQwenPrefillBuckets[0]));

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

static double abs_sum(const float *x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)x[i]);
    return total;
}

static void cpu_linear_batch(const float *x_seq,
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
            for (int i = 0; i < in_dim; i++) acc += x[i] * w[i];
            out[o] = acc;
        }
    }
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

static NSDictionary *build_qproj_linear_only_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
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

static NSDictionary *build_kv_linear_only_wdict(int layer, NSString *blob_dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    NSString *prefix = [NSString stringWithFormat:@"%@/layer%d", blob_dir, layer];
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

typedef struct {
    OrionProgram *prog_q;
    OrionProgram *prog_kv;
    OrionProgram *prog_ffn;
    int layer_idx;
    int bucket;
    int d_model;
    int q_dim;
    int kv_dim;
    int qkv_input_mode;
    int q_uses_cpu_rms;
    int kv_uses_cpu_rms;
    int q_proj_uses_cpu;
    int k_proj_uses_cpu;
    int v_proj_uses_cpu;
    int q_query_uses_cpu;
    int q_gate_uses_cpu;
    int ffn_uses_cpu;
    int q_gate_cpu_channel_count;
    int q_gate_cpu_channels[32];
    int v_proj_cpu_channel_count;
    int v_proj_cpu_channels[32];
} OrionQwen35AneBridge;

enum {
    ORION_QKV_INPUT_MODE_ANE_RMS = 0,
    ORION_QKV_INPUT_MODE_CPU_RMS = 1,
    ORION_QKV_INPUT_MODE_KV_CPU_RMS = 2,
};

static int parse_index_list_local(const char *csv, int *out, int max_count, int limit) {
    if (!csv || !*csv || !out || max_count <= 0) return 0;
    char *copy = strdup(csv);
    if (!copy) return 0;
    int count = 0;
    char *save = NULL;
    for (char *tok = strtok_r(copy, ",", &save); tok && count < max_count; tok = strtok_r(NULL, ",", &save)) {
        while (*tok && isspace((unsigned char)*tok)) tok++;
        if (!*tok) continue;
        char *end = NULL;
        long value = strtol(tok, &end, 10);
        if (end == tok || value < 0 || value >= limit) continue;
        int dup = 0;
        for (int i = 0; i < count; i++) {
            if (out[i] == (int)value) {
                dup = 1;
                break;
            }
        }
        if (!dup) out[count++] = (int)value;
    }
    free(copy);
    return count;
}

static int override_enabled_for_layer(const char *env_name, int layer, int layer_limit) {
    const char *csv = getenv(env_name);
    if (!csv || !*csv) return 1;
    int layers[64] = {0};
    int count = parse_index_list_local(csv, layers, (int)(sizeof(layers) / sizeof(layers[0])), layer_limit);
    if (count <= 0) return 1;
    for (int i = 0; i < count; i++) {
        if (layers[i] == layer) return 1;
    }
    return 0;
}

static int load_qkv_input_mode(void) {
    const char *mode = getenv("ORION_QKV_INPUT_MODE");
    if (!mode || !*mode) return ORION_QKV_INPUT_MODE_ANE_RMS;
    if (strcmp(mode, "cpu_rms") == 0) return ORION_QKV_INPUT_MODE_CPU_RMS;
    if (strcmp(mode, "kv_cpu_rms") == 0) return ORION_QKV_INPUT_MODE_KV_CPU_RMS;
    return ORION_QKV_INPUT_MODE_ANE_RMS;
}

static int q_uses_cpu_rms_mode(int qkv_input_mode) {
    return qkv_input_mode == ORION_QKV_INPUT_MODE_CPU_RMS;
}

static int kv_uses_cpu_rms_mode(int qkv_input_mode) {
    return qkv_input_mode == ORION_QKV_INPUT_MODE_CPU_RMS ||
           qkv_input_mode == ORION_QKV_INPUT_MODE_KV_CPU_RMS;
}

static const char *qkv_input_mode_label(int qkv_input_mode) {
    switch (qkv_input_mode) {
        case ORION_QKV_INPUT_MODE_CPU_RMS:
            return "cpu_rms_linear_only";
        case ORION_QKV_INPUT_MODE_KV_CPU_RMS:
            return "kv_cpu_rms_linear_only";
        default:
            return "ane_rms_plus_linear";
    }
}

static int use_cpu_v_proj_override(void) {
    const char *full = getenv("ORION_FULL_ATTN_SOURCE");
    if (full && strcmp(full, "cpu") == 0) return 1;
    const char *source = getenv("ORION_V_PROJ_SOURCE");
    return source && strcmp(source, "cpu") == 0;
}

static int use_cpu_k_proj_override(void) {
    const char *full = getenv("ORION_FULL_ATTN_SOURCE");
    if (full && strcmp(full, "cpu") == 0) return 1;
    const char *source = getenv("ORION_K_PROJ_SOURCE");
    return source && strcmp(source, "cpu") == 0;
}

static int use_matmul_v_proj_mode(void) {
    const char *source = getenv("ORION_V_PROJ_SOURCE");
    return source && strcmp(source, "matmul") == 0;
}

static int load_v_proj_cpu_channel_override(int *out_channels, int max_channels, int q_dim) {
    const char *csv = getenv("ORION_V_PROJ_CPU_CHANNELS");
    if (!csv || !*csv) return 0;
    return parse_index_list_local(csv, out_channels, max_channels, q_dim);
}

static int use_cpu_q_proj_override(void) {
    const char *full = getenv("ORION_FULL_ATTN_SOURCE");
    if (full && strcmp(full, "cpu") == 0) return 1;
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

static int use_cpu_ffn_override(void) {
    const char *full = getenv("ORION_FULL_ATTN_SOURCE");
    if (full && strcmp(full, "cpu") == 0) return 1;
    const char *source = getenv("ORION_FFN_SOURCE");
    return source && strcmp(source, "cpu") == 0;
}

static int load_q_gate_cpu_channel_preset(const char *source, int *out_channels, int max_channels, int q_dim) {
    static const int kSeed2[] = {143, 3994};
    static const int kSeed4[] = {143, 3994, 3768, 1353};
    const int *preset = NULL;
    int preset_count = 0;
    if (!source || !out_channels || max_channels <= 0) return 0;
    if (strcmp(source, "seed2") == 0) {
        preset = kSeed2;
        preset_count = (int)(sizeof(kSeed2) / sizeof(kSeed2[0]));
    } else if (strcmp(source, "seed4") == 0) {
        preset = kSeed4;
        preset_count = (int)(sizeof(kSeed4) / sizeof(kSeed4[0]));
    }
    if (!preset) return 0;
    int count = 0;
    for (int i = 0; i < preset_count && count < max_channels; i++) {
        int channel = preset[i];
        if (channel < 0 || channel >= q_dim) continue;
        out_channels[count++] = channel;
    }
    return count;
}

static int load_q_gate_cpu_channel_override(int *out_channels, int max_channels, int q_dim) {
    const char *source = getenv("ORION_Q_GATE_SOURCE");
    int preset_count = load_q_gate_cpu_channel_preset(source, out_channels, max_channels, q_dim);
    if (preset_count > 0) return preset_count;
    const char *csv = getenv("ORION_Q_GATE_CPU_CHANNELS");
    if (!csv || !*csv) return 0;
    return parse_index_list_local(csv, out_channels, max_channels, q_dim);
}

static const char *q_proj_source_label(int q_proj_uses_cpu) {
    return q_proj_uses_cpu ? "cpu_linear" : "ane_linear";
}

static const char *q_query_source_label(int q_query_uses_cpu) {
    return q_query_uses_cpu ? "cpu_linear" : "ane_linear";
}

static const char *k_proj_source_label(int k_proj_uses_cpu) {
    return k_proj_uses_cpu ? "cpu_linear" : "ane_linear";
}

static const char *q_gate_source_label(const char *source, int q_gate_uses_cpu, int partial_channel_count) {
    if (q_gate_uses_cpu) return "cpu_linear";
    if (source && strcmp(source, "seed2") == 0) return "ane_linear_seed2";
    if (source && strcmp(source, "seed4") == 0) return "ane_linear_seed4";
    if (partial_channel_count > 0) return "ane_linear_cpu_channels";
    return "ane_linear";
}

static const char *ffn_source_label(int ffn_uses_cpu) {
    return ffn_uses_cpu ? "cpu" : "ane";
}

static const char *v_proj_source_label(int v_proj_uses_cpu, int partial_channel_count) {
    if (v_proj_uses_cpu) return "cpu_linear";
    if (partial_channel_count > 0) return "ane_linear_cpu_channels";
    if (use_matmul_v_proj_mode()) return "ane_matmul";
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

static void apply_q_gate_cpu_channel_overrides(float *dst_q_proj_seq,
                                               const float *cpu_q_proj_seq,
                                               int seq_len,
                                               int q_dim,
                                               const int *channels,
                                               int channel_count) {
    if (!dst_q_proj_seq || !cpu_q_proj_seq || !channels || channel_count <= 0 || seq_len <= 0 || q_dim <= 0) {
        return;
    }
    for (int i = 0; i < channel_count; i++) {
        int channel = channels[i];
        if (channel < 0 || channel >= q_dim) continue;
        for (int s = 0; s < seq_len; s++) {
            dst_q_proj_seq[(size_t)s * (q_dim * 2) + q_dim + channel] =
                cpu_q_proj_seq[(size_t)s * (q_dim * 2) + q_dim + channel];
        }
    }
}

static int load_full_attention_layers(OrionQwen35Manifest *manifest, int *out_layers, int max_layers) {
    if (!manifest || !manifest->manifest_path || !out_layers || max_layers <= 0) return 0;
    NSString *manifestPath = [NSString stringWithUTF8String:manifest->manifest_path];
    NSDictionary *root = load_json(manifestPath);
    NSDictionary *runtime = [root isKindOfClass:[NSDictionary class]] ? root[@"runtime"] : nil;
    NSArray *layerTypes = [runtime isKindOfClass:[NSDictionary class]] ? runtime[@"layer_types"] : nil;
    if (![layerTypes isKindOfClass:[NSArray class]]) return 0;

    int count = 0;
    for (NSInteger i = 0; i < [layerTypes count] && count < max_layers; i++) {
        id value = layerTypes[i];
        if ([value isKindOfClass:[NSString class]] && [(NSString *)value isEqualToString:@"full_attention"]) {
            out_layers[count++] = (int)i;
        }
    }
    return count;
}

static void bridge_release(OrionQwen35AneBridge *bridge) {
    if (!bridge) return;
    if (bridge->prog_q) orion_release_program(bridge->prog_q);
    if (bridge->prog_kv) orion_release_program(bridge->prog_kv);
    if (bridge->prog_ffn) orion_release_program(bridge->prog_ffn);
    memset(bridge, 0, sizeof(*bridge));
}

static int bridge_init(OrionQwen35AneBridge *bridge,
                       NSString *blobDir,
                       int layer,
                       int bucket,
                       OrionQwen35Manifest *manifest,
                       int qkv_input_mode,
                       int q_proj_uses_cpu,
                       int k_proj_uses_cpu,
                       int v_proj_uses_cpu,
                       int q_query_uses_cpu,
                       int q_gate_uses_cpu,
                       int ffn_uses_cpu) {
    memset(bridge, 0, sizeof(*bridge));
    const int q_uses_cpu_rms = q_uses_cpu_rms_mode(qkv_input_mode);
    const int kv_uses_cpu_rms = kv_uses_cpu_rms_mode(qkv_input_mode);
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
    NSString *mil_q = compile_graph(
        q_uses_cpu_rms
            ? orion_frontend_qwen35_prefill_q_proj_linear_only(layer, bucket, &cfg)
            : orion_frontend_qwen35_prefill_q_proj(layer, bucket, &cfg)
    );
    NSString *mil_kv = compile_graph(
        kv_uses_cpu_rms
            ? orion_frontend_qwen35_prefill_kv_proj_linear_only(layer, bucket, &cfg)
            : orion_frontend_qwen35_prefill_kv_proj(layer, bucket, &cfg)
    );
    NSString *mil_ffn = compile_graph(orion_frontend_qwen35_prefill_ffn(layer, bucket, &cfg));
    if (!mil_q || !mil_kv || !mil_ffn) return 0;

    bridge->prog_q = orion_compile_mil(
        mil_q.UTF8String,
        q_uses_cpu_rms ? build_qproj_linear_only_wdict(layer, blobDir) : build_qproj_wdict(layer, blobDir),
        "qwen35_9b_decode_q"
    );
    bridge->prog_kv = orion_compile_mil(
        mil_kv.UTF8String,
        kv_uses_cpu_rms ? build_kv_linear_only_wdict(layer, blobDir) : build_kv_wdict(layer, blobDir),
        "qwen35_9b_decode_kv"
    );
    bridge->prog_ffn = orion_compile_mil(mil_ffn.UTF8String, build_ffn_wdict(layer, blobDir), "qwen35_9b_decode_ffn");
    if (!bridge->prog_q || !bridge->prog_kv || !bridge->prog_ffn) {
        bridge_release(bridge);
        return 0;
    }

    bridge->layer_idx = layer;
    bridge->bucket = bucket;
    bridge->d_model = manifest->d_model;
    bridge->q_dim = manifest->n_head * manifest->head_dim;
    bridge->kv_dim = manifest->n_kv_head * manifest->head_dim;
    bridge->qkv_input_mode = qkv_input_mode;
    bridge->q_uses_cpu_rms = q_uses_cpu_rms;
    bridge->kv_uses_cpu_rms = kv_uses_cpu_rms;
    bridge->q_proj_uses_cpu = q_proj_uses_cpu;
    bridge->k_proj_uses_cpu = k_proj_uses_cpu;
    bridge->v_proj_uses_cpu = v_proj_uses_cpu;
    bridge->q_query_uses_cpu = q_query_uses_cpu;
    bridge->q_gate_uses_cpu = q_gate_uses_cpu;
    bridge->ffn_uses_cpu = ffn_uses_cpu;
    bridge->q_gate_cpu_channel_count = load_q_gate_cpu_channel_override(
        bridge->q_gate_cpu_channels,
        (int)(sizeof(bridge->q_gate_cpu_channels) / sizeof(bridge->q_gate_cpu_channels[0])),
        bridge->q_dim
    );
    if (bridge->q_gate_cpu_channel_count > 0 &&
        !override_enabled_for_layer("ORION_Q_GATE_CPU_CHANNEL_LAYERS", layer, manifest->n_layer)) {
        bridge->q_gate_cpu_channel_count = 0;
    }
    if (bridge->ffn_uses_cpu &&
        !override_enabled_for_layer("ORION_FFN_CPU_LAYERS", layer, manifest->n_layer)) {
        bridge->ffn_uses_cpu = 0;
    }
    bridge->v_proj_cpu_channel_count = load_v_proj_cpu_channel_override(
        bridge->v_proj_cpu_channels,
        (int)(sizeof(bridge->v_proj_cpu_channels) / sizeof(bridge->v_proj_cpu_channels[0])),
        bridge->q_dim
    );
    return 1;
}

static int mixed_full_attention_layer(const char *blob_dir,
                                      OrionQwen35Manifest *manifest,
                                      OrionQwen35AneBridge *bridge,
                                      const float *hidden_in,
                                      int seq_len,
                                      float *hidden_out) {
    const int layer = bridge->layer_idx;
    const int d_model = manifest->d_model;
    const int d_ff = manifest->d_ff;
    const int n_head = manifest->n_head;
    const int n_kv_head = manifest->n_kv_head;
    const int head_dim = manifest->head_dim;
    const int q_dim = bridge->q_dim;
    const int kv_dim = bridge->kv_dim;
    const int full_cpu_proj_path =
        bridge->q_proj_uses_cpu &&
        bridge->k_proj_uses_cpu &&
        bridge->v_proj_uses_cpu &&
        bridge->ffn_uses_cpu;

    float *input_ln = NULL;
    float *post_ln = NULL;
    float *gate_proj = NULL;
    float *up_proj = NULL;
    float *down_proj = NULL;
    float *o_proj = NULL;
    float *q_norm = NULL;
    float *k_norm = NULL;
    float *q_proj = NULL;
    float *k_proj = NULL;
    float *v_proj = NULL;
    float *normed = NULL;
    float *cpu_k_proj_seq = NULL;
    float *cpu_v_proj_seq = NULL;
    float *q_proj_seq = NULL;
    float *k_proj_seq = NULL;
    float *v_proj_seq = NULL;
    float *attn = NULL;
    float *hidden_attn = NULL;
    float *post_normed = NULL;
    float *mlp = NULL;
    IOSurfaceRef ioInQ = NULL;
    IOSurfaceRef ioInKV = NULL;
    IOSurfaceRef ioQ = NULL;
    IOSurfaceRef ioK = NULL;
    IOSurfaceRef ioV = NULL;
    IOSurfaceRef ioFfnIn = NULL;
    IOSurfaceRef ioHidden = NULL;

    input_ln = load_exact(blob_dir, layer, "input_layernorm.bin", d_model);
    post_ln = load_exact(blob_dir, layer, "post_attention_layernorm.bin", d_model);
    gate_proj = load_exact(blob_dir, layer, "mlp_gate_proj.bin", d_ff * d_model);
    up_proj = load_exact(blob_dir, layer, "mlp_up_proj.bin", d_ff * d_model);
    down_proj = load_exact(blob_dir, layer, "mlp_down_proj.bin", d_model * d_ff);
    o_proj = load_exact(blob_dir, layer, "self_attn_o_proj.bin", d_model * q_dim);
    q_norm = load_exact(blob_dir, layer, "self_attn_q_norm.bin", head_dim);
    k_norm = load_exact(blob_dir, layer, "self_attn_k_norm.bin", head_dim);
    if (bridge->q_proj_uses_cpu || bridge->q_query_uses_cpu || bridge->q_gate_uses_cpu || bridge->q_gate_cpu_channel_count > 0 || full_cpu_proj_path) {
        q_proj = load_exact(blob_dir, layer, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
    }
    if (bridge->k_proj_uses_cpu || full_cpu_proj_path) {
        k_proj = load_exact(blob_dir, layer, "self_attn_k_proj.bin", kv_dim * d_model);
    }
    if (bridge->v_proj_uses_cpu || bridge->v_proj_cpu_channel_count > 0 || full_cpu_proj_path) {
        v_proj = load_exact(blob_dir, layer, "self_attn_v_proj.bin", kv_dim * d_model);
    }
    if (!input_ln || !post_ln || !gate_proj || !up_proj || !down_proj || !o_proj || !q_norm || !k_norm ||
        ((bridge->q_proj_uses_cpu || bridge->q_query_uses_cpu || bridge->q_gate_uses_cpu || bridge->q_gate_cpu_channel_count > 0 || full_cpu_proj_path) && !q_proj) ||
        ((bridge->k_proj_uses_cpu || full_cpu_proj_path) && !k_proj) ||
        ((bridge->v_proj_uses_cpu || bridge->v_proj_cpu_channel_count > 0 || full_cpu_proj_path) && !v_proj)) goto fail;

    normed = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    cpu_k_proj_seq = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    cpu_v_proj_seq = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    q_proj_seq = (float *)calloc((size_t)seq_len * (q_dim * 2), sizeof(float));
    k_proj_seq = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    v_proj_seq = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    hidden_attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    post_normed = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    mlp = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    if (!normed || !cpu_k_proj_seq || !cpu_v_proj_seq || !q_proj_seq || !k_proj_seq || !v_proj_seq || !attn || !hidden_attn || !post_normed || !mlp) goto fail;

    for (int t = 0; t < seq_len; t++) {
        orion_qwen_cpu_rmsnorm(hidden_in + t * d_model, input_ln, d_model, 1e-6f, normed + t * d_model);
    }

    if (full_cpu_proj_path) {
        orion_qwen_cpu_full_attention_prefill_with_rope(
            normed, seq_len, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
            d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
            attn
        );

        for (int i = 0; i < seq_len * d_model; i++) hidden_attn[i] = hidden_in[i] + attn[i];
        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(hidden_attn + s * d_model, post_ln, d_model, 1e-6f, post_normed + s * d_model);
            orion_qwen_cpu_swiglu_ffn(post_normed + s * d_model, gate_proj, up_proj, down_proj, d_model, d_ff, mlp + s * d_model);
        }
        for (int i = 0; i < seq_len * d_model; i++) hidden_out[i] = hidden_attn[i] + mlp[i];

        if (input_ln) free(input_ln);
        if (post_ln) free(post_ln);
        if (gate_proj) free(gate_proj);
        if (up_proj) free(up_proj);
        if (down_proj) free(down_proj);
        if (o_proj) free(o_proj);
        if (q_norm) free(q_norm);
        if (k_norm) free(k_norm);
        if (q_proj) free(q_proj);
        if (k_proj) free(k_proj);
        if (v_proj) free(v_proj);
        if (normed) free(normed);
        if (cpu_k_proj_seq) free(cpu_k_proj_seq);
        if (cpu_v_proj_seq) free(cpu_v_proj_seq);
        if (q_proj_seq) free(q_proj_seq);
        if (k_proj_seq) free(k_proj_seq);
        if (v_proj_seq) free(v_proj_seq);
        if (attn) free(attn);
        if (hidden_attn) free(hidden_attn);
        if (post_normed) free(post_normed);
        if (mlp) free(mlp);
        return 1;
    }

    const float *q_input = bridge->q_uses_cpu_rms ? normed : hidden_in;
    const float *kv_input = bridge->kv_uses_cpu_rms ? normed : hidden_in;
    ioInQ = make_cpu_seq_input_surface(q_input, seq_len, bridge->bucket, d_model);
    ioInKV = make_cpu_seq_input_surface(kv_input, seq_len, bridge->bucket, d_model);
    ioQ = make_f32_surface((q_dim * 2) * bridge->bucket, 0.0f);
    ioK = make_f32_surface(kv_dim * bridge->bucket, 0.0f);
    ioV = make_f32_surface(kv_dim * bridge->bucket, 0.0f);

    IOSurfaceRef insQ[] = {ioInQ};
    IOSurfaceRef outsQ[] = {ioQ};
    IOSurfaceRef insKV[] = {ioInKV};
    IOSurfaceRef outsKV[] = {ioK, ioV};
    if (!orion_eval(bridge->prog_q, insQ, 1, outsQ, 1) || !orion_eval(bridge->prog_kv, insKV, 1, outsKV, 2)) goto fail;

    read_ane_surface_prefix(ioQ, q_dim * 2, seq_len, bridge->bucket, q_proj_seq);
    read_ane_surface_prefix(ioK, kv_dim, seq_len, bridge->bucket, k_proj_seq);
    read_ane_surface_prefix(ioV, kv_dim, seq_len, bridge->bucket, v_proj_seq);
    if (bridge->q_proj_uses_cpu || bridge->q_query_uses_cpu || bridge->q_gate_uses_cpu || bridge->q_gate_cpu_channel_count > 0) {
        float *cpu_q_proj_seq = (float *)calloc((size_t)seq_len * (q_dim * 2), sizeof(float));
        if (!cpu_q_proj_seq) goto fail;
        cpu_linear_batch(normed, seq_len, q_proj, d_model, q_dim * 2, cpu_q_proj_seq);
        for (int s = 0; s < seq_len; s++) {
            float *dst = q_proj_seq + (size_t)s * (q_dim * 2);
            const float *src = cpu_q_proj_seq + (size_t)s * (q_dim * 2);
            if (bridge->q_proj_uses_cpu || bridge->q_query_uses_cpu) {
                memcpy(dst, src, (size_t)q_dim * sizeof(float));
            }
            if (bridge->q_proj_uses_cpu || bridge->q_gate_uses_cpu) {
                memcpy(dst + q_dim, src + q_dim, (size_t)q_dim * sizeof(float));
            }
        }
        if (!bridge->q_proj_uses_cpu && !bridge->q_gate_uses_cpu && bridge->q_gate_cpu_channel_count > 0) {
            apply_q_gate_cpu_channel_overrides(q_proj_seq, cpu_q_proj_seq, seq_len, q_dim,
                                               bridge->q_gate_cpu_channels, bridge->q_gate_cpu_channel_count);
        }
        free(cpu_q_proj_seq);
    }
    if (bridge->k_proj_uses_cpu) {
        cpu_linear_batch(normed, seq_len, k_proj, d_model, kv_dim, cpu_k_proj_seq);
        memcpy(k_proj_seq, cpu_k_proj_seq, (size_t)seq_len * kv_dim * sizeof(float));
    }
    if (bridge->v_proj_uses_cpu || bridge->v_proj_cpu_channel_count > 0) {
        cpu_linear_batch(normed, seq_len, v_proj, d_model, kv_dim, cpu_v_proj_seq);
    }
    if (bridge->v_proj_uses_cpu) {
        memcpy(v_proj_seq, cpu_v_proj_seq, (size_t)seq_len * kv_dim * sizeof(float));
    } else if (bridge->v_proj_cpu_channel_count > 0) {
        const int q_per_kv = n_head / n_kv_head;
        apply_v_proj_cpu_channel_overrides(v_proj_seq, cpu_v_proj_seq, seq_len, kv_dim, head_dim, q_per_kv,
                                           bridge->v_proj_cpu_channels, bridge->v_proj_cpu_channel_count);
    }

    orion_qwen_cpu_full_attention_from_projections_with_rope(
        q_proj_seq, k_proj_seq, v_proj_seq, seq_len,
        o_proj, q_norm, k_norm,
        d_model, n_head, n_kv_head, head_dim,
        manifest->rope_theta, manifest->partial_rotary_factor,
        attn
    );

    for (int i = 0; i < seq_len * d_model; i++) hidden_attn[i] = hidden_in[i] + attn[i];
    if (bridge->ffn_uses_cpu) {
        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(hidden_attn + s * d_model, post_ln, d_model, 1e-6f, post_normed + s * d_model);
            orion_qwen_cpu_swiglu_ffn(post_normed + s * d_model, gate_proj, up_proj, down_proj, d_model, d_ff, mlp + s * d_model);
        }
        for (int i = 0; i < seq_len * d_model; i++) hidden_out[i] = hidden_attn[i] + mlp[i];
    } else {
        ioFfnIn = make_cpu_seq_input_surface(hidden_attn, seq_len, bridge->bucket, d_model);
        ioHidden = make_f32_surface(d_model * bridge->bucket, 0.0f);
        IOSurfaceRef insFFN[] = {ioFfnIn};
        IOSurfaceRef outsFFN[] = {ioHidden};
        if (!orion_eval(bridge->prog_ffn, insFFN, 1, outsFFN, 1)) goto fail;
        read_ane_surface_prefix(ioHidden, d_model, seq_len, bridge->bucket, hidden_out);
    }

    if (ioInQ) CFRelease(ioInQ);
    if (ioInKV) CFRelease(ioInKV);
    if (ioQ) CFRelease(ioQ);
    if (ioK) CFRelease(ioK);
    if (ioV) CFRelease(ioV);
    if (ioFfnIn) CFRelease(ioFfnIn);
    if (ioHidden) CFRelease(ioHidden);
    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj); free(o_proj); free(q_norm); free(k_norm); free(q_proj); free(k_proj); free(v_proj);
    free(normed); free(cpu_k_proj_seq); free(cpu_v_proj_seq); free(q_proj_seq); free(k_proj_seq); free(v_proj_seq); free(attn); free(hidden_attn); free(post_normed); free(mlp);
    return 1;

fail:
    if (ioInQ) CFRelease(ioInQ);
    if (ioInKV) CFRelease(ioInKV);
    if (ioQ) CFRelease(ioQ);
    if (ioK) CFRelease(ioK);
    if (ioV) CFRelease(ioV);
    if (ioFfnIn) CFRelease(ioFfnIn);
    if (ioHidden) CFRelease(ioHidden);
    if (input_ln) free(input_ln);
    if (post_ln) free(post_ln);
    if (gate_proj) free(gate_proj);
    if (up_proj) free(up_proj);
    if (down_proj) free(down_proj);
    if (o_proj) free(o_proj);
    if (q_norm) free(q_norm);
    if (k_norm) free(k_norm);
    if (q_proj) free(q_proj);
    if (k_proj) free(k_proj);
    if (v_proj) free(v_proj);
    if (normed) free(normed);
    if (cpu_k_proj_seq) free(cpu_k_proj_seq);
    if (cpu_v_proj_seq) free(cpu_v_proj_seq);
    if (q_proj_seq) free(q_proj_seq);
    if (k_proj_seq) free(k_proj_seq);
    if (v_proj_seq) free(v_proj_seq);
    if (attn) free(attn);
    if (hidden_attn) free(hidden_attn);
    if (post_normed) free(post_normed);
    if (mlp) free(mlp);
    return 0;
}

static int sampled_topk_logits(const char *blob_dir,
                               const char *lm_head_name,
                               const float *hidden,
                               int d_model,
                               int sample_vocab,
                               int *top_id,
                               float *top_logit,
                               int *runnerup_id,
                               float *runnerup_logit) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/model/%s", blob_dir, lm_head_name);
    float *row = (float *)malloc((size_t)d_model * sizeof(float));
    if (!row) return 0;

    int best_id = -1;
    int second_id = -1;
    float best = -INFINITY;
    float second = -INFINITY;
    for (int tok = 0; tok < sample_vocab; tok++) {
        if (!orion_read_blob_row_f32(path, tok, d_model, row)) {
            free(row);
            return 0;
        }
        float dot = 0.0f;
        for (int i = 0; i < d_model; i++) dot += hidden[i] * row[i];
        if (dot > best) {
            second = best;
            second_id = best_id;
            best = dot;
            best_id = tok;
        } else if (dot > second) {
            second = dot;
            second_id = tok;
        }
    }

    free(row);
    *top_id = best_id;
    *top_logit = best;
    *runnerup_id = second_id;
    *runnerup_logit = second;
    return 1;
}

static int compute_next_token_mixed(const char *blob_dir,
                                    OrionQwen35Manifest *manifest,
                                    OrionQwen35AneBridge *bridges,
                                    const unsigned char *bridge_mask,
                                    const int *token_ids,
                                    int seq_len,
                                    int sample_vocab,
                                    const char *lm_head_name_override,
                                    int *top_id,
                                    float *top_logit,
                                    int *runnerup_id,
                                    float *runnerup_logit,
                                    double *last_hidden_abs_sum) {
    const int d_model = manifest->d_model;
    const int d_ff = manifest->d_ff;

    float *hidden = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *normed = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *mixer_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *mlp_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *scratch = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *last_hidden = (float *)calloc((size_t)d_model, sizeof(float));
    float *final_norm = NULL;
    if (!hidden || !normed || !mixer_out || !mlp_out || !scratch || !last_hidden) goto fail;

    char embed_path[2048];
    snprintf(embed_path, sizeof(embed_path), "%s/model/embed_tokens.bin", blob_dir);
    for (int s = 0; s < seq_len; s++) {
        if (!orion_read_blob_row_f32(embed_path, token_ids[s], d_model, hidden + s * d_model)) goto fail;
    }

    for (int layer_idx = 0; layer_idx < manifest->n_layer; layer_idx++) {
        if (bridge_mask && bridge_mask[layer_idx]) {
            if (!mixed_full_attention_layer(blob_dir, manifest, &bridges[layer_idx], hidden, seq_len, mixer_out)) goto fail;
            memcpy(hidden, mixer_out, (size_t)seq_len * d_model * sizeof(float));
            continue;
        }

        float *input_ln = load_exact(blob_dir, layer_idx, "input_layernorm.bin", d_model);
        float *post_ln = load_exact(blob_dir, layer_idx, "post_attention_layernorm.bin", d_model);
        float *gate_proj = load_exact(blob_dir, layer_idx, "mlp_gate_proj.bin", d_ff * d_model);
        float *up_proj = load_exact(blob_dir, layer_idx, "mlp_up_proj.bin", d_ff * d_model);
        float *down_proj = load_exact(blob_dir, layer_idx, "mlp_down_proj.bin", d_model * d_ff);
        if (!input_ln || !post_ln || !gate_proj || !up_proj || !down_proj) {
            free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
            goto fail;
        }

        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(hidden + s * d_model, input_ln, d_model, 1e-6f, normed + s * d_model);
        }
        memset(mixer_out, 0, (size_t)seq_len * d_model * sizeof(float));

        char full_q_path[2048];
        snprintf(full_q_path, sizeof(full_q_path), "%s/layer%d/self_attn_q_proj.bin", blob_dir, layer_idx);
        if (file_exists(full_q_path)) {
            const int n_head = manifest->n_head;
            const int n_kv_head = manifest->n_kv_head;
            const int head_dim = manifest->head_dim;
            const int q_dim = n_head * head_dim;
            const int kv_dim = n_kv_head * head_dim;
            float *q_proj = load_exact(blob_dir, layer_idx, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
            float *k_proj = load_exact(blob_dir, layer_idx, "self_attn_k_proj.bin", kv_dim * d_model);
            float *v_proj = load_exact(blob_dir, layer_idx, "self_attn_v_proj.bin", kv_dim * d_model);
            float *o_proj = load_exact(blob_dir, layer_idx, "self_attn_o_proj.bin", d_model * q_dim);
            float *q_norm = load_exact(blob_dir, layer_idx, "self_attn_q_norm.bin", head_dim);
            float *k_norm = load_exact(blob_dir, layer_idx, "self_attn_k_norm.bin", head_dim);
            if (!q_proj || !k_proj || !v_proj || !o_proj || !q_norm || !k_norm) {
                free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
                free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
                goto fail;
            }
            orion_qwen_cpu_full_attention_prefill_with_rope(
                normed, seq_len, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
                d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
                mixer_out
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
                goto fail;
            }
            orion_qwen_cpu_linear_attention_recurrent_prefill(
                normed, seq_len, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
                conv1d, dt_bias, a_log, norm_weight, out_proj,
                d_model, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel,
                mixer_out
            );
            free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
            free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
        }

        for (int i = 0; i < seq_len * d_model; i++) hidden[i] += mixer_out[i];
        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(hidden + s * d_model, post_ln, d_model, 1e-6f, scratch + s * d_model);
            orion_qwen_cpu_swiglu_ffn(scratch + s * d_model, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out + s * d_model);
        }
        for (int i = 0; i < seq_len * d_model; i++) hidden[i] += mlp_out[i];

        free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
    }

    char final_norm_path[2048];
    snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
    final_norm = orion_read_blob_f32_exact(final_norm_path, d_model);
    if (!final_norm) goto fail;
    orion_qwen_cpu_rmsnorm(hidden + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, last_hidden);

    const char *lm_head_name = lm_head_name_override
        ? lm_head_name_override
        : (manifest->tie_word_embeddings ? "embed_tokens.bin" : "lm_head.bin");
    if (!sampled_topk_logits(blob_dir, lm_head_name, last_hidden, d_model, sample_vocab,
                             top_id, top_logit, runnerup_id, runnerup_logit)) goto fail;
    *last_hidden_abs_sum = abs_sum(last_hidden, d_model);

    free(hidden); free(normed); free(mixer_out); free(mlp_out); free(scratch); free(last_hidden); free(final_norm);
    return 1;

fail:
    free(hidden); free(normed); free(mixer_out); free(mlp_out); free(scratch); free(last_hidden); free(final_norm);
    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s <blob_dir> <tokenizer_dir> [single|all_full] [prompt] [gen_len] [sample_vocab] [lm_head_name]\n", argv[0]);
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
        OrionGPT2Tokenizer* tok = orion_gpt2_tokenizer_load_with_regex(vocabPath.UTF8String, mergesPath.UTF8String, regex.UTF8String);
        if (!tok) {
            fprintf(stderr, "FAIL: tokenizer load failed\n");
            return 1;
        }

        OrionQwen35Manifest* manifest = orion_qwen35_manifest_load(blob_dir);
        if (!manifest) {
            fprintf(stderr, "FAIL: manifest loader returned NULL\n");
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        const char *mode = (argc >= 4) ? argv[3] : "single";
        const int qkv_input_mode = load_qkv_input_mode();
        const int q_proj_uses_cpu = use_cpu_q_proj_override();
        const int k_proj_uses_cpu = use_cpu_k_proj_override();
        const int q_query_uses_cpu = use_cpu_q_query_override();
        const int q_gate_uses_cpu = use_cpu_q_gate_override();
        const int ffn_uses_cpu = use_cpu_ffn_override();
        const char *q_gate_source = getenv("ORION_Q_GATE_SOURCE");
        const int v_proj_uses_cpu = use_cpu_v_proj_override();
        int q_gate_cpu_channels[32] = {0};
        const int q_gate_cpu_channel_count = load_q_gate_cpu_channel_override(
            q_gate_cpu_channels,
            (int)(sizeof(q_gate_cpu_channels) / sizeof(q_gate_cpu_channels[0])),
            manifest->n_head * manifest->head_dim
        );
        int v_proj_cpu_channels[32] = {0};
        const int v_proj_cpu_channel_count = load_v_proj_cpu_channel_override(
            v_proj_cpu_channels,
            (int)(sizeof(v_proj_cpu_channels) / sizeof(v_proj_cpu_channels[0])),
            manifest->n_head * manifest->head_dim
        );

        const char *prompt = (argc >= 5) ? argv[4] : "YES";
        int gen_len = (argc >= 6) ? atoi(argv[5]) : 2;
        int sample_vocab = (argc >= 7) ? atoi(argv[6]) : 4096;
        const char *lm_head_name_override = (argc >= 8) ? argv[7] : NULL;
        if (gen_len <= 0 || gen_len > 8) gen_len = 2;
        if (sample_vocab <= 0) sample_vocab = 4096;
        int tokens[64] = {0};
        int prompt_len = orion_gpt2_encode(tok, prompt, tokens, 64);
        if (prompt_len <= 0) {
            fprintf(stderr, "FAIL: prompt encode failed\n");
            orion_qwen35_manifest_free(manifest);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        const int required_seq_len = prompt_len + gen_len - 1;
        const int bucket = orion_select_bucket(required_seq_len, kQwenPrefillBuckets, kQwenPrefillNumBuckets);
        if (bucket <= 0) {
            fprintf(stderr, "FAIL: no bucket fits required_seq_len=%d\n", required_seq_len);
            orion_qwen35_manifest_free(manifest);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        OrionQwen35AneBridge *bridges = calloc((size_t)manifest->n_layer, sizeof(OrionQwen35AneBridge));
        unsigned char *bridge_mask = calloc((size_t)manifest->n_layer, sizeof(unsigned char));
        int bridged_layer = 3;
        int bridged_layer_count = 0;
        if (!bridges || !bridge_mask) {
            fprintf(stderr, "FAIL: bridge allocation failed\n");
            free(bridges);
            free(bridge_mask);
            orion_qwen35_manifest_free(manifest);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        if (strcmp(mode, "all_full") == 0) {
            int full_layers[128] = {0};
            int full_count = load_full_attention_layers(manifest, full_layers, 128);
            if (full_count <= 0) {
                fprintf(stderr, "FAIL: no full_attention layers found\n");
                free(bridges);
                free(bridge_mask);
                orion_qwen35_manifest_free(manifest);
                orion_gpt2_tokenizer_free(tok);
                return 1;
            }
            bridged_layer = full_layers[0];
            for (int i = 0; i < full_count; i++) {
                int layer = full_layers[i];
                if (!bridge_init(&bridges[layer], [NSString stringWithUTF8String:blob_dir], layer, bucket, manifest, qkv_input_mode, q_proj_uses_cpu, k_proj_uses_cpu, v_proj_uses_cpu, q_query_uses_cpu, q_gate_uses_cpu, ffn_uses_cpu)) {
                    fprintf(stderr, "FAIL: bridge_init failed for layer %d\n", layer);
                    for (int j = 0; j < manifest->n_layer; j++) bridge_release(&bridges[j]);
                    free(bridges);
                    free(bridge_mask);
                    orion_qwen35_manifest_free(manifest);
                    orion_gpt2_tokenizer_free(tok);
                    return 1;
                }
                bridge_mask[layer] = 1;
                bridged_layer_count += 1;
            }
        } else {
            if (!bridge_init(&bridges[bridged_layer], [NSString stringWithUTF8String:blob_dir], bridged_layer, bucket, manifest, qkv_input_mode, q_proj_uses_cpu, k_proj_uses_cpu, v_proj_uses_cpu, q_query_uses_cpu, q_gate_uses_cpu, ffn_uses_cpu)) {
                fprintf(stderr, "FAIL: bridge_init failed\n");
                free(bridges);
                free(bridge_mask);
                orion_qwen35_manifest_free(manifest);
                orion_gpt2_tokenizer_free(tok);
                return 1;
            }
            bridge_mask[bridged_layer] = 1;
            bridged_layer_count = 1;
        }

        int gen_ids[8] = {0};
        float gen_logits[8] = {0.0f};
        int gen_runnerup_ids[8] = {0};
        float gen_runnerup_logits[8] = {0.0f};
        float gen_top2_margins[8] = {0.0f};
        double final_hidden_abs = 0.0;
        int seq[128] = {0};
        memcpy(seq, tokens, (size_t)prompt_len * sizeof(int));
        int seq_len = prompt_len;

        for (int step = 0; step < gen_len; step++) {
            int next_id = -1;
            float next_logit = -INFINITY;
            int runnerup_id = -1;
            float runnerup_logit = -INFINITY;
            if (!compute_next_token_mixed(blob_dir, manifest, bridges, bridge_mask, seq, seq_len, sample_vocab,
                                          lm_head_name_override, &next_id, &next_logit,
                                          &runnerup_id, &runnerup_logit, &final_hidden_abs)) {
                fprintf(stderr, "FAIL: mixed top1 compute failed at step %d\n", step);
                for (int i = 0; i < manifest->n_layer; i++) bridge_release(&bridges[i]);
                free(bridges);
                free(bridge_mask);
                orion_qwen35_manifest_free(manifest);
                orion_gpt2_tokenizer_free(tok);
                return 1;
            }
            gen_ids[step] = next_id;
            gen_logits[step] = next_logit;
            gen_runnerup_ids[step] = runnerup_id;
            gen_runnerup_logits[step] = runnerup_logit;
            gen_top2_margins[step] = next_logit - runnerup_logit;
            seq[seq_len++] = next_id;
        }

        char *decoded_prompt = orion_gpt2_decode(tok, tokens, prompt_len);
        char *decoded_gen = orion_gpt2_decode(tok, gen_ids, gen_len);
        char *decoded_full = orion_gpt2_decode(tok, seq, seq_len);

        printf("PASS: qwen35 9b decode loop mixed ane cpu smoke\n");
        printf("  prompt=%s\n", prompt);
        printf("  prompt_len=%d\n", prompt_len);
        printf("  gen_len=%d\n", gen_len);
        printf("  bridge_mode=%s\n", mode);
        printf("  bucket=%d\n", bucket);
        printf("  qkv_input_mode=%s\n", qkv_input_mode_label(qkv_input_mode));
        printf("  q_proj_source=%s\n", q_proj_source_label(q_proj_uses_cpu));
        printf("  q_query_source=%s\n", q_query_source_label(q_query_uses_cpu));
        printf("  k_proj_source=%s\n", k_proj_source_label(k_proj_uses_cpu));
        printf("  q_gate_source=%s\n", q_gate_source_label(q_gate_source, q_gate_uses_cpu, q_gate_cpu_channel_count));
        printf("  ffn_source=%s\n", ffn_source_label(ffn_uses_cpu));
        printf("  q_gate_cpu_channel_count=%d\n", q_gate_cpu_channel_count);
        for (int i = 0; i < q_gate_cpu_channel_count; i++) {
            printf("  q_gate_cpu_channel_rank=%d channel=%d\n", i + 1, q_gate_cpu_channels[i]);
        }
        printf("  v_proj_source=%s\n", v_proj_source_label(v_proj_uses_cpu, v_proj_cpu_channel_count));
        printf("  v_proj_cpu_channel_count=%d\n", v_proj_cpu_channel_count);
        for (int i = 0; i < v_proj_cpu_channel_count; i++) {
            printf("  v_proj_cpu_channel_rank=%d channel=%d\n", i + 1, v_proj_cpu_channels[i]);
        }
        printf("  bridged_layer=%d\n", bridged_layer);
        printf("  bridged_layer_count=%d\n", bridged_layer_count);
        printf("  sample_vocab=%d\n", sample_vocab);
        printf("  lm_head_name=%s\n", lm_head_name_override ? lm_head_name_override : (manifest->tie_word_embeddings ? "embed_tokens.bin" : "lm_head.bin"));
        for (int i = 0; i < gen_len; i++) {
            printf("  gen_token_%d=%d\n", i, gen_ids[i]);
            printf("  gen_logit_%d=%.6f\n", i, gen_logits[i]);
            printf("  gen_runnerup_token_%d=%d\n", i, gen_runnerup_ids[i]);
            printf("  gen_runnerup_logit_%d=%.6f\n", i, gen_runnerup_logits[i]);
            printf("  gen_top2_margin_%d=%.6f\n", i, gen_top2_margins[i]);
        }
        printf("  final_hidden_abs_sum=%.6f\n", final_hidden_abs);
        printf("  decoded_prompt=%s\n", decoded_prompt ? decoded_prompt : "");
        printf("  decoded_generated=%s\n", decoded_gen ? decoded_gen : "");
        printf("  decoded_full=%s\n", decoded_full ? decoded_full : "");
        printf("  next_blocker=%s\n", strcmp(mode, "all_full") == 0 ? "evaluate prompt quality and extend beyond full-attention bridge coverage" : "expand bridge from one full-attention layer to all full-attention layers");

        free(decoded_prompt);
        free(decoded_gen);
        free(decoded_full);
        for (int i = 0; i < manifest->n_layer; i++) bridge_release(&bridges[i]);
        free(bridges);
        free(bridge_mask);
        orion_qwen35_manifest_free(manifest);
        orion_gpt2_tokenizer_free(tok);
        return 0;
    }
}
