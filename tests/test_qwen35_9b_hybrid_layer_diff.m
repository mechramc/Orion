#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <ctype.h>
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

static double abs_sum(const float *x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)x[i]);
    return total;
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

static void cpu_linear_batch(const float *x_seq,
                             int seq_len,
                             const float *weight,
                             int in_dim,
                             int out_dim,
                             float *out_seq);

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

static int use_cpu_q_proj_override(void) {
    const char *full = getenv("ORION_FULL_ATTN_SOURCE");
    if (full && strcmp(full, "cpu") == 0) return 1;
    const char *source = getenv("ORION_Q_PROJ_SOURCE");
    return source && strcmp(source, "cpu") == 0;
}

static int use_cpu_k_proj_override(void) {
    const char *full = getenv("ORION_FULL_ATTN_SOURCE");
    if (full && strcmp(full, "cpu") == 0) return 1;
    const char *source = getenv("ORION_K_PROJ_SOURCE");
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

static int use_cpu_v_proj_override(void) {
    const char *full = getenv("ORION_FULL_ATTN_SOURCE");
    if (full && strcmp(full, "cpu") == 0) return 1;
    const char *source = getenv("ORION_V_PROJ_SOURCE");
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
        "qwen35_9b_diff_q"
    );
    bridge->prog_kv = orion_compile_mil(
        mil_kv.UTF8String,
        kv_uses_cpu_rms ? build_kv_linear_only_wdict(layer, blobDir) : build_kv_wdict(layer, blobDir),
        "qwen35_9b_diff_kv"
    );
    bridge->prog_ffn = orion_compile_mil(mil_ffn.UTF8String, build_ffn_wdict(layer, blobDir), "qwen35_9b_diff_ffn");
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

    float *input_ln = NULL;
    float *post_ln = NULL;
    float *gate_proj = NULL;
    float *up_proj = NULL;
    float *down_proj = NULL;
    float *normed = NULL;
    float *mixer_out = NULL;
    float *scratch = NULL;
    float *mlp_out = NULL;

    input_ln = load_exact(blob_dir, layer_idx, "input_layernorm.bin", d_model);
    post_ln = load_exact(blob_dir, layer_idx, "post_attention_layernorm.bin", d_model);
    gate_proj = load_exact(blob_dir, layer_idx, "mlp_gate_proj.bin", d_ff * d_model);
    up_proj = load_exact(blob_dir, layer_idx, "mlp_up_proj.bin", d_ff * d_model);
    down_proj = load_exact(blob_dir, layer_idx, "mlp_down_proj.bin", d_model * d_ff);
    if (!input_ln || !post_ln || !gate_proj || !up_proj || !down_proj) goto fail;

    normed = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    mixer_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    scratch = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    mlp_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    if (!normed || !mixer_out || !scratch || !mlp_out) goto fail;

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

    for (int i = 0; i < seq_len * d_model; i++) hidden_out[i] = hidden_in[i] + mixer_out[i];
    for (int s = 0; s < seq_len; s++) {
        orion_qwen_cpu_rmsnorm(hidden_out + s * d_model, post_ln, d_model, 1e-6f, scratch + s * d_model);
        orion_qwen_cpu_swiglu_ffn(scratch + s * d_model, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out + s * d_model);
    }
    for (int i = 0; i < seq_len * d_model; i++) hidden_out[i] += mlp_out[i];

    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
    free(normed); free(mixer_out); free(scratch); free(mlp_out);
    return 1;

fail:
    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
    free(normed); free(mixer_out); free(scratch); free(mlp_out);
    return 0;
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
    if (!orion_eval(bridge->prog_q, insQ, 1, outsQ, 1) || !orion_eval(bridge->prog_kv, insKV, 1, outsKV, 2)) goto fail_with_surfaces;

    read_ane_surface_prefix(ioQ, q_dim * 2, seq_len, bridge->bucket, q_proj_seq);
    read_ane_surface_prefix(ioK, kv_dim, seq_len, bridge->bucket, k_proj_seq);
    read_ane_surface_prefix(ioV, kv_dim, seq_len, bridge->bucket, v_proj_seq);
    if (bridge->q_proj_uses_cpu || bridge->q_query_uses_cpu || bridge->q_gate_uses_cpu || bridge->q_gate_cpu_channel_count > 0) {
        float *cpu_q_proj_seq = (float *)calloc((size_t)seq_len * (q_dim * 2), sizeof(float));
        if (!cpu_q_proj_seq) goto fail_with_surfaces;
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
        if (!orion_eval(bridge->prog_ffn, insFFN, 1, outsFFN, 1)) goto fail_with_surfaces;
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

fail_with_surfaces:
    if (ioInQ) CFRelease(ioInQ);
    if (ioInKV) CFRelease(ioInKV);
    if (ioQ) CFRelease(ioQ);
    if (ioK) CFRelease(ioK);
    if (ioV) CFRelease(ioV);
    if (ioFfnIn) CFRelease(ioFfnIn);
    if (ioHidden) CFRelease(ioHidden);
fail:
    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj); free(o_proj); free(q_norm); free(k_norm); free(q_proj); free(k_proj); free(v_proj);
    free(normed); free(cpu_k_proj_seq); free(cpu_v_proj_seq); free(q_proj_seq); free(k_proj_seq); free(v_proj_seq); free(attn); free(hidden_attn); free(post_normed); free(mlp);
    return 0;
}

static int apply_hybrid_layer(const char *blob_dir,
                              OrionQwen35Manifest *manifest,
                              OrionQwen35AneBridge *bridges,
                              const unsigned char *bridge_mask,
                              int layer_idx,
                              const float *hidden_in,
                              int seq_len,
                              float *hidden_out) {
    if (bridge_mask && bridge_mask[layer_idx]) {
        return mixed_full_attention_layer(blob_dir, manifest, &bridges[layer_idx], hidden_in, seq_len, hidden_out);
    }
    return apply_cpu_layer(blob_dir, manifest, layer_idx, hidden_in, seq_len, hidden_out);
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

static int sampled_topk_logits(const char *blob_dir,
                               const char *lm_head_name,
                               const float *hidden,
                               int d_model,
                               int sample_vocab,
                               int *top_id,
                               float *top_logit) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/model/%s", blob_dir, lm_head_name);
    float *row = (float *)malloc((size_t)d_model * sizeof(float));
    if (!row) return 0;

    int best_id = -1;
    float best = -INFINITY;
    for (int tok = 0; tok < sample_vocab; tok++) {
        if (!orion_read_blob_row_f32(path, tok, d_model, row)) {
            free(row);
            return 0;
        }
        float dot = 0.0f;
        for (int i = 0; i < d_model; i++) dot += hidden[i] * row[i];
        if (dot > best) {
            best = dot;
            best_id = tok;
        }
    }

    free(row);
    *top_id = best_id;
    *top_logit = best;
    return 1;
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

#define ORION_GATE_ATTR_REPORT_MAX 64
#define ORION_GATE_INPUT_ATTR_REPORT_MAX 16
#define ORION_ATTN_V_ATTR_REPORT_MAX 16
#define ORION_FFN_DOWN_ATTR_REPORT_MAX 16
#define ORION_TRACE_DIM_MAX 16

typedef struct {
    int channel;
    double delta_hidden;
    double single_pair_gap;
} OrionGateAttrImpact;

typedef struct {
    int dim;
    double abs_contrib;
    double signed_contrib;
    double delta_ffn_rms;
} OrionGateInputAttrImpact;

typedef struct {
    int dim;
    double cpu_norm;
    double hybrid_norm;
    double delta_norm;
    double pair_delta;
    double abs_contrib;
    double signed_contrib;
} OrionAttnVAttrImpact;

typedef struct {
    int dim;
    double cpu_norm;
    double hybrid_norm;
    double delta_norm;
    double pair_delta;
    double abs_contrib;
    double signed_contrib;
} OrionFfnDownAttrImpact;

static int compare_gate_attr_impact_asc(const void *lhs, const void *rhs) {
    const OrionGateAttrImpact *a = (const OrionGateAttrImpact *)lhs;
    const OrionGateAttrImpact *b = (const OrionGateAttrImpact *)rhs;
    if (a->single_pair_gap < b->single_pair_gap) return -1;
    if (a->single_pair_gap > b->single_pair_gap) return 1;
    if (a->channel < b->channel) return -1;
    if (a->channel > b->channel) return 1;
    return 0;
}

static int compare_gate_input_attr_impact_desc(const void *lhs, const void *rhs) {
    const OrionGateInputAttrImpact *a = (const OrionGateInputAttrImpact *)lhs;
    const OrionGateInputAttrImpact *b = (const OrionGateInputAttrImpact *)rhs;
    if (a->abs_contrib > b->abs_contrib) return -1;
    if (a->abs_contrib < b->abs_contrib) return 1;
    if (a->dim < b->dim) return -1;
    if (a->dim > b->dim) return 1;
    return 0;
}

static int compare_attn_v_attr_impact_desc(const void *lhs, const void *rhs) {
    const OrionAttnVAttrImpact *a = (const OrionAttnVAttrImpact *)lhs;
    const OrionAttnVAttrImpact *b = (const OrionAttnVAttrImpact *)rhs;
    if (a->abs_contrib > b->abs_contrib) return -1;
    if (a->abs_contrib < b->abs_contrib) return 1;
    if (a->dim < b->dim) return -1;
    if (a->dim > b->dim) return 1;
    return 0;
}

static int compare_ffn_down_attr_impact_desc(const void *lhs, const void *rhs) {
    const OrionFfnDownAttrImpact *a = (const OrionFfnDownAttrImpact *)lhs;
    const OrionFfnDownAttrImpact *b = (const OrionFfnDownAttrImpact *)rhs;
    if (a->abs_contrib > b->abs_contrib) return -1;
    if (a->abs_contrib < b->abs_contrib) return 1;
    if (a->dim < b->dim) return -1;
    if (a->dim > b->dim) return 1;
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

static int load_pair_delta_row(const char *blob_dir,
                               const char *lm_head_name,
                               int d_model,
                               int candidate_a,
                               int candidate_b,
                               float *pair_delta_out) {
    if (!pair_delta_out) return 0;
    char path[2048];
    snprintf(path, sizeof(path), "%s/model/%s", blob_dir, lm_head_name);
    float *row_a = (float *)malloc((size_t)d_model * sizeof(float));
    float *row_b = (float *)malloc((size_t)d_model * sizeof(float));
    if (!row_a || !row_b) {
        free(row_a);
        free(row_b);
        return 0;
    }
    int ok = orion_read_blob_row_f32(path, candidate_a, d_model, row_a) &&
             orion_read_blob_row_f32(path, candidate_b, d_model, row_b);
    if (ok) {
        for (int i = 0; i < d_model; i++) pair_delta_out[i] = row_a[i] - row_b[i];
    }
    free(row_a);
    free(row_b);
    return ok;
}

static void pair_gap_from_hidden_last_pair_delta(const float *final_norm,
                                                 const float *hidden_last,
                                                 const float *pair_delta,
                                                 int d_model,
                                                 float *normed_last_scratch,
                                                 double *out_gap) {
    orion_qwen_cpu_rmsnorm(hidden_last, final_norm, d_model, 1e-6f, normed_last_scratch);
    double dot = 0.0;
    for (int i = 0; i < d_model; i++) dot += (double)normed_last_scratch[i] * (double)pair_delta[i];
    *out_gap = dot;
}

typedef struct {
    double cpu_hidden_attn_pair_gap;
    double ane_hidden_attn_pair_gap;
    double cpu_final_pair_gap;
    double ane_final_pair_gap;
    int cpu_hidden_attn_pref_token;
    int ane_hidden_attn_pref_token;
    int cpu_final_pref_token;
    int ane_final_pref_token;
} OrionStagePairSensitivity;

typedef struct {
    double final_pair_gap;
    int final_pref_token;
    int top_id;
    float top_logit;
} OrionTailPairReplay;

typedef struct {
    double normed_mean_abs_diff;
    double normed_max_abs_diff;
    double mixed_linear_mean_abs_diff;
    double mixed_linear_max_abs_diff;
    double mixed_conv_mean_abs_diff;
    double mixed_conv_max_abs_diff;
    double query_mean_abs_diff;
    double query_max_abs_diff;
    double key_mean_abs_diff;
    double key_max_abs_diff;
    double value_mean_abs_diff;
    double value_max_abs_diff;
    double z_mean_abs_diff;
    double z_max_abs_diff;
    double beta_mean_abs_diff;
    double beta_max_abs_diff;
    double g_mean_abs_diff;
    double g_max_abs_diff;
    double core_pre_mean_abs_diff;
    double core_pre_max_abs_diff;
    double core_mean_abs_diff;
    double core_max_abs_diff;
    double attn_out_mean_abs_diff;
    double attn_out_max_abs_diff;
    double hidden_attn_mean_abs_diff;
    double hidden_attn_max_abs_diff;
    double final_mean_abs_diff;
    double final_max_abs_diff;
    double cpu_hidden_attn_pair_gap;
    double hybrid_hidden_attn_pair_gap;
    double cpu_final_pair_gap;
    double hybrid_final_pair_gap;
    int cpu_hidden_attn_pref_token;
    int hybrid_hidden_attn_pref_token;
    int cpu_final_pref_token;
    int hybrid_final_pref_token;
} OrionLinearStageSensitivity;

typedef struct {
    double normed_mean_abs_diff;
    double normed_max_abs_diff;
    double q_proj_mean_abs_diff;
    double q_proj_max_abs_diff;
    double q_gate_mean_abs_diff;
    double q_gate_max_abs_diff;
    double k_proj_mean_abs_diff;
    double k_proj_max_abs_diff;
    double v_proj_mean_abs_diff;
    double v_proj_max_abs_diff;
    double attn_gate_sigmoid_mean_abs_diff;
    double attn_gate_sigmoid_max_abs_diff;
    double attn_context_mean_abs_diff;
    double attn_context_max_abs_diff;
    double attn_gated_context_mean_abs_diff;
    double attn_gated_context_max_abs_diff;
    double attn_out_mean_abs_diff;
    double attn_out_max_abs_diff;
    double attn_qgate_only_mean_abs_diff;
    double attn_qgate_only_max_abs_diff;
    double attn_v_only_mean_abs_diff;
    double attn_v_only_max_abs_diff;
    double attn_context_only_mean_abs_diff;
    double attn_context_only_max_abs_diff;
    double attn_sigmoid_only_mean_abs_diff;
    double attn_sigmoid_only_max_abs_diff;
    double attn_gated_context_only_mean_abs_diff;
    double attn_gated_context_only_max_abs_diff;
    double hidden_attn_mean_abs_diff;
    double hidden_attn_max_abs_diff;
    double ffn_rms_mean_abs_diff;
    double ffn_rms_max_abs_diff;
    double ffn_gate_mean_abs_diff;
    double ffn_gate_max_abs_diff;
    double ffn_up_mean_abs_diff;
    double ffn_up_max_abs_diff;
    double ffn_silu_mean_abs_diff;
    double ffn_silu_max_abs_diff;
    double ffn_hidden_mean_abs_diff;
    double ffn_hidden_max_abs_diff;
    double ffn_down_mean_abs_diff;
    double ffn_down_max_abs_diff;
    double final_mean_abs_diff;
    double final_max_abs_diff;
    double cpu_hidden_attn_pair_gap;
    double hybrid_hidden_attn_pair_gap;
    double cpu_res_hybrid_qgate_hidden_attn_pair_gap;
    double cpu_res_hybrid_v_hidden_attn_pair_gap;
    double cpu_res_hybrid_context_hidden_attn_pair_gap;
    double cpu_res_hybrid_sigmoid_hidden_attn_pair_gap;
    double cpu_res_hybrid_gated_context_hidden_attn_pair_gap;
    double cpu_final_pair_gap;
    double hybrid_final_pair_gap;
    double cpu_res_hybrid_qgate_final_pair_gap;
    double cpu_res_hybrid_v_final_pair_gap;
    double cpu_res_hybrid_context_final_pair_gap;
    double cpu_res_hybrid_sigmoid_final_pair_gap;
    double cpu_res_hybrid_gated_context_final_pair_gap;
    int cpu_hidden_attn_pref_token;
    int hybrid_hidden_attn_pref_token;
    int cpu_res_hybrid_qgate_hidden_attn_pref_token;
    int cpu_res_hybrid_v_hidden_attn_pref_token;
    int cpu_res_hybrid_context_hidden_attn_pref_token;
    int cpu_res_hybrid_sigmoid_hidden_attn_pref_token;
    int cpu_res_hybrid_gated_context_hidden_attn_pref_token;
    int cpu_final_pref_token;
    int hybrid_final_pref_token;
    int cpu_res_hybrid_qgate_final_pref_token;
    int cpu_res_hybrid_v_final_pref_token;
    int cpu_res_hybrid_context_final_pref_token;
    int cpu_res_hybrid_sigmoid_final_pref_token;
    int cpu_res_hybrid_gated_context_final_pref_token;
    double cpu_res_hybrid_down_final_pair_gap;
    double hybrid_res_cpu_down_final_pair_gap;
    int cpu_res_hybrid_down_final_pref_token;
    int hybrid_res_cpu_down_final_pref_token;
    double cpu_res_cpu_silu_hybrid_up_final_pair_gap;
    double cpu_res_hybrid_silu_cpu_up_final_pair_gap;
    int cpu_res_cpu_silu_hybrid_up_final_pref_token;
    int cpu_res_hybrid_silu_cpu_up_final_pref_token;
    int gate_attr_total_channels;
    int gate_attr_report_count;
    int gate_attr_first_cumulative_flip_rank;
    int gate_attr_flip_channel;
    double gate_attr_flip_delta_hidden;
    double gate_attr_flip_cumulative_pair_gap;
    double gate_attr_all_cumulative_pair_gap;
    int gate_input_attr_channel_limit;
    int gate_input_attr_count;
    int attn_v_attr_count;
    int attn_gated_context_attr_count;
    int ffn_down_attr_count;
    int gate_attr_channels[ORION_GATE_ATTR_REPORT_MAX];
    double gate_attr_cpu_gate[ORION_GATE_ATTR_REPORT_MAX];
    double gate_attr_hybrid_gate[ORION_GATE_ATTR_REPORT_MAX];
    double gate_attr_delta_gate[ORION_GATE_ATTR_REPORT_MAX];
    double gate_attr_cpu_silu[ORION_GATE_ATTR_REPORT_MAX];
    double gate_attr_hybrid_silu[ORION_GATE_ATTR_REPORT_MAX];
    double gate_attr_delta_silu[ORION_GATE_ATTR_REPORT_MAX];
    double gate_attr_cpu_up[ORION_GATE_ATTR_REPORT_MAX];
    double gate_attr_delta_hidden[ORION_GATE_ATTR_REPORT_MAX];
    double gate_attr_single_pair_gap[ORION_GATE_ATTR_REPORT_MAX];
    int gate_attr_single_pref_token[ORION_GATE_ATTR_REPORT_MAX];
    double gate_attr_cumulative_pair_gap[ORION_GATE_ATTR_REPORT_MAX];
    int gate_attr_cumulative_pref_token[ORION_GATE_ATTR_REPORT_MAX];
    int gate_input_attr_dims[ORION_GATE_INPUT_ATTR_REPORT_MAX];
    double gate_input_attr_abs_contrib[ORION_GATE_INPUT_ATTR_REPORT_MAX];
    double gate_input_attr_signed_contrib[ORION_GATE_INPUT_ATTR_REPORT_MAX];
    double gate_input_attr_delta_ffn_rms[ORION_GATE_INPUT_ATTR_REPORT_MAX];
    int attn_v_attr_dims[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_v_attr_cpu_norm[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_v_attr_hybrid_norm[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_v_attr_delta_norm[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_v_attr_pair_delta[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_v_attr_abs_contrib[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_v_attr_signed_contrib[ORION_ATTN_V_ATTR_REPORT_MAX];
    int attn_gated_context_attr_dims[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_gated_context_attr_cpu_norm[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_gated_context_attr_hybrid_norm[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_gated_context_attr_delta_norm[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_gated_context_attr_pair_delta[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_gated_context_attr_abs_contrib[ORION_ATTN_V_ATTR_REPORT_MAX];
    double attn_gated_context_attr_signed_contrib[ORION_ATTN_V_ATTR_REPORT_MAX];
    int ffn_down_attr_dims[ORION_FFN_DOWN_ATTR_REPORT_MAX];
    double ffn_down_attr_cpu_norm[ORION_FFN_DOWN_ATTR_REPORT_MAX];
    double ffn_down_attr_hybrid_norm[ORION_FFN_DOWN_ATTR_REPORT_MAX];
    double ffn_down_attr_delta_norm[ORION_FFN_DOWN_ATTR_REPORT_MAX];
    double ffn_down_attr_pair_delta[ORION_FFN_DOWN_ATTR_REPORT_MAX];
    double ffn_down_attr_abs_contrib[ORION_FFN_DOWN_ATTR_REPORT_MAX];
    double ffn_down_attr_signed_contrib[ORION_FFN_DOWN_ATTR_REPORT_MAX];
} OrionFullInputStageSensitivity;

static void l2norm_vec_local(const float *x, int dim, float eps, float *out) {
    float sumsq = 0.0f;
    for (int i = 0; i < dim; i++) sumsq += x[i] * x[i];
    float inv = 1.0f / sqrtf(sumsq + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * inv;
}

static void qwen_rmsnorm_gated_local(const float *x,
                                     const float *gate,
                                     const float *weight,
                                     int dim,
                                     float eps,
                                     float *out) {
    float mean_sq = 0.0f;
    for (int i = 0; i < dim; i++) mean_sq += x[i] * x[i];
    mean_sq /= (float)dim;
    float inv = 1.0f / sqrtf(mean_sq + eps);
    for (int i = 0; i < dim; i++) {
        float sig = 1.0f / (1.0f + expf(-gate[i]));
        out[i] = (x[i] * inv) * weight[i] * (gate[i] * sig);
    }
}

static inline float sigmoid_scalar_local(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void silu_vec_local(const float *x, int n, float *out) {
    for (int i = 0; i < n; i++) out[i] = x[i] * sigmoid_scalar_local(x[i]);
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

static int capture_attention_gate_context_from_projected_qkv_local(const float *q_proj_out_seq,
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
                                                                   float *gate_sigmoid_out,
                                                                   float *context_out,
                                                                   float *gated_context_out) {
    if (!q_proj_out_seq || !k_proj_out_seq || !v_proj_out_seq || !q_norm || !k_norm ||
        !gate_sigmoid_out || !context_out || !gated_context_out ||
        seq_len <= 0 || n_head <= 0 || n_kv_head <= 0 || head_dim <= 0) {
        return 0;
    }

    const int q_dim = n_head * head_dim;
    const int kv_dim = n_kv_head * head_dim;
    const int q_per_kv = n_head / n_kv_head;
    const float scale = 1.0f / sqrtf((float)head_dim);

    float *gate_raw = (float *)calloc((size_t)seq_len * q_dim, sizeof(float));
    float *q_normed = (float *)calloc((size_t)seq_len * q_dim, sizeof(float));
    float *k_normed = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    float *q_rope = (float *)calloc((size_t)seq_len * q_dim, sizeof(float));
    float *k_rope = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    float *scores = (float *)calloc((size_t)seq_len * seq_len, sizeof(float));
    float *probs = (float *)calloc((size_t)seq_len * seq_len, sizeof(float));
    float *qh = (float *)calloc((size_t)seq_len * head_dim, sizeof(float));
    float *kh = (float *)calloc((size_t)seq_len * head_dim, sizeof(float));
    float *vh = (float *)calloc((size_t)seq_len * head_dim, sizeof(float));
    float *context_h = (float *)calloc((size_t)seq_len * head_dim, sizeof(float));
    if (!gate_raw || !q_normed || !k_normed || !q_rope || !k_rope ||
        !scores || !probs || !qh || !kh || !vh || !context_h) {
        free(gate_raw); free(q_normed); free(k_normed); free(q_rope); free(k_rope);
        free(scores); free(probs); free(qh); free(kh); free(vh); free(context_h);
        return 0;
    }

    memset(context_out, 0, (size_t)seq_len * q_dim * sizeof(float));
    memset(gated_context_out, 0, (size_t)seq_len * q_dim * sizeof(float));

    for (int s = 0; s < seq_len; s++) {
        const float *q_src = q_proj_out_seq + (size_t)s * (q_dim * 2);
        memcpy(q_normed + s * q_dim, q_src, (size_t)q_dim * sizeof(float));
        memcpy(gate_raw + s * q_dim, q_src + q_dim, (size_t)q_dim * sizeof(float));
        memcpy(k_normed + s * kv_dim, k_proj_out_seq + (size_t)s * kv_dim, (size_t)kv_dim * sizeof(float));
        for (int i = 0; i < q_dim; i++) {
            gate_sigmoid_out[s * q_dim + i] = sigmoid_scalar_local(gate_raw[s * q_dim + i]);
        }
    }

    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < n_head; h++) {
            float *q_head = q_normed + s * q_dim + h * head_dim;
            l2norm_vec_local(q_head, head_dim, 1e-6f, q_head);
            for (int i = 0; i < head_dim; i++) q_head[i] *= q_norm[i] * scale;
        }
        for (int kvh = 0; kvh < n_kv_head; kvh++) {
            float *k_head = k_normed + s * kv_dim + kvh * head_dim;
            l2norm_vec_local(k_head, head_dim, 1e-6f, k_head);
            for (int i = 0; i < head_dim; i++) k_head[i] *= k_norm[i];
        }
    }

    memcpy(q_rope, q_normed, (size_t)seq_len * q_dim * sizeof(float));
    memcpy(k_rope, k_normed, (size_t)seq_len * kv_dim * sizeof(float));
    apply_rope_text_inplace_local(q_rope, k_rope, seq_len, n_head, n_kv_head, head_dim, rope_theta, partial_rotary_factor);

    for (int h = 0; h < n_head; h++) {
        const int kv_head = h / q_per_kv;
        memset(scores, 0, (size_t)seq_len * seq_len * sizeof(float));
        memset(probs, 0, (size_t)seq_len * seq_len * sizeof(float));
        memset(context_h, 0, (size_t)seq_len * head_dim * sizeof(float));

        for (int s = 0; s < seq_len; s++) {
            memcpy(qh + s * head_dim, q_rope + s * q_dim + h * head_dim, (size_t)head_dim * sizeof(float));
            memcpy(kh + s * head_dim, k_rope + s * kv_dim + kv_head * head_dim, (size_t)head_dim * sizeof(float));
            memcpy(vh + s * head_dim, v_proj_out_seq + s * kv_dim + kv_head * head_dim, (size_t)head_dim * sizeof(float));
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    seq_len, seq_len, head_dim,
                    scale, qh, head_dim, kh, head_dim,
                    0.0f, scores, seq_len);

        for (int i = 0; i < seq_len; i++) {
            for (int j = i + 1; j < seq_len; j++) scores[i * seq_len + j] = -INFINITY;
            float max_val = scores[i * seq_len];
            for (int j = 1; j <= i; j++) {
                if (scores[i * seq_len + j] > max_val) max_val = scores[i * seq_len + j];
            }
            float sum = 0.0f;
            for (int j = 0; j <= i; j++) {
                probs[i * seq_len + j] = expf(scores[i * seq_len + j] - max_val);
                sum += probs[i * seq_len + j];
            }
            for (int j = 0; j <= i; j++) probs[i * seq_len + j] /= sum;
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    seq_len, head_dim, seq_len,
                    1.0f, probs, seq_len, vh, head_dim,
                    0.0f, context_h, head_dim);

        for (int s = 0; s < seq_len; s++) {
            memcpy(context_out + s * q_dim + h * head_dim,
                   context_h + s * head_dim,
                   (size_t)head_dim * sizeof(float));
        }
    }

    for (int i = 0; i < seq_len * q_dim; i++) {
        gated_context_out[i] = context_out[i] * gate_sigmoid_out[i];
    }

    free(gate_raw); free(q_normed); free(k_normed); free(q_rope); free(k_rope);
    free(scores); free(probs); free(qh); free(kh); free(vh); free(context_h);
    return 1;
}

static int capture_full_attention_layer_outputs(const float *hidden_in,
                                                int seq_len,
                                                int d_model,
                                                int d_ff,
                                                int n_head,
                                                int n_kv_head,
                                                int head_dim,
                                                float rope_theta,
                                                float partial_rotary_factor,
                                                const float *input_ln,
                                                const float *post_ln,
                                                const float *q_proj,
                                                const float *k_proj,
                                                const float *v_proj,
                                                const float *o_proj,
                                                const float *q_norm,
                                                const float *k_norm,
                                                const float *gate_proj,
                                                const float *up_proj,
                                                const float *down_proj,
                                                float *normed_out,
                                                float *q_proj_out,
                                                float *k_proj_out,
                                                float *v_proj_out,
                                                float *attn_out,
                                                float *hidden_attn_out,
                                                float *ffn_rms_out,
                                                float *ffn_gate_out,
                                                float *ffn_up_out,
                                                float *ffn_silu_out,
                                                float *ffn_hidden_out,
                                                float *ffn_down_out,
                                                float *final_out) {
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_kv_head * head_dim;
    const int total = seq_len * d_model;
    const int total_ff = seq_len * d_ff;

    for (int s = 0; s < seq_len; s++) {
        orion_qwen_cpu_rmsnorm(hidden_in + s * d_model, input_ln, d_model, 1e-6f, normed_out + s * d_model);
    }

    cpu_linear_batch(normed_out, seq_len, q_proj, d_model, q_dim * 2, q_proj_out);
    cpu_linear_batch(normed_out, seq_len, k_proj, d_model, kv_dim, k_proj_out);
    cpu_linear_batch(normed_out, seq_len, v_proj, d_model, kv_dim, v_proj_out);

    orion_qwen_cpu_full_attention_from_projections_with_rope(
        q_proj_out, k_proj_out, v_proj_out, seq_len, o_proj, q_norm, k_norm,
        d_model, n_head, n_kv_head, head_dim, rope_theta, partial_rotary_factor,
        attn_out
    );

    for (int i = 0; i < total; i++) hidden_attn_out[i] = hidden_in[i] + attn_out[i];
    for (int s = 0; s < seq_len; s++) {
        orion_qwen_cpu_rmsnorm(hidden_attn_out + s * d_model, post_ln, d_model, 1e-6f, ffn_rms_out + s * d_model);
    }

    cpu_linear_batch(ffn_rms_out, seq_len, gate_proj, d_model, d_ff, ffn_gate_out);
    cpu_linear_batch(ffn_rms_out, seq_len, up_proj, d_model, d_ff, ffn_up_out);
    silu_vec_local(ffn_gate_out, total_ff, ffn_silu_out);
    for (int i = 0; i < total_ff; i++) ffn_hidden_out[i] = ffn_silu_out[i] * ffn_up_out[i];
    cpu_linear_batch(ffn_hidden_out, seq_len, down_proj, d_ff, d_model, ffn_down_out);
    for (int i = 0; i < total; i++) final_out[i] = hidden_attn_out[i] + ffn_down_out[i];
    return 1;
}

static int capture_linear_layer_outputs(const float *hidden_in,
                                        int seq_len,
                                        int d_model,
                                        int d_ff,
                                        int num_k_heads,
                                        int num_v_heads,
                                        int head_k_dim,
                                        int head_v_dim,
                                        int conv_kernel,
                                        const float *input_ln,
                                        const float *post_ln,
                                        const float *gate_proj,
                                        const float *up_proj,
                                        const float *down_proj,
                                        const float *in_proj_qkv,
                                        const float *in_proj_z,
                                        const float *in_proj_a,
                                        const float *in_proj_b,
                                        const float *conv1d,
                                        const float *dt_bias,
                                        const float *a_log,
                                        const float *norm_weight,
                                        const float *out_proj,
                                        float *normed_out,
                                        float *mixed_linear_out,
                                        float *mixed_conv_out,
                                        float *query_out,
                                        float *key_out,
                                        float *value_out,
                                        float *z_out,
                                        float *beta_out,
                                        float *g_out,
                                        float *core_pre_out,
                                        float *core_out,
                                        float *attn_out,
                                        float *hidden_attn_out,
                                        float *final_out) {
    const int key_dim = num_k_heads * head_k_dim;
    const int value_dim = num_v_heads * head_v_dim;
    const int conv_dim = key_dim * 2 + value_dim;
    const float scale = 1.0f / sqrtf((float)head_k_dim);

    float *state = (float *)calloc((size_t)num_v_heads * head_k_dim * head_v_dim, sizeof(float));
    float *q_norm = (float *)calloc((size_t)head_k_dim, sizeof(float));
    float *k_norm = (float *)calloc((size_t)head_k_dim, sizeof(float));
    float *kv_mem = (float *)calloc((size_t)head_v_dim, sizeof(float));
    float *delta = (float *)calloc((size_t)head_v_dim, sizeof(float));
    float *core_head = (float *)calloc((size_t)head_v_dim, sizeof(float));
    if (!state || !q_norm || !k_norm || !kv_mem || !delta || !core_head) {
        free(state);
        free(q_norm);
        free(k_norm);
        free(kv_mem);
        free(delta);
        free(core_head);
        return 0;
    }

    for (int s = 0; s < seq_len; s++) {
        orion_qwen_cpu_rmsnorm(hidden_in + s * d_model, input_ln, d_model, 1e-6f, normed_out + s * d_model);
    }

    cpu_linear_batch(normed_out, seq_len, in_proj_qkv, d_model, conv_dim, mixed_linear_out);
    cpu_linear_batch(normed_out, seq_len, in_proj_z, d_model, value_dim, z_out);
    cpu_linear_batch(normed_out, seq_len, in_proj_b, d_model, num_v_heads, beta_out);
    cpu_linear_batch(normed_out, seq_len, in_proj_a, d_model, num_v_heads, g_out);

    for (int t = 0; t < seq_len; t++) {
        for (int c = 0; c < conv_dim; c++) {
            float sum = 0.0f;
            const float *kernel = conv1d + c * conv_kernel;
            for (int k = 0; k < conv_kernel; k++) {
                int src_t = t - (conv_kernel - 1) + k;
                float x = 0.0f;
                if (src_t >= 0 && src_t < seq_len) {
                    x = mixed_linear_out[src_t * conv_dim + c];
                }
                sum += kernel[k] * x;
            }
            float sig = 1.0f / (1.0f + expf(-sum));
            mixed_conv_out[t * conv_dim + c] = sum * sig;
        }

        memcpy(query_out + t * key_dim, mixed_conv_out + t * conv_dim, (size_t)key_dim * sizeof(float));
        memcpy(key_out + t * key_dim, mixed_conv_out + t * conv_dim + key_dim, (size_t)key_dim * sizeof(float));
        memcpy(value_out + t * value_dim, mixed_conv_out + t * conv_dim + key_dim * 2, (size_t)value_dim * sizeof(float));
    }

    for (int i = 0; i < seq_len * num_v_heads; i++) {
        beta_out[i] = 1.0f / (1.0f + expf(-beta_out[i]));
    }
    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < num_v_heads; h++) {
            float a = g_out[t * num_v_heads + h];
            float dt = dt_bias[h];
            float al = a_log[h];
            float softplus = (a + dt > 20.0f) ? (a + dt) : log1pf(expf(a + dt));
            g_out[t * num_v_heads + h] = -expf(al) * softplus;
        }
    }

    for (int t = 0; t < seq_len; t++) {
        for (int h = 0; h < num_v_heads; h++) {
            const float *q_t = query_out + t * key_dim + h * head_k_dim;
            const float *k_t = key_out + t * key_dim + h * head_k_dim;
            const float *v_t = value_out + t * value_dim + h * head_v_dim;
            const float *z_t = z_out + t * value_dim + h * head_v_dim;
            float *state_h = state + h * head_k_dim * head_v_dim;
            float *core_pre_t = core_pre_out + t * value_dim + h * head_v_dim;
            float *core_t = core_out + t * value_dim + h * head_v_dim;

            l2norm_vec_local(q_t, head_k_dim, 1e-6f, q_norm);
            l2norm_vec_local(k_t, head_k_dim, 1e-6f, k_norm);
            for (int i = 0; i < head_k_dim; i++) q_norm[i] *= scale;

            float decay = expf(g_out[t * num_v_heads + h]);
            float beta_t = beta_out[t * num_v_heads + h];
            for (int i = 0; i < head_k_dim * head_v_dim; i++) state_h[i] *= decay;

            for (int v_idx = 0; v_idx < head_v_dim; v_idx++) {
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < head_k_dim; k_idx++) {
                    sum += state_h[k_idx * head_v_dim + v_idx] * k_norm[k_idx];
                }
                kv_mem[v_idx] = sum;
                delta[v_idx] = (v_t[v_idx] - sum) * beta_t;
            }
            for (int k_idx = 0; k_idx < head_k_dim; k_idx++) {
                for (int v_idx = 0; v_idx < head_v_dim; v_idx++) {
                    state_h[k_idx * head_v_dim + v_idx] += k_norm[k_idx] * delta[v_idx];
                }
            }
            for (int v_idx = 0; v_idx < head_v_dim; v_idx++) {
                float sum = 0.0f;
                for (int k_idx = 0; k_idx < head_k_dim; k_idx++) {
                    sum += state_h[k_idx * head_v_dim + v_idx] * q_norm[k_idx];
                }
                core_pre_t[v_idx] = sum;
            }
            qwen_rmsnorm_gated_local(core_pre_t, z_t, norm_weight, head_v_dim, 1e-6f, core_t);
        }
    }

    cpu_linear_batch(core_out, seq_len, out_proj, value_dim, d_model, attn_out);
    for (int i = 0; i < seq_len * d_model; i++) hidden_attn_out[i] = hidden_in[i] + attn_out[i];
    if (!apply_cpu_ffn(hidden_attn_out, seq_len, d_model, d_ff, post_ln, gate_proj, up_proj, down_proj, final_out)) {
        free(state);
        free(q_norm);
        free(k_norm);
        free(kv_mem);
        free(delta);
        free(core_head);
        return 0;
    }

    free(state);
    free(q_norm);
    free(k_norm);
    free(kv_mem);
    free(delta);
    free(core_head);
    return 1;
}

static int run_linear_input_stage_compare(const char *blob_dir,
                                          OrionQwen35Manifest *manifest,
                                          int layer,
                                          const float *cpu_hidden_in,
                                          const float *hybrid_hidden_in,
                                          int seq_len,
                                          int candidate_a,
                                          int candidate_b,
                                          OrionLinearStageSensitivity *out) {
    if (!manifest || !cpu_hidden_in || !hybrid_hidden_in || !out) return 0;

    const int d_model = manifest->d_model;
    const int d_ff = manifest->d_ff;
    const char *lm_head_name = manifest->tie_word_embeddings ? "embed_tokens.bin" : "lm_head.bin";
    const int pair_ids[2] = {candidate_a, candidate_b};

    char path_qkv[2048], path_out[2048], path_dt[2048], path_norm[2048], path_conv[2048];
    snprintf(path_qkv, sizeof(path_qkv), "%s/layer%d/linear_attn_in_proj_qkv.bin", blob_dir, layer);
    snprintf(path_out, sizeof(path_out), "%s/layer%d/linear_attn_out_proj.bin", blob_dir, layer);
    snprintf(path_dt, sizeof(path_dt), "%s/layer%d/linear_attn_dt_bias.bin", blob_dir, layer);
    snprintf(path_norm, sizeof(path_norm), "%s/layer%d/linear_attn_norm.bin", blob_dir, layer);
    snprintf(path_conv, sizeof(path_conv), "%s/layer%d/linear_attn_conv1d.bin", blob_dir, layer);

    const int qkv_rows = orion_blob_element_count(path_qkv) / d_model;
    const int value_dim = orion_blob_element_count(path_out) / d_model;
    const int num_v_heads = orion_blob_element_count(path_dt);
    const int head_v_dim = orion_blob_element_count(path_norm);
    const int key_dim = (qkv_rows - value_dim) / 2;
    const int num_k_heads = num_v_heads;
    const int head_k_dim = key_dim / num_k_heads;
    const int conv_kernel = orion_blob_element_count(path_conv) / qkv_rows;
    const int conv_dim = qkv_rows;
    const int total = seq_len * d_model;
    const int total_conv = seq_len * conv_dim;
    const int total_key = seq_len * key_dim;
    const int total_value = seq_len * value_dim;
    const int total_heads = seq_len * num_v_heads;

    float *input_ln = load_exact(blob_dir, layer, "input_layernorm.bin", d_model);
    float *post_ln = load_exact(blob_dir, layer, "post_attention_layernorm.bin", d_model);
    float *gate_proj = load_exact(blob_dir, layer, "mlp_gate_proj.bin", d_ff * d_model);
    float *up_proj = load_exact(blob_dir, layer, "mlp_up_proj.bin", d_ff * d_model);
    float *down_proj = load_exact(blob_dir, layer, "mlp_down_proj.bin", d_model * d_ff);
    float *in_proj_qkv = load_exact(blob_dir, layer, "linear_attn_in_proj_qkv.bin", qkv_rows * d_model);
    float *in_proj_z = load_exact(blob_dir, layer, "linear_attn_in_proj_z.bin", value_dim * d_model);
    float *in_proj_a = load_exact(blob_dir, layer, "linear_attn_in_proj_a.bin", num_v_heads * d_model);
    float *in_proj_b = load_exact(blob_dir, layer, "linear_attn_in_proj_b.bin", num_v_heads * d_model);
    float *conv1d = load_exact(blob_dir, layer, "linear_attn_conv1d.bin", qkv_rows * conv_kernel);
    float *dt_bias = load_exact(blob_dir, layer, "linear_attn_dt_bias.bin", num_v_heads);
    float *a_log = load_exact(blob_dir, layer, "linear_attn_a_log.bin", num_v_heads);
    float *norm_weight = load_exact(blob_dir, layer, "linear_attn_norm.bin", head_v_dim);
    float *out_proj = load_exact(blob_dir, layer, "linear_attn_out_proj.bin", d_model * value_dim);
    float *final_norm = NULL;

    float *cpu_normed = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_normed = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_mixed_linear = (float *)calloc((size_t)total_conv, sizeof(float));
    float *hybrid_mixed_linear = (float *)calloc((size_t)total_conv, sizeof(float));
    float *cpu_mixed_conv = (float *)calloc((size_t)total_conv, sizeof(float));
    float *hybrid_mixed_conv = (float *)calloc((size_t)total_conv, sizeof(float));
    float *cpu_query = (float *)calloc((size_t)total_key, sizeof(float));
    float *hybrid_query = (float *)calloc((size_t)total_key, sizeof(float));
    float *cpu_key = (float *)calloc((size_t)total_key, sizeof(float));
    float *hybrid_key = (float *)calloc((size_t)total_key, sizeof(float));
    float *cpu_value = (float *)calloc((size_t)total_value, sizeof(float));
    float *hybrid_value = (float *)calloc((size_t)total_value, sizeof(float));
    float *cpu_z = (float *)calloc((size_t)total_value, sizeof(float));
    float *hybrid_z = (float *)calloc((size_t)total_value, sizeof(float));
    float *cpu_beta = (float *)calloc((size_t)total_heads, sizeof(float));
    float *hybrid_beta = (float *)calloc((size_t)total_heads, sizeof(float));
    float *cpu_g = (float *)calloc((size_t)total_heads, sizeof(float));
    float *hybrid_g = (float *)calloc((size_t)total_heads, sizeof(float));
    float *cpu_core_pre = (float *)calloc((size_t)total_value, sizeof(float));
    float *hybrid_core_pre = (float *)calloc((size_t)total_value, sizeof(float));
    float *cpu_core = (float *)calloc((size_t)total_value, sizeof(float));
    float *hybrid_core = (float *)calloc((size_t)total_value, sizeof(float));
    float *cpu_attn_out = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_attn_out = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_final = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_final = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_last = (float *)calloc((size_t)d_model, sizeof(float));
    float *hybrid_last = (float *)calloc((size_t)d_model, sizeof(float));

    char final_norm_path[2048];
    snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
    final_norm = orion_read_blob_f32_exact(final_norm_path, d_model);

    if (!input_ln || !post_ln || !gate_proj || !up_proj || !down_proj || !in_proj_qkv || !in_proj_z ||
        !in_proj_a || !in_proj_b || !conv1d || !dt_bias || !a_log || !norm_weight || !out_proj ||
        !final_norm || !cpu_normed || !hybrid_normed || !cpu_mixed_linear || !hybrid_mixed_linear ||
        !cpu_mixed_conv || !hybrid_mixed_conv || !cpu_query || !hybrid_query || !cpu_key || !hybrid_key ||
        !cpu_value || !hybrid_value || !cpu_z || !hybrid_z || !cpu_beta || !hybrid_beta || !cpu_g || !hybrid_g ||
        !cpu_core_pre || !hybrid_core_pre || !cpu_core || !hybrid_core || !cpu_attn_out || !hybrid_attn_out ||
        !cpu_hidden_attn || !hybrid_hidden_attn || !cpu_final || !hybrid_final || !cpu_last || !hybrid_last) {
        goto fail;
    }

    if (!capture_linear_layer_outputs(cpu_hidden_in, seq_len, d_model, d_ff, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                                      conv_kernel, input_ln, post_ln, gate_proj, up_proj, down_proj,
                                      in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, dt_bias, a_log, norm_weight, out_proj,
                                      cpu_normed, cpu_mixed_linear, cpu_mixed_conv, cpu_query, cpu_key, cpu_value, cpu_z, cpu_beta, cpu_g,
                                      cpu_core_pre, cpu_core, cpu_attn_out, cpu_hidden_attn, cpu_final) ||
        !capture_linear_layer_outputs(hybrid_hidden_in, seq_len, d_model, d_ff, num_k_heads, num_v_heads, head_k_dim, head_v_dim,
                                      conv_kernel, input_ln, post_ln, gate_proj, up_proj, down_proj,
                                      in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, conv1d, dt_bias, a_log, norm_weight, out_proj,
                                      hybrid_normed, hybrid_mixed_linear, hybrid_mixed_conv, hybrid_query, hybrid_key, hybrid_value, hybrid_z, hybrid_beta, hybrid_g,
                                      hybrid_core_pre, hybrid_core, hybrid_attn_out, hybrid_hidden_attn, hybrid_final)) {
        goto fail;
    }

    out->normed_mean_abs_diff = mean_abs_diff(cpu_normed, hybrid_normed, total);
    out->normed_max_abs_diff = max_abs_diff(cpu_normed, hybrid_normed, total);
    out->mixed_linear_mean_abs_diff = mean_abs_diff(cpu_mixed_linear, hybrid_mixed_linear, total_conv);
    out->mixed_linear_max_abs_diff = max_abs_diff(cpu_mixed_linear, hybrid_mixed_linear, total_conv);
    out->mixed_conv_mean_abs_diff = mean_abs_diff(cpu_mixed_conv, hybrid_mixed_conv, total_conv);
    out->mixed_conv_max_abs_diff = max_abs_diff(cpu_mixed_conv, hybrid_mixed_conv, total_conv);
    out->query_mean_abs_diff = mean_abs_diff(cpu_query, hybrid_query, total_key);
    out->query_max_abs_diff = max_abs_diff(cpu_query, hybrid_query, total_key);
    out->key_mean_abs_diff = mean_abs_diff(cpu_key, hybrid_key, total_key);
    out->key_max_abs_diff = max_abs_diff(cpu_key, hybrid_key, total_key);
    out->value_mean_abs_diff = mean_abs_diff(cpu_value, hybrid_value, total_value);
    out->value_max_abs_diff = max_abs_diff(cpu_value, hybrid_value, total_value);
    out->z_mean_abs_diff = mean_abs_diff(cpu_z, hybrid_z, total_value);
    out->z_max_abs_diff = max_abs_diff(cpu_z, hybrid_z, total_value);
    out->beta_mean_abs_diff = mean_abs_diff(cpu_beta, hybrid_beta, total_heads);
    out->beta_max_abs_diff = max_abs_diff(cpu_beta, hybrid_beta, total_heads);
    out->g_mean_abs_diff = mean_abs_diff(cpu_g, hybrid_g, total_heads);
    out->g_max_abs_diff = max_abs_diff(cpu_g, hybrid_g, total_heads);
    out->core_pre_mean_abs_diff = mean_abs_diff(cpu_core_pre, hybrid_core_pre, total_value);
    out->core_pre_max_abs_diff = max_abs_diff(cpu_core_pre, hybrid_core_pre, total_value);
    out->core_mean_abs_diff = mean_abs_diff(cpu_core, hybrid_core, total_value);
    out->core_max_abs_diff = max_abs_diff(cpu_core, hybrid_core, total_value);
    out->attn_out_mean_abs_diff = mean_abs_diff(cpu_attn_out, hybrid_attn_out, total);
    out->attn_out_max_abs_diff = max_abs_diff(cpu_attn_out, hybrid_attn_out, total);
    out->hidden_attn_mean_abs_diff = mean_abs_diff(cpu_hidden_attn, hybrid_hidden_attn, total);
    out->hidden_attn_max_abs_diff = max_abs_diff(cpu_hidden_attn, hybrid_hidden_attn, total);
    out->final_mean_abs_diff = mean_abs_diff(cpu_final, hybrid_final, total);
    out->final_max_abs_diff = max_abs_diff(cpu_final, hybrid_final, total);

    orion_qwen_cpu_rmsnorm(cpu_hidden_attn + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_last);
    orion_qwen_cpu_rmsnorm(hybrid_hidden_attn + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, hybrid_last);
    float cpu_hidden_pair_logits[2] = {0.0f, 0.0f};
    float hybrid_hidden_pair_logits[2] = {0.0f, 0.0f};
    float cpu_final_pair_logits[2] = {0.0f, 0.0f};
    float hybrid_final_pair_logits[2] = {0.0f, 0.0f};
    if (!selected_token_logits(blob_dir, lm_head_name, cpu_last, d_model, pair_ids, 2, cpu_hidden_pair_logits) ||
        !selected_token_logits(blob_dir, lm_head_name, hybrid_last, d_model, pair_ids, 2, hybrid_hidden_pair_logits)) {
        goto fail;
    }
    orion_qwen_cpu_rmsnorm(cpu_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_last);
    orion_qwen_cpu_rmsnorm(hybrid_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, hybrid_last);
    if (!selected_token_logits(blob_dir, lm_head_name, cpu_last, d_model, pair_ids, 2, cpu_final_pair_logits) ||
        !selected_token_logits(blob_dir, lm_head_name, hybrid_last, d_model, pair_ids, 2, hybrid_final_pair_logits)) {
        goto fail;
    }

    out->cpu_hidden_attn_pair_gap = (double)cpu_hidden_pair_logits[0] - (double)cpu_hidden_pair_logits[1];
    out->hybrid_hidden_attn_pair_gap = (double)hybrid_hidden_pair_logits[0] - (double)hybrid_hidden_pair_logits[1];
    out->cpu_final_pair_gap = (double)cpu_final_pair_logits[0] - (double)cpu_final_pair_logits[1];
    out->hybrid_final_pair_gap = (double)hybrid_final_pair_logits[0] - (double)hybrid_final_pair_logits[1];
    out->cpu_hidden_attn_pref_token = (out->cpu_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->hybrid_hidden_attn_pref_token = (out->hybrid_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_final_pref_token = (out->cpu_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->hybrid_final_pref_token = (out->hybrid_final_pair_gap >= 0.0) ? candidate_a : candidate_b;

    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
    free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b); free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
    free(final_norm);
    free(cpu_normed); free(hybrid_normed); free(cpu_mixed_linear); free(hybrid_mixed_linear); free(cpu_mixed_conv); free(hybrid_mixed_conv);
    free(cpu_query); free(hybrid_query); free(cpu_key); free(hybrid_key); free(cpu_value); free(hybrid_value); free(cpu_z); free(hybrid_z);
    free(cpu_beta); free(hybrid_beta); free(cpu_g); free(hybrid_g); free(cpu_core_pre); free(hybrid_core_pre); free(cpu_core); free(hybrid_core);
    free(cpu_attn_out); free(hybrid_attn_out); free(cpu_hidden_attn); free(hybrid_hidden_attn); free(cpu_final); free(hybrid_final);
    free(cpu_last); free(hybrid_last);
    return 1;

fail:
    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
    free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b); free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
    free(final_norm);
    free(cpu_normed); free(hybrid_normed); free(cpu_mixed_linear); free(hybrid_mixed_linear); free(cpu_mixed_conv); free(hybrid_mixed_conv);
    free(cpu_query); free(hybrid_query); free(cpu_key); free(hybrid_key); free(cpu_value); free(hybrid_value); free(cpu_z); free(hybrid_z);
    free(cpu_beta); free(hybrid_beta); free(cpu_g); free(hybrid_g); free(cpu_core_pre); free(hybrid_core_pre); free(cpu_core); free(hybrid_core);
    free(cpu_attn_out); free(hybrid_attn_out); free(cpu_hidden_attn); free(hybrid_hidden_attn); free(cpu_final); free(hybrid_final);
    free(cpu_last); free(hybrid_last);
    return 0;
}

static int run_full_input_stage_compare(const char *blob_dir,
                                        OrionQwen35Manifest *manifest,
                                        int layer,
                                        const float *cpu_hidden_in,
                                        const float *hybrid_hidden_in,
                                        int seq_len,
                                        int candidate_a,
                                        int candidate_b,
                                        int gate_attr_report_topk,
                                        OrionFullInputStageSensitivity *out) {
    if (!manifest || !cpu_hidden_in || !hybrid_hidden_in || !out) return 0;

    const int d_model = manifest->d_model;
    const int d_ff = manifest->d_ff;
    const int n_head = manifest->n_head;
    const int n_kv_head = manifest->n_kv_head;
    const int head_dim = manifest->head_dim;
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_kv_head * head_dim;
    const int total = seq_len * d_model;
    const int total_q = seq_len * (q_dim * 2);
    const int total_kv = seq_len * kv_dim;
    const int total_context = seq_len * q_dim;
    const int total_ff = seq_len * d_ff;
    const char *lm_head_name = manifest->tie_word_embeddings ? "embed_tokens.bin" : "lm_head.bin";
    const int pair_ids[2] = {candidate_a, candidate_b};

    float *input_ln = load_exact(blob_dir, layer, "input_layernorm.bin", d_model);
    float *post_ln = load_exact(blob_dir, layer, "post_attention_layernorm.bin", d_model);
    float *q_proj = load_exact(blob_dir, layer, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
    float *k_proj = load_exact(blob_dir, layer, "self_attn_k_proj.bin", kv_dim * d_model);
    float *v_proj = load_exact(blob_dir, layer, "self_attn_v_proj.bin", kv_dim * d_model);
    float *o_proj = load_exact(blob_dir, layer, "self_attn_o_proj.bin", d_model * q_dim);
    float *q_norm = load_exact(blob_dir, layer, "self_attn_q_norm.bin", head_dim);
    float *k_norm = load_exact(blob_dir, layer, "self_attn_k_norm.bin", head_dim);
    float *gate_proj = load_exact(blob_dir, layer, "mlp_gate_proj.bin", d_ff * d_model);
    float *up_proj = load_exact(blob_dir, layer, "mlp_up_proj.bin", d_ff * d_model);
    float *down_proj = load_exact(blob_dir, layer, "mlp_down_proj.bin", d_model * d_ff);
    float *final_norm = NULL;

    float *cpu_normed = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_normed = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_q = (float *)calloc((size_t)total_q, sizeof(float));
    float *hybrid_q = (float *)calloc((size_t)total_q, sizeof(float));
    float *cpu_k = (float *)calloc((size_t)total_kv, sizeof(float));
    float *hybrid_k = (float *)calloc((size_t)total_kv, sizeof(float));
    float *cpu_v = (float *)calloc((size_t)total_kv, sizeof(float));
    float *hybrid_v = (float *)calloc((size_t)total_kv, sizeof(float));
    float *cpu_gate_sigmoid = (float *)calloc((size_t)total_context, sizeof(float));
    float *hybrid_gate_sigmoid = (float *)calloc((size_t)total_context, sizeof(float));
    float *cpu_context = (float *)calloc((size_t)total_context, sizeof(float));
    float *hybrid_context = (float *)calloc((size_t)total_context, sizeof(float));
    float *cpu_gated_context = (float *)calloc((size_t)total_context, sizeof(float));
    float *hybrid_gated_context = (float *)calloc((size_t)total_context, sizeof(float));
    float *cpu_attn = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_q_hybrid_gate = (float *)calloc((size_t)total_q, sizeof(float));
    float *cpu_hybrid_qgate_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_v_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_context_gated_context = (float *)calloc((size_t)total_context, sizeof(float));
    float *cpu_hybrid_context_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_sigmoid_gated_context = (float *)calloc((size_t)total_context, sizeof(float));
    float *cpu_hybrid_sigmoid_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_gated_context_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_qgate_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_v_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_context_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_sigmoid_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_gated_context_hidden_attn = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_ffn_rms = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_ffn_rms = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_ffn_gate = (float *)calloc((size_t)total_ff, sizeof(float));
    float *hybrid_ffn_gate = (float *)calloc((size_t)total_ff, sizeof(float));
    float *cpu_ffn_up = (float *)calloc((size_t)total_ff, sizeof(float));
    float *hybrid_ffn_up = (float *)calloc((size_t)total_ff, sizeof(float));
    float *cpu_ffn_silu = (float *)calloc((size_t)total_ff, sizeof(float));
    float *hybrid_ffn_silu = (float *)calloc((size_t)total_ff, sizeof(float));
    float *cpu_ffn_hidden = (float *)calloc((size_t)total_ff, sizeof(float));
    float *hybrid_ffn_hidden = (float *)calloc((size_t)total_ff, sizeof(float));
    float *cpu_ffn_down = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_ffn_down = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_final = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_final = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_qgate_final = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_v_final = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_context_final = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_sigmoid_final = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_hybrid_gated_context_final = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_res_hybrid_down_final = (float *)calloc((size_t)total, sizeof(float));
    float *hybrid_res_cpu_down_final = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_silu_hybrid_up_hidden = (float *)calloc((size_t)total_ff, sizeof(float));
    float *hybrid_silu_cpu_up_hidden = (float *)calloc((size_t)total_ff, sizeof(float));
    float *cpu_res_cpu_silu_hybrid_up_down = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_res_hybrid_silu_cpu_up_down = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_res_cpu_silu_hybrid_up_final = (float *)calloc((size_t)total, sizeof(float));
    float *cpu_res_hybrid_silu_cpu_up_final = (float *)calloc((size_t)total, sizeof(float));
    float *gate_attr_single_last = (float *)calloc((size_t)d_model, sizeof(float));
    float *gate_attr_cumulative_last = (float *)calloc((size_t)d_model, sizeof(float));
    float *gate_attr_norm_last = (float *)calloc((size_t)d_model, sizeof(float));
    OrionGateAttrImpact *gate_attr_impacts = (OrionGateAttrImpact *)calloc((size_t)d_ff, sizeof(OrionGateAttrImpact));
    OrionGateInputAttrImpact *gate_input_attr_impacts = (OrionGateInputAttrImpact *)calloc((size_t)d_model, sizeof(OrionGateInputAttrImpact));
    OrionAttnVAttrImpact *attn_v_attr_impacts = (OrionAttnVAttrImpact *)calloc((size_t)d_model, sizeof(OrionAttnVAttrImpact));
    OrionAttnVAttrImpact *attn_gated_context_attr_impacts = (OrionAttnVAttrImpact *)calloc((size_t)d_model, sizeof(OrionAttnVAttrImpact));
    OrionFfnDownAttrImpact *ffn_down_attr_impacts = (OrionFfnDownAttrImpact *)calloc((size_t)d_model, sizeof(OrionFfnDownAttrImpact));
    float *pair_delta = (float *)calloc((size_t)d_model, sizeof(float));
    float *cpu_last = (float *)calloc((size_t)d_model, sizeof(float));
    float *hybrid_last = (float *)calloc((size_t)d_model, sizeof(float));
    float *cpu_v_hidden_norm_last = (float *)calloc((size_t)d_model, sizeof(float));
    float *hybrid_v_hidden_norm_last = (float *)calloc((size_t)d_model, sizeof(float));
    float *cpu_gated_context_hidden_norm_last = (float *)calloc((size_t)d_model, sizeof(float));
    float *hybrid_gated_context_hidden_norm_last = (float *)calloc((size_t)d_model, sizeof(float));
    float *cpu_ffn_down_norm_last = (float *)calloc((size_t)d_model, sizeof(float));
    float *hybrid_ffn_down_norm_last = (float *)calloc((size_t)d_model, sizeof(float));

    char final_norm_path[2048];
    snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
    final_norm = orion_read_blob_f32_exact(final_norm_path, d_model);

    if (!input_ln || !post_ln || !q_proj || !k_proj || !v_proj || !o_proj || !q_norm || !k_norm ||
        !gate_proj || !up_proj || !down_proj || !final_norm ||
        !cpu_normed || !hybrid_normed || !cpu_q || !hybrid_q || !cpu_k || !hybrid_k ||
        !cpu_v || !hybrid_v || !cpu_gate_sigmoid || !hybrid_gate_sigmoid || !cpu_context || !hybrid_context ||
        !cpu_gated_context || !hybrid_gated_context || !cpu_attn || !hybrid_attn ||
        !cpu_q_hybrid_gate || !cpu_hybrid_qgate_attn || !cpu_hybrid_v_attn ||
        !cpu_hybrid_context_gated_context || !cpu_hybrid_context_attn || !cpu_hybrid_sigmoid_gated_context ||
        !cpu_hybrid_sigmoid_attn || !cpu_hybrid_gated_context_attn ||
        !cpu_hidden_attn || !hybrid_hidden_attn ||
        !cpu_hybrid_qgate_hidden_attn || !cpu_hybrid_v_hidden_attn ||
        !cpu_hybrid_context_hidden_attn || !cpu_hybrid_sigmoid_hidden_attn || !cpu_hybrid_gated_context_hidden_attn ||
        !cpu_ffn_rms || !hybrid_ffn_rms || !cpu_ffn_gate || !hybrid_ffn_gate || !cpu_ffn_up || !hybrid_ffn_up ||
        !cpu_ffn_silu || !hybrid_ffn_silu || !cpu_ffn_hidden || !hybrid_ffn_hidden ||
        !cpu_ffn_down || !hybrid_ffn_down || !cpu_final || !hybrid_final ||
        !cpu_hybrid_qgate_final || !cpu_hybrid_v_final ||
        !cpu_hybrid_context_final || !cpu_hybrid_sigmoid_final || !cpu_hybrid_gated_context_final ||
        !cpu_res_hybrid_down_final || !hybrid_res_cpu_down_final ||
        !cpu_silu_hybrid_up_hidden || !hybrid_silu_cpu_up_hidden ||
        !cpu_res_cpu_silu_hybrid_up_down || !cpu_res_hybrid_silu_cpu_up_down ||
        !cpu_res_cpu_silu_hybrid_up_final || !cpu_res_hybrid_silu_cpu_up_final ||
        !gate_attr_single_last || !gate_attr_cumulative_last || !gate_attr_norm_last || !gate_attr_impacts || !gate_input_attr_impacts || !attn_v_attr_impacts || !attn_gated_context_attr_impacts || !ffn_down_attr_impacts ||
        !pair_delta ||
        !cpu_last || !hybrid_last || !cpu_v_hidden_norm_last || !hybrid_v_hidden_norm_last ||
        !cpu_gated_context_hidden_norm_last || !hybrid_gated_context_hidden_norm_last ||
        !cpu_ffn_down_norm_last || !hybrid_ffn_down_norm_last) {
        goto fail;
    }

    if (!capture_full_attention_layer_outputs(cpu_hidden_in, seq_len, d_model, d_ff, n_head, n_kv_head, head_dim,
                                              manifest->rope_theta, manifest->partial_rotary_factor,
                                              input_ln, post_ln, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
                                              gate_proj, up_proj, down_proj,
                                              cpu_normed, cpu_q, cpu_k, cpu_v, cpu_attn, cpu_hidden_attn,
                                              cpu_ffn_rms, cpu_ffn_gate, cpu_ffn_up, cpu_ffn_silu, cpu_ffn_hidden,
                                              cpu_ffn_down, cpu_final) ||
        !capture_full_attention_layer_outputs(hybrid_hidden_in, seq_len, d_model, d_ff, n_head, n_kv_head, head_dim,
                                              manifest->rope_theta, manifest->partial_rotary_factor,
                                              input_ln, post_ln, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
                                              gate_proj, up_proj, down_proj,
                                              hybrid_normed, hybrid_q, hybrid_k, hybrid_v, hybrid_attn, hybrid_hidden_attn,
                                              hybrid_ffn_rms, hybrid_ffn_gate, hybrid_ffn_up, hybrid_ffn_silu, hybrid_ffn_hidden,
                                              hybrid_ffn_down, hybrid_final)) {
        goto fail;
    }

    if (!capture_attention_gate_context_from_projected_qkv_local(cpu_q, cpu_k, cpu_v, seq_len,
                                                                 q_norm, k_norm, n_head, n_kv_head, head_dim,
                                                                 manifest->rope_theta, manifest->partial_rotary_factor,
                                                                 cpu_gate_sigmoid, cpu_context, cpu_gated_context) ||
        !capture_attention_gate_context_from_projected_qkv_local(hybrid_q, hybrid_k, hybrid_v, seq_len,
                                                                 q_norm, k_norm, n_head, n_kv_head, head_dim,
                                                                 manifest->rope_theta, manifest->partial_rotary_factor,
                                                                 hybrid_gate_sigmoid, hybrid_context, hybrid_gated_context)) {
        goto fail;
    }

    memcpy(cpu_q_hybrid_gate, cpu_q, (size_t)total_q * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        memcpy(cpu_q_hybrid_gate + (size_t)s * (q_dim * 2) + q_dim,
               hybrid_q + (size_t)s * (q_dim * 2) + q_dim,
               (size_t)q_dim * sizeof(float));
    }
    orion_qwen_cpu_full_attention_from_projections_with_rope(
        cpu_q_hybrid_gate, cpu_k, cpu_v, seq_len, o_proj, q_norm, k_norm,
        d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
        cpu_hybrid_qgate_attn
    );
    orion_qwen_cpu_full_attention_from_projections_with_rope(
        cpu_q, cpu_k, hybrid_v, seq_len, o_proj, q_norm, k_norm,
        d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
        cpu_hybrid_v_attn
    );

    for (int i = 0; i < total_context; i++) {
        cpu_hybrid_context_gated_context[i] = hybrid_context[i] * cpu_gate_sigmoid[i];
    }
    cpu_linear_batch(cpu_hybrid_context_gated_context, seq_len, o_proj, q_dim, d_model, cpu_hybrid_context_attn);
    for (int i = 0; i < total_context; i++) {
        cpu_hybrid_sigmoid_gated_context[i] = cpu_context[i] * hybrid_gate_sigmoid[i];
    }
    cpu_linear_batch(cpu_hybrid_sigmoid_gated_context, seq_len, o_proj, q_dim, d_model, cpu_hybrid_sigmoid_attn);
    cpu_linear_batch(hybrid_gated_context, seq_len, o_proj, q_dim, d_model, cpu_hybrid_gated_context_attn);

    for (int i = 0; i < total; i++) {
        cpu_hybrid_qgate_hidden_attn[i] = cpu_hidden_in[i] + cpu_hybrid_qgate_attn[i];
        cpu_hybrid_v_hidden_attn[i] = cpu_hidden_in[i] + cpu_hybrid_v_attn[i];
        cpu_hybrid_context_hidden_attn[i] = cpu_hidden_in[i] + cpu_hybrid_context_attn[i];
        cpu_hybrid_sigmoid_hidden_attn[i] = cpu_hidden_in[i] + cpu_hybrid_sigmoid_attn[i];
        cpu_hybrid_gated_context_hidden_attn[i] = cpu_hidden_in[i] + cpu_hybrid_gated_context_attn[i];
    }
    if (!apply_cpu_ffn(cpu_hybrid_qgate_hidden_attn, seq_len, d_model, d_ff, post_ln, gate_proj, up_proj, down_proj, cpu_hybrid_qgate_final) ||
        !apply_cpu_ffn(cpu_hybrid_v_hidden_attn, seq_len, d_model, d_ff, post_ln, gate_proj, up_proj, down_proj, cpu_hybrid_v_final) ||
        !apply_cpu_ffn(cpu_hybrid_context_hidden_attn, seq_len, d_model, d_ff, post_ln, gate_proj, up_proj, down_proj, cpu_hybrid_context_final) ||
        !apply_cpu_ffn(cpu_hybrid_sigmoid_hidden_attn, seq_len, d_model, d_ff, post_ln, gate_proj, up_proj, down_proj, cpu_hybrid_sigmoid_final) ||
        !apply_cpu_ffn(cpu_hybrid_gated_context_hidden_attn, seq_len, d_model, d_ff, post_ln, gate_proj, up_proj, down_proj, cpu_hybrid_gated_context_final)) {
        goto fail;
    }

    out->normed_mean_abs_diff = mean_abs_diff(cpu_normed, hybrid_normed, total);
    out->normed_max_abs_diff = max_abs_diff(cpu_normed, hybrid_normed, total);
    out->q_proj_mean_abs_diff = mean_abs_diff(cpu_q, hybrid_q, total_q);
    out->q_proj_max_abs_diff = max_abs_diff(cpu_q, hybrid_q, total_q);
    out->q_gate_mean_abs_diff = mean_abs_diff(cpu_q + q_dim, hybrid_q + q_dim, seq_len * q_dim);
    out->q_gate_max_abs_diff = max_abs_diff(cpu_q + q_dim, hybrid_q + q_dim, seq_len * q_dim);
    out->k_proj_mean_abs_diff = mean_abs_diff(cpu_k, hybrid_k, total_kv);
    out->k_proj_max_abs_diff = max_abs_diff(cpu_k, hybrid_k, total_kv);
    out->v_proj_mean_abs_diff = mean_abs_diff(cpu_v, hybrid_v, total_kv);
    out->v_proj_max_abs_diff = max_abs_diff(cpu_v, hybrid_v, total_kv);
    out->attn_gate_sigmoid_mean_abs_diff = mean_abs_diff(cpu_gate_sigmoid, hybrid_gate_sigmoid, total_context);
    out->attn_gate_sigmoid_max_abs_diff = max_abs_diff(cpu_gate_sigmoid, hybrid_gate_sigmoid, total_context);
    out->attn_context_mean_abs_diff = mean_abs_diff(cpu_context, hybrid_context, total_context);
    out->attn_context_max_abs_diff = max_abs_diff(cpu_context, hybrid_context, total_context);
    out->attn_gated_context_mean_abs_diff = mean_abs_diff(cpu_gated_context, hybrid_gated_context, total_context);
    out->attn_gated_context_max_abs_diff = max_abs_diff(cpu_gated_context, hybrid_gated_context, total_context);
    out->attn_out_mean_abs_diff = mean_abs_diff(cpu_attn, hybrid_attn, total);
    out->attn_out_max_abs_diff = max_abs_diff(cpu_attn, hybrid_attn, total);
    out->attn_qgate_only_mean_abs_diff = mean_abs_diff(cpu_attn, cpu_hybrid_qgate_attn, total);
    out->attn_qgate_only_max_abs_diff = max_abs_diff(cpu_attn, cpu_hybrid_qgate_attn, total);
    out->attn_v_only_mean_abs_diff = mean_abs_diff(cpu_attn, cpu_hybrid_v_attn, total);
    out->attn_v_only_max_abs_diff = max_abs_diff(cpu_attn, cpu_hybrid_v_attn, total);
    out->attn_context_only_mean_abs_diff = mean_abs_diff(cpu_attn, cpu_hybrid_context_attn, total);
    out->attn_context_only_max_abs_diff = max_abs_diff(cpu_attn, cpu_hybrid_context_attn, total);
    out->attn_sigmoid_only_mean_abs_diff = mean_abs_diff(cpu_attn, cpu_hybrid_sigmoid_attn, total);
    out->attn_sigmoid_only_max_abs_diff = max_abs_diff(cpu_attn, cpu_hybrid_sigmoid_attn, total);
    out->attn_gated_context_only_mean_abs_diff = mean_abs_diff(cpu_attn, cpu_hybrid_gated_context_attn, total);
    out->attn_gated_context_only_max_abs_diff = max_abs_diff(cpu_attn, cpu_hybrid_gated_context_attn, total);
    out->hidden_attn_mean_abs_diff = mean_abs_diff(cpu_hidden_attn, hybrid_hidden_attn, total);
    out->hidden_attn_max_abs_diff = max_abs_diff(cpu_hidden_attn, hybrid_hidden_attn, total);
    out->ffn_rms_mean_abs_diff = mean_abs_diff(cpu_ffn_rms, hybrid_ffn_rms, total);
    out->ffn_rms_max_abs_diff = max_abs_diff(cpu_ffn_rms, hybrid_ffn_rms, total);
    out->ffn_gate_mean_abs_diff = mean_abs_diff(cpu_ffn_gate, hybrid_ffn_gate, total_ff);
    out->ffn_gate_max_abs_diff = max_abs_diff(cpu_ffn_gate, hybrid_ffn_gate, total_ff);
    out->ffn_up_mean_abs_diff = mean_abs_diff(cpu_ffn_up, hybrid_ffn_up, total_ff);
    out->ffn_up_max_abs_diff = max_abs_diff(cpu_ffn_up, hybrid_ffn_up, total_ff);
    out->ffn_silu_mean_abs_diff = mean_abs_diff(cpu_ffn_silu, hybrid_ffn_silu, total_ff);
    out->ffn_silu_max_abs_diff = max_abs_diff(cpu_ffn_silu, hybrid_ffn_silu, total_ff);
    out->ffn_hidden_mean_abs_diff = mean_abs_diff(cpu_ffn_hidden, hybrid_ffn_hidden, total_ff);
    out->ffn_hidden_max_abs_diff = max_abs_diff(cpu_ffn_hidden, hybrid_ffn_hidden, total_ff);
    out->ffn_down_mean_abs_diff = mean_abs_diff(cpu_ffn_down, hybrid_ffn_down, total);
    out->ffn_down_max_abs_diff = max_abs_diff(cpu_ffn_down, hybrid_ffn_down, total);
    out->final_mean_abs_diff = mean_abs_diff(cpu_final, hybrid_final, total);
    out->final_max_abs_diff = max_abs_diff(cpu_final, hybrid_final, total);

    for (int i = 0; i < total; i++) {
        cpu_res_hybrid_down_final[i] = cpu_hidden_attn[i] + hybrid_ffn_down[i];
        hybrid_res_cpu_down_final[i] = hybrid_hidden_attn[i] + cpu_ffn_down[i];
    }
    for (int i = 0; i < total_ff; i++) {
        cpu_silu_hybrid_up_hidden[i] = cpu_ffn_silu[i] * hybrid_ffn_up[i];
        hybrid_silu_cpu_up_hidden[i] = hybrid_ffn_silu[i] * cpu_ffn_up[i];
    }
    cpu_linear_batch(cpu_silu_hybrid_up_hidden, seq_len, down_proj, d_ff, d_model, cpu_res_cpu_silu_hybrid_up_down);
    cpu_linear_batch(hybrid_silu_cpu_up_hidden, seq_len, down_proj, d_ff, d_model, cpu_res_hybrid_silu_cpu_up_down);
    for (int i = 0; i < total; i++) {
        cpu_res_cpu_silu_hybrid_up_final[i] = cpu_hidden_attn[i] + cpu_res_cpu_silu_hybrid_up_down[i];
        cpu_res_hybrid_silu_cpu_up_final[i] = cpu_hidden_attn[i] + cpu_res_hybrid_silu_cpu_up_down[i];
    }

    float cpu_hidden_logits[2] = {0.0f, 0.0f};
    float hybrid_hidden_logits[2] = {0.0f, 0.0f};
    float cpu_final_logits[2] = {0.0f, 0.0f};
    float hybrid_final_logits[2] = {0.0f, 0.0f};
    float cpu_res_hybrid_down_logits[2] = {0.0f, 0.0f};
    float hybrid_res_cpu_down_logits[2] = {0.0f, 0.0f};
    float cpu_res_cpu_silu_hybrid_up_logits[2] = {0.0f, 0.0f};
    float cpu_res_hybrid_silu_cpu_up_logits[2] = {0.0f, 0.0f};
    orion_qwen_cpu_rmsnorm(cpu_hidden_attn + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_last);
    orion_qwen_cpu_rmsnorm(hybrid_hidden_attn + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, hybrid_last);
    if (!selected_token_logits(blob_dir, lm_head_name, cpu_last, d_model, pair_ids, 2, cpu_hidden_logits) ||
        !selected_token_logits(blob_dir, lm_head_name, hybrid_last, d_model, pair_ids, 2, hybrid_hidden_logits)) {
        goto fail;
    }
    orion_qwen_cpu_rmsnorm(cpu_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_last);
    orion_qwen_cpu_rmsnorm(hybrid_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, hybrid_last);
    if (!selected_token_logits(blob_dir, lm_head_name, cpu_last, d_model, pair_ids, 2, cpu_final_logits) ||
        !selected_token_logits(blob_dir, lm_head_name, hybrid_last, d_model, pair_ids, 2, hybrid_final_logits)) {
        goto fail;
    }
    orion_qwen_cpu_rmsnorm(cpu_res_hybrid_down_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_last);
    orion_qwen_cpu_rmsnorm(hybrid_res_cpu_down_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, hybrid_last);
    if (!selected_token_logits(blob_dir, lm_head_name, cpu_last, d_model, pair_ids, 2, cpu_res_hybrid_down_logits) ||
        !selected_token_logits(blob_dir, lm_head_name, hybrid_last, d_model, pair_ids, 2, hybrid_res_cpu_down_logits)) {
        goto fail;
    }
    orion_qwen_cpu_rmsnorm(cpu_res_cpu_silu_hybrid_up_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_last);
    orion_qwen_cpu_rmsnorm(cpu_res_hybrid_silu_cpu_up_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, hybrid_last);
    if (!selected_token_logits(blob_dir, lm_head_name, cpu_last, d_model, pair_ids, 2, cpu_res_cpu_silu_hybrid_up_logits) ||
        !selected_token_logits(blob_dir, lm_head_name, hybrid_last, d_model, pair_ids, 2, cpu_res_hybrid_silu_cpu_up_logits)) {
        goto fail;
    }

    out->cpu_hidden_attn_pair_gap = (double)cpu_hidden_logits[0] - (double)cpu_hidden_logits[1];
    out->hybrid_hidden_attn_pair_gap = (double)hybrid_hidden_logits[0] - (double)hybrid_hidden_logits[1];
    out->cpu_final_pair_gap = (double)cpu_final_logits[0] - (double)cpu_final_logits[1];
    out->hybrid_final_pair_gap = (double)hybrid_final_logits[0] - (double)hybrid_final_logits[1];
    out->cpu_res_hybrid_down_final_pair_gap = (double)cpu_res_hybrid_down_logits[0] - (double)cpu_res_hybrid_down_logits[1];
    out->hybrid_res_cpu_down_final_pair_gap = (double)hybrid_res_cpu_down_logits[0] - (double)hybrid_res_cpu_down_logits[1];
    out->cpu_res_cpu_silu_hybrid_up_final_pair_gap = (double)cpu_res_cpu_silu_hybrid_up_logits[0] - (double)cpu_res_cpu_silu_hybrid_up_logits[1];
    out->cpu_res_hybrid_silu_cpu_up_final_pair_gap = (double)cpu_res_hybrid_silu_cpu_up_logits[0] - (double)cpu_res_hybrid_silu_cpu_up_logits[1];
    out->cpu_hidden_attn_pref_token = (out->cpu_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->hybrid_hidden_attn_pref_token = (out->hybrid_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_final_pref_token = (out->cpu_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->hybrid_final_pref_token = (out->hybrid_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_down_final_pref_token = (out->cpu_res_hybrid_down_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->hybrid_res_cpu_down_final_pref_token = (out->hybrid_res_cpu_down_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_cpu_silu_hybrid_up_final_pref_token = (out->cpu_res_cpu_silu_hybrid_up_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_silu_cpu_up_final_pref_token = (out->cpu_res_hybrid_silu_cpu_up_final_pair_gap >= 0.0) ? candidate_a : candidate_b;

    const float *cpu_final_last = cpu_final + (seq_len - 1) * d_model;
    const float *cpu_ffn_rms_last = cpu_ffn_rms + (seq_len - 1) * d_model;
    const float *hybrid_ffn_rms_last = hybrid_ffn_rms + (seq_len - 1) * d_model;
    const float *cpu_ffn_gate_last = cpu_ffn_gate + (seq_len - 1) * d_ff;
    const float *hybrid_ffn_gate_last = hybrid_ffn_gate + (seq_len - 1) * d_ff;
    const float *cpu_ffn_silu_last = cpu_ffn_silu + (seq_len - 1) * d_ff;
    const float *hybrid_ffn_silu_last = hybrid_ffn_silu + (seq_len - 1) * d_ff;
    const float *cpu_ffn_up_last = cpu_ffn_up + (seq_len - 1) * d_ff;
    if (!load_pair_delta_row(blob_dir, lm_head_name, d_model, candidate_a, candidate_b, pair_delta)) {
        goto fail;
    }

    pair_gap_from_hidden_last_pair_delta(final_norm,
                                         cpu_hybrid_qgate_hidden_attn + (seq_len - 1) * d_model,
                                         pair_delta, d_model, gate_attr_norm_last,
                                         &out->cpu_res_hybrid_qgate_hidden_attn_pair_gap);
    pair_gap_from_hidden_last_pair_delta(final_norm,
                                         cpu_hybrid_v_hidden_attn + (seq_len - 1) * d_model,
                                         pair_delta, d_model, gate_attr_norm_last,
                                         &out->cpu_res_hybrid_v_hidden_attn_pair_gap);
    pair_gap_from_hidden_last_pair_delta(final_norm,
                                         cpu_hybrid_context_hidden_attn + (seq_len - 1) * d_model,
                                         pair_delta, d_model, gate_attr_norm_last,
                                         &out->cpu_res_hybrid_context_hidden_attn_pair_gap);
    pair_gap_from_hidden_last_pair_delta(final_norm,
                                         cpu_hybrid_sigmoid_hidden_attn + (seq_len - 1) * d_model,
                                         pair_delta, d_model, gate_attr_norm_last,
                                         &out->cpu_res_hybrid_sigmoid_hidden_attn_pair_gap);
    pair_gap_from_hidden_last_pair_delta(final_norm,
                                         cpu_hybrid_gated_context_hidden_attn + (seq_len - 1) * d_model,
                                         pair_delta, d_model, gate_attr_norm_last,
                                         &out->cpu_res_hybrid_gated_context_hidden_attn_pair_gap);
    pair_gap_from_hidden_last_pair_delta(final_norm,
                                         cpu_hybrid_qgate_final + (seq_len - 1) * d_model,
                                         pair_delta, d_model, gate_attr_norm_last,
                                         &out->cpu_res_hybrid_qgate_final_pair_gap);
    pair_gap_from_hidden_last_pair_delta(final_norm,
                                         cpu_hybrid_v_final + (seq_len - 1) * d_model,
                                         pair_delta, d_model, gate_attr_norm_last,
                                         &out->cpu_res_hybrid_v_final_pair_gap);
    pair_gap_from_hidden_last_pair_delta(final_norm,
                                         cpu_hybrid_context_final + (seq_len - 1) * d_model,
                                         pair_delta, d_model, gate_attr_norm_last,
                                         &out->cpu_res_hybrid_context_final_pair_gap);
    pair_gap_from_hidden_last_pair_delta(final_norm,
                                         cpu_hybrid_sigmoid_final + (seq_len - 1) * d_model,
                                         pair_delta, d_model, gate_attr_norm_last,
                                         &out->cpu_res_hybrid_sigmoid_final_pair_gap);
    pair_gap_from_hidden_last_pair_delta(final_norm,
                                         cpu_hybrid_gated_context_final + (seq_len - 1) * d_model,
                                         pair_delta, d_model, gate_attr_norm_last,
                                         &out->cpu_res_hybrid_gated_context_final_pair_gap);
    out->cpu_res_hybrid_qgate_hidden_attn_pref_token =
        (out->cpu_res_hybrid_qgate_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_v_hidden_attn_pref_token =
        (out->cpu_res_hybrid_v_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_context_hidden_attn_pref_token =
        (out->cpu_res_hybrid_context_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_sigmoid_hidden_attn_pref_token =
        (out->cpu_res_hybrid_sigmoid_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_gated_context_hidden_attn_pref_token =
        (out->cpu_res_hybrid_gated_context_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_qgate_final_pref_token =
        (out->cpu_res_hybrid_qgate_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_v_final_pref_token =
        (out->cpu_res_hybrid_v_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_context_final_pref_token =
        (out->cpu_res_hybrid_context_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_sigmoid_final_pref_token =
        (out->cpu_res_hybrid_sigmoid_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_res_hybrid_gated_context_final_pref_token =
        (out->cpu_res_hybrid_gated_context_final_pair_gap >= 0.0) ? candidate_a : candidate_b;

    out->attn_v_attr_count = 0;
    orion_qwen_cpu_rmsnorm(cpu_hidden_attn + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_v_hidden_norm_last);
    orion_qwen_cpu_rmsnorm(cpu_hybrid_v_hidden_attn + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, hybrid_v_hidden_norm_last);
    for (int d = 0; d < d_model; d++) {
        double delta_norm = (double)hybrid_v_hidden_norm_last[d] - (double)cpu_v_hidden_norm_last[d];
        double signed_contrib = delta_norm * (double)pair_delta[d];
        attn_v_attr_impacts[d].dim = d;
        attn_v_attr_impacts[d].cpu_norm = cpu_v_hidden_norm_last[d];
        attn_v_attr_impacts[d].hybrid_norm = hybrid_v_hidden_norm_last[d];
        attn_v_attr_impacts[d].delta_norm = delta_norm;
        attn_v_attr_impacts[d].pair_delta = pair_delta[d];
        attn_v_attr_impacts[d].abs_contrib = fabs(signed_contrib);
        attn_v_attr_impacts[d].signed_contrib = signed_contrib;
    }
    qsort(attn_v_attr_impacts, (size_t)d_model, sizeof(OrionAttnVAttrImpact), compare_attn_v_attr_impact_desc);
    const int attn_v_report_count = (d_model < ORION_ATTN_V_ATTR_REPORT_MAX) ? d_model : ORION_ATTN_V_ATTR_REPORT_MAX;
    for (int rank = 0; rank < attn_v_report_count; rank++) {
        out->attn_v_attr_dims[rank] = attn_v_attr_impacts[rank].dim;
        out->attn_v_attr_cpu_norm[rank] = attn_v_attr_impacts[rank].cpu_norm;
        out->attn_v_attr_hybrid_norm[rank] = attn_v_attr_impacts[rank].hybrid_norm;
        out->attn_v_attr_delta_norm[rank] = attn_v_attr_impacts[rank].delta_norm;
        out->attn_v_attr_pair_delta[rank] = attn_v_attr_impacts[rank].pair_delta;
        out->attn_v_attr_abs_contrib[rank] = attn_v_attr_impacts[rank].abs_contrib;
        out->attn_v_attr_signed_contrib[rank] = attn_v_attr_impacts[rank].signed_contrib;
        out->attn_v_attr_count = rank + 1;
    }

    out->attn_gated_context_attr_count = 0;
    orion_qwen_cpu_rmsnorm(cpu_hidden_attn + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_gated_context_hidden_norm_last);
    orion_qwen_cpu_rmsnorm(cpu_hybrid_gated_context_hidden_attn + (seq_len - 1) * d_model,
                           final_norm, d_model, 1e-6f, hybrid_gated_context_hidden_norm_last);
    for (int d = 0; d < d_model; d++) {
        double delta_norm = (double)hybrid_gated_context_hidden_norm_last[d] - (double)cpu_gated_context_hidden_norm_last[d];
        double signed_contrib = delta_norm * (double)pair_delta[d];
        attn_gated_context_attr_impacts[d].dim = d;
        attn_gated_context_attr_impacts[d].cpu_norm = cpu_gated_context_hidden_norm_last[d];
        attn_gated_context_attr_impacts[d].hybrid_norm = hybrid_gated_context_hidden_norm_last[d];
        attn_gated_context_attr_impacts[d].delta_norm = delta_norm;
        attn_gated_context_attr_impacts[d].pair_delta = pair_delta[d];
        attn_gated_context_attr_impacts[d].abs_contrib = fabs(signed_contrib);
        attn_gated_context_attr_impacts[d].signed_contrib = signed_contrib;
    }
    qsort(attn_gated_context_attr_impacts, (size_t)d_model, sizeof(OrionAttnVAttrImpact), compare_attn_v_attr_impact_desc);
    const int attn_gated_context_report_count =
        (d_model < ORION_ATTN_V_ATTR_REPORT_MAX) ? d_model : ORION_ATTN_V_ATTR_REPORT_MAX;
    for (int rank = 0; rank < attn_gated_context_report_count; rank++) {
        out->attn_gated_context_attr_dims[rank] = attn_gated_context_attr_impacts[rank].dim;
        out->attn_gated_context_attr_cpu_norm[rank] = attn_gated_context_attr_impacts[rank].cpu_norm;
        out->attn_gated_context_attr_hybrid_norm[rank] = attn_gated_context_attr_impacts[rank].hybrid_norm;
        out->attn_gated_context_attr_delta_norm[rank] = attn_gated_context_attr_impacts[rank].delta_norm;
        out->attn_gated_context_attr_pair_delta[rank] = attn_gated_context_attr_impacts[rank].pair_delta;
        out->attn_gated_context_attr_abs_contrib[rank] = attn_gated_context_attr_impacts[rank].abs_contrib;
        out->attn_gated_context_attr_signed_contrib[rank] = attn_gated_context_attr_impacts[rank].signed_contrib;
        out->attn_gated_context_attr_count = rank + 1;
    }

    out->ffn_down_attr_count = 0;
    orion_qwen_cpu_rmsnorm(cpu_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_ffn_down_norm_last);
    orion_qwen_cpu_rmsnorm(cpu_res_hybrid_down_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, hybrid_ffn_down_norm_last);
    for (int d = 0; d < d_model; d++) {
        double delta_norm = (double)hybrid_ffn_down_norm_last[d] - (double)cpu_ffn_down_norm_last[d];
        double signed_contrib = delta_norm * (double)pair_delta[d];
        ffn_down_attr_impacts[d].dim = d;
        ffn_down_attr_impacts[d].cpu_norm = cpu_ffn_down_norm_last[d];
        ffn_down_attr_impacts[d].hybrid_norm = hybrid_ffn_down_norm_last[d];
        ffn_down_attr_impacts[d].delta_norm = delta_norm;
        ffn_down_attr_impacts[d].pair_delta = pair_delta[d];
        ffn_down_attr_impacts[d].abs_contrib = fabs(signed_contrib);
        ffn_down_attr_impacts[d].signed_contrib = signed_contrib;
    }
    qsort(ffn_down_attr_impacts, (size_t)d_model, sizeof(OrionFfnDownAttrImpact), compare_ffn_down_attr_impact_desc);
    const int ffn_down_report_count = (d_model < ORION_FFN_DOWN_ATTR_REPORT_MAX) ? d_model : ORION_FFN_DOWN_ATTR_REPORT_MAX;
    for (int rank = 0; rank < ffn_down_report_count; rank++) {
        out->ffn_down_attr_dims[rank] = ffn_down_attr_impacts[rank].dim;
        out->ffn_down_attr_cpu_norm[rank] = ffn_down_attr_impacts[rank].cpu_norm;
        out->ffn_down_attr_hybrid_norm[rank] = ffn_down_attr_impacts[rank].hybrid_norm;
        out->ffn_down_attr_delta_norm[rank] = ffn_down_attr_impacts[rank].delta_norm;
        out->ffn_down_attr_pair_delta[rank] = ffn_down_attr_impacts[rank].pair_delta;
        out->ffn_down_attr_abs_contrib[rank] = ffn_down_attr_impacts[rank].abs_contrib;
        out->ffn_down_attr_signed_contrib[rank] = ffn_down_attr_impacts[rank].signed_contrib;
        out->ffn_down_attr_count = rank + 1;
    }

    if (gate_attr_report_topk < 0) gate_attr_report_topk = 0;
    if (gate_attr_report_topk > ORION_GATE_ATTR_REPORT_MAX) gate_attr_report_topk = ORION_GATE_ATTR_REPORT_MAX;

    for (int i = 0; i < d_ff; i++) {
        double delta_hidden = ((double)hybrid_ffn_silu_last[i] - (double)cpu_ffn_silu_last[i]) *
                              (double)cpu_ffn_up_last[i];
        for (int o = 0; o < d_model; o++) {
            gate_attr_single_last[o] = cpu_final_last[o] + (float)(delta_hidden * (double)down_proj[o * d_ff + i]);
        }
        pair_gap_from_hidden_last_pair_delta(final_norm, gate_attr_single_last, pair_delta, d_model,
                                             gate_attr_norm_last, &gate_attr_impacts[i].single_pair_gap);
        gate_attr_impacts[i].channel = i;
        gate_attr_impacts[i].delta_hidden = delta_hidden;
    }

    qsort(gate_attr_impacts, (size_t)d_ff, sizeof(OrionGateAttrImpact), compare_gate_attr_impact_asc);

    memcpy(gate_attr_cumulative_last, cpu_final_last, (size_t)d_model * sizeof(float));
    out->gate_attr_total_channels = d_ff;
    out->gate_attr_report_count = 0;
    out->gate_attr_first_cumulative_flip_rank = -1;
    out->gate_attr_flip_channel = -1;
    out->gate_attr_flip_delta_hidden = 0.0;
    out->gate_attr_flip_cumulative_pair_gap = INFINITY;
    out->gate_attr_all_cumulative_pair_gap = out->cpu_final_pair_gap;
    out->gate_input_attr_channel_limit = 0;
    out->gate_input_attr_count = 0;
    for (int rank = 0; rank < d_ff; rank++) {
        const OrionGateAttrImpact impact = gate_attr_impacts[rank];
        for (int o = 0; o < d_model; o++) {
            gate_attr_cumulative_last[o] += (float)(impact.delta_hidden * (double)down_proj[o * d_ff + impact.channel]);
        }

        double cumulative_pair_gap = 0.0;
        pair_gap_from_hidden_last_pair_delta(final_norm, gate_attr_cumulative_last, pair_delta, d_model,
                                             gate_attr_norm_last, &cumulative_pair_gap);
        out->gate_attr_all_cumulative_pair_gap = cumulative_pair_gap;
        if (rank < gate_attr_report_topk) {
            out->gate_attr_channels[rank] = impact.channel;
            out->gate_attr_cpu_gate[rank] = cpu_ffn_gate_last[impact.channel];
            out->gate_attr_hybrid_gate[rank] = hybrid_ffn_gate_last[impact.channel];
            out->gate_attr_delta_gate[rank] = (double)hybrid_ffn_gate_last[impact.channel] - (double)cpu_ffn_gate_last[impact.channel];
            out->gate_attr_cpu_silu[rank] = cpu_ffn_silu_last[impact.channel];
            out->gate_attr_hybrid_silu[rank] = hybrid_ffn_silu_last[impact.channel];
            out->gate_attr_delta_silu[rank] = (double)hybrid_ffn_silu_last[impact.channel] - (double)cpu_ffn_silu_last[impact.channel];
            out->gate_attr_cpu_up[rank] = cpu_ffn_up_last[impact.channel];
            out->gate_attr_delta_hidden[rank] = impact.delta_hidden;
            out->gate_attr_single_pair_gap[rank] = impact.single_pair_gap;
            out->gate_attr_single_pref_token[rank] = (impact.single_pair_gap >= 0.0) ? candidate_a : candidate_b;
            out->gate_attr_cumulative_pair_gap[rank] = cumulative_pair_gap;
            out->gate_attr_cumulative_pref_token[rank] = (cumulative_pair_gap >= 0.0) ? candidate_a : candidate_b;
            out->gate_attr_report_count = rank + 1;
        }
        if (out->gate_attr_first_cumulative_flip_rank < 0 && cumulative_pair_gap < 0.0) {
            out->gate_attr_first_cumulative_flip_rank = rank + 1;
            out->gate_attr_flip_channel = impact.channel;
            out->gate_attr_flip_delta_hidden = impact.delta_hidden;
            out->gate_attr_flip_cumulative_pair_gap = cumulative_pair_gap;
        }
    }

    int gate_input_channel_limit = out->gate_attr_first_cumulative_flip_rank;
    if (gate_input_channel_limit <= 0) gate_input_channel_limit = gate_attr_report_topk;
    if (gate_input_channel_limit > d_ff) gate_input_channel_limit = d_ff;
    out->gate_input_attr_channel_limit = gate_input_channel_limit;
    for (int d = 0; d < d_model; d++) {
        double delta_ffn_rms = (double)hybrid_ffn_rms_last[d] - (double)cpu_ffn_rms_last[d];
        double abs_contrib = 0.0;
        double signed_contrib = 0.0;
        for (int rank = 0; rank < gate_input_channel_limit; rank++) {
            const int channel = gate_attr_impacts[rank].channel;
            double contrib = delta_ffn_rms * (double)gate_proj[channel * d_model + d];
            abs_contrib += fabs(contrib);
            signed_contrib += contrib;
        }
        gate_input_attr_impacts[d].dim = d;
        gate_input_attr_impacts[d].abs_contrib = abs_contrib;
        gate_input_attr_impacts[d].signed_contrib = signed_contrib;
        gate_input_attr_impacts[d].delta_ffn_rms = delta_ffn_rms;
    }
    qsort(gate_input_attr_impacts, (size_t)d_model, sizeof(OrionGateInputAttrImpact), compare_gate_input_attr_impact_desc);
    const int gate_input_report_count = (d_model < ORION_GATE_INPUT_ATTR_REPORT_MAX) ? d_model : ORION_GATE_INPUT_ATTR_REPORT_MAX;
    for (int rank = 0; rank < gate_input_report_count; rank++) {
        out->gate_input_attr_dims[rank] = gate_input_attr_impacts[rank].dim;
        out->gate_input_attr_abs_contrib[rank] = gate_input_attr_impacts[rank].abs_contrib;
        out->gate_input_attr_signed_contrib[rank] = gate_input_attr_impacts[rank].signed_contrib;
        out->gate_input_attr_delta_ffn_rms[rank] = gate_input_attr_impacts[rank].delta_ffn_rms;
        out->gate_input_attr_count = rank + 1;
    }

    free(input_ln); free(post_ln); free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
    free(gate_proj); free(up_proj); free(down_proj); free(final_norm);
    free(cpu_normed); free(hybrid_normed); free(cpu_q); free(hybrid_q); free(cpu_k); free(hybrid_k); free(cpu_v); free(hybrid_v);
    free(cpu_gate_sigmoid); free(hybrid_gate_sigmoid); free(cpu_context); free(hybrid_context); free(cpu_gated_context); free(hybrid_gated_context);
    free(cpu_attn); free(hybrid_attn); free(cpu_q_hybrid_gate); free(cpu_hybrid_qgate_attn); free(cpu_hybrid_v_attn); free(cpu_hybrid_context_gated_context); free(cpu_hybrid_context_attn); free(cpu_hybrid_sigmoid_gated_context);
    free(cpu_hybrid_sigmoid_attn); free(cpu_hybrid_gated_context_attn);
    free(cpu_hidden_attn); free(hybrid_hidden_attn); free(cpu_hybrid_qgate_hidden_attn); free(cpu_hybrid_v_hidden_attn); free(cpu_hybrid_context_hidden_attn); free(cpu_hybrid_sigmoid_hidden_attn); free(cpu_hybrid_gated_context_hidden_attn); free(cpu_ffn_rms); free(hybrid_ffn_rms);
    free(cpu_ffn_gate); free(hybrid_ffn_gate); free(cpu_ffn_up); free(hybrid_ffn_up); free(cpu_ffn_silu); free(hybrid_ffn_silu);
    free(cpu_ffn_hidden); free(hybrid_ffn_hidden); free(cpu_ffn_down); free(hybrid_ffn_down); free(cpu_final); free(hybrid_final);
    free(cpu_hybrid_qgate_final); free(cpu_hybrid_v_final); free(cpu_hybrid_context_final); free(cpu_hybrid_sigmoid_final); free(cpu_hybrid_gated_context_final);
    free(cpu_res_hybrid_down_final); free(hybrid_res_cpu_down_final);
    free(cpu_silu_hybrid_up_hidden); free(hybrid_silu_cpu_up_hidden);
    free(cpu_res_cpu_silu_hybrid_up_down); free(cpu_res_hybrid_silu_cpu_up_down);
    free(cpu_res_cpu_silu_hybrid_up_final); free(cpu_res_hybrid_silu_cpu_up_final);
    free(gate_attr_single_last); free(gate_attr_cumulative_last); free(gate_attr_norm_last); free(gate_attr_impacts); free(gate_input_attr_impacts); free(attn_v_attr_impacts); free(attn_gated_context_attr_impacts); free(ffn_down_attr_impacts); free(pair_delta);
    free(cpu_last); free(hybrid_last); free(cpu_v_hidden_norm_last); free(hybrid_v_hidden_norm_last); free(cpu_gated_context_hidden_norm_last); free(hybrid_gated_context_hidden_norm_last); free(cpu_ffn_down_norm_last); free(hybrid_ffn_down_norm_last);
    return 1;

fail:
    free(input_ln); free(post_ln); free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
    free(gate_proj); free(up_proj); free(down_proj); free(final_norm);
    free(cpu_normed); free(hybrid_normed); free(cpu_q); free(hybrid_q); free(cpu_k); free(hybrid_k); free(cpu_v); free(hybrid_v);
    free(cpu_gate_sigmoid); free(hybrid_gate_sigmoid); free(cpu_context); free(hybrid_context); free(cpu_gated_context); free(hybrid_gated_context);
    free(cpu_attn); free(hybrid_attn); free(cpu_q_hybrid_gate); free(cpu_hybrid_qgate_attn); free(cpu_hybrid_v_attn); free(cpu_hybrid_context_gated_context); free(cpu_hybrid_context_attn); free(cpu_hybrid_sigmoid_gated_context);
    free(cpu_hybrid_sigmoid_attn); free(cpu_hybrid_gated_context_attn);
    free(cpu_hidden_attn); free(hybrid_hidden_attn); free(cpu_hybrid_qgate_hidden_attn); free(cpu_hybrid_v_hidden_attn); free(cpu_hybrid_context_hidden_attn); free(cpu_hybrid_sigmoid_hidden_attn); free(cpu_hybrid_gated_context_hidden_attn); free(cpu_ffn_rms); free(hybrid_ffn_rms);
    free(cpu_ffn_gate); free(hybrid_ffn_gate); free(cpu_ffn_up); free(hybrid_ffn_up); free(cpu_ffn_silu); free(hybrid_ffn_silu);
    free(cpu_ffn_hidden); free(hybrid_ffn_hidden); free(cpu_ffn_down); free(hybrid_ffn_down); free(cpu_final); free(hybrid_final);
    free(cpu_hybrid_qgate_final); free(cpu_hybrid_v_final); free(cpu_hybrid_context_final); free(cpu_hybrid_sigmoid_final); free(cpu_hybrid_gated_context_final);
    free(cpu_res_hybrid_down_final); free(hybrid_res_cpu_down_final);
    free(cpu_silu_hybrid_up_hidden); free(hybrid_silu_cpu_up_hidden);
    free(cpu_res_cpu_silu_hybrid_up_down); free(cpu_res_hybrid_silu_cpu_up_down);
    free(cpu_res_cpu_silu_hybrid_up_final); free(cpu_res_hybrid_silu_cpu_up_final);
    free(gate_attr_single_last); free(gate_attr_cumulative_last); free(gate_attr_norm_last); free(gate_attr_impacts); free(gate_input_attr_impacts); free(attn_v_attr_impacts); free(attn_gated_context_attr_impacts); free(ffn_down_attr_impacts); free(pair_delta);
    free(cpu_last); free(hybrid_last); free(cpu_v_hidden_norm_last); free(hybrid_v_hidden_norm_last); free(cpu_gated_context_hidden_norm_last); free(hybrid_gated_context_hidden_norm_last); free(cpu_ffn_down_norm_last); free(hybrid_ffn_down_norm_last);
    return 0;
}

static int run_cpu_tail_pair_replay(const char *blob_dir,
                                    OrionQwen35Manifest *manifest,
                                    int start_layer,
                                    const float *hidden_in,
                                    int seq_len,
                                    int sample_vocab,
                                    int candidate_a,
                                    int candidate_b,
                                    OrionTailPairReplay *out) {
    if (!manifest || !hidden_in || !out || start_layer < 0 || start_layer >= manifest->n_layer) return 0;

    const int d_model = manifest->d_model;
    const int total = seq_len * d_model;
    const char *lm_head_name = manifest->tie_word_embeddings ? "embed_tokens.bin" : "lm_head.bin";
    const int pair_ids[2] = {candidate_a, candidate_b};

    float *curr = (float *)calloc((size_t)total, sizeof(float));
    float *next = (float *)calloc((size_t)total, sizeof(float));
    float *final_norm = NULL;
    float *last = (float *)calloc((size_t)d_model, sizeof(float));
    if (!curr || !next || !last) {
        free(curr);
        free(next);
        free(last);
        return 0;
    }

    memcpy(curr, hidden_in, (size_t)total * sizeof(float));
    for (int layer_idx = start_layer; layer_idx < manifest->n_layer; layer_idx++) {
        if (!apply_cpu_layer(blob_dir, manifest, layer_idx, curr, seq_len, next)) {
            free(curr);
            free(next);
            free(last);
            return 0;
        }
        float *tmp = curr;
        curr = next;
        next = tmp;
    }

    char final_norm_path[2048];
    snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
    final_norm = orion_read_blob_f32_exact(final_norm_path, d_model);
    if (!final_norm) {
        free(curr);
        free(next);
        free(last);
        return 0;
    }

    orion_qwen_cpu_rmsnorm(curr + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, last);
    float pair_logits[2] = {0.0f, 0.0f};
    if (!selected_token_logits(blob_dir, lm_head_name, last, d_model, pair_ids, 2, pair_logits) ||
        !sampled_topk_logits(blob_dir, lm_head_name, last, d_model, sample_vocab, &out->top_id, &out->top_logit)) {
        free(curr);
        free(next);
        free(last);
        free(final_norm);
        return 0;
    }

    out->final_pair_gap = (double)pair_logits[0] - (double)pair_logits[1];
    out->final_pref_token = (out->final_pair_gap >= 0.0) ? candidate_a : candidate_b;

    free(curr);
    free(next);
    free(last);
    free(final_norm);
    return 1;
}

static int run_same_input_stage_pair_compare(const char *blob_dir,
                                             OrionQwen35Manifest *manifest,
                                             OrionQwen35AneBridge *bridge,
                                             const float *hidden_in,
                                             int seq_len,
                                             int candidate_a,
                                             int candidate_b,
                                             OrionStagePairSensitivity *out) {
    if (!bridge || !out) return 0;

    const int layer = bridge->layer_idx;
    const int d_model = manifest->d_model;
    const int d_ff = manifest->d_ff;
    const int n_head = manifest->n_head;
    const int n_kv_head = manifest->n_kv_head;
    const int head_dim = manifest->head_dim;
    const int q_dim = bridge->q_dim;
    const int kv_dim = bridge->kv_dim;
    const char *lm_head_name = manifest->tie_word_embeddings ? "embed_tokens.bin" : "lm_head.bin";
    const int pair_ids[2] = {candidate_a, candidate_b};

    float *input_ln = load_exact(blob_dir, layer, "input_layernorm.bin", d_model);
    float *post_ln = load_exact(blob_dir, layer, "post_attention_layernorm.bin", d_model);
    float *gate_proj = load_exact(blob_dir, layer, "mlp_gate_proj.bin", d_ff * d_model);
    float *up_proj = load_exact(blob_dir, layer, "mlp_up_proj.bin", d_ff * d_model);
    float *down_proj = load_exact(blob_dir, layer, "mlp_down_proj.bin", d_model * d_ff);
    float *o_proj = load_exact(blob_dir, layer, "self_attn_o_proj.bin", d_model * q_dim);
    float *q_norm = load_exact(blob_dir, layer, "self_attn_q_norm.bin", head_dim);
    float *k_norm = load_exact(blob_dir, layer, "self_attn_k_norm.bin", head_dim);
    float *final_norm = NULL;
    float *normed = NULL;
    float *cpu_q = NULL;
    float *cpu_k = NULL;
    float *cpu_v = NULL;
    float *ane_q = NULL;
    float *ane_k = NULL;
    float *ane_v = NULL;
    float *cpu_attn = NULL;
    float *ane_attn = NULL;
    float *cpu_hidden_attn = NULL;
    float *ane_hidden_attn = NULL;
    float *cpu_final = NULL;
    float *ane_final = NULL;
    float *cpu_last = NULL;
    float *ane_last = NULL;
    float *q_proj_w = NULL;
    float *k_proj_w = NULL;
    float *v_proj_w = NULL;
    IOSurfaceRef ioInQ = NULL;
    IOSurfaceRef ioInKV = NULL;
    IOSurfaceRef ioQ = NULL;
    IOSurfaceRef ioK = NULL;
    IOSurfaceRef ioV = NULL;
    IOSurfaceRef ioFfnIn = NULL;
    IOSurfaceRef ioHidden = NULL;

    char final_norm_path[2048];
    snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
    final_norm = orion_read_blob_f32_exact(final_norm_path, d_model);

    normed = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    cpu_q = (float *)calloc((size_t)seq_len * (q_dim * 2), sizeof(float));
    cpu_k = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    cpu_v = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    ane_q = (float *)calloc((size_t)seq_len * (q_dim * 2), sizeof(float));
    ane_k = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    ane_v = (float *)calloc((size_t)seq_len * kv_dim, sizeof(float));
    cpu_attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    ane_attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    cpu_hidden_attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    ane_hidden_attn = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    cpu_final = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    ane_final = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    cpu_last = (float *)calloc((size_t)d_model, sizeof(float));
    ane_last = (float *)calloc((size_t)d_model, sizeof(float));
    if (!input_ln || !post_ln || !gate_proj || !up_proj || !down_proj || !o_proj || !q_norm || !k_norm ||
        !final_norm || !normed || !cpu_q || !cpu_k || !cpu_v || !ane_q || !ane_k || !ane_v ||
        !cpu_attn || !ane_attn || !cpu_hidden_attn || !ane_hidden_attn || !cpu_final || !ane_final ||
        !cpu_last || !ane_last) {
        goto fail;
    }

    for (int s = 0; s < seq_len; s++) {
        orion_qwen_cpu_rmsnorm(hidden_in + s * d_model, input_ln, d_model, 1e-6f, normed + s * d_model);
    }
    q_proj_w = load_exact(blob_dir, layer, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
    k_proj_w = load_exact(blob_dir, layer, "self_attn_k_proj.bin", kv_dim * d_model);
    v_proj_w = load_exact(blob_dir, layer, "self_attn_v_proj.bin", kv_dim * d_model);
    if (!q_proj_w || !k_proj_w || !v_proj_w) {
        goto fail;
    }
    cpu_linear_batch(normed, seq_len, q_proj_w, d_model, q_dim * 2, cpu_q);
    cpu_linear_batch(normed, seq_len, k_proj_w, d_model, kv_dim, cpu_k);
    cpu_linear_batch(normed, seq_len, v_proj_w, d_model, kv_dim, cpu_v);

    {
        const float *q_input = bridge->q_uses_cpu_rms ? normed : hidden_in;
        const float *kv_input = bridge->kv_uses_cpu_rms ? normed : hidden_in;
        ioInQ = make_cpu_seq_input_surface(q_input, seq_len, bridge->bucket, d_model);
        ioInKV = make_cpu_seq_input_surface(kv_input, seq_len, bridge->bucket, d_model);
    }
    ioQ = make_f32_surface((q_dim * 2) * bridge->bucket, 0.0f);
    ioK = make_f32_surface(kv_dim * bridge->bucket, 0.0f);
    ioV = make_f32_surface(kv_dim * bridge->bucket, 0.0f);
    IOSurfaceRef insQ[] = {ioInQ};
    IOSurfaceRef outsQ[] = {ioQ};
    IOSurfaceRef insKV[] = {ioInKV};
    IOSurfaceRef outsKV[] = {ioK, ioV};
    if (!orion_eval(bridge->prog_q, insQ, 1, outsQ, 1) || !orion_eval(bridge->prog_kv, insKV, 1, outsKV, 2)) {
        goto fail;
    }
    read_ane_surface_prefix(ioQ, q_dim * 2, seq_len, bridge->bucket, ane_q);
    read_ane_surface_prefix(ioK, kv_dim, seq_len, bridge->bucket, ane_k);
    read_ane_surface_prefix(ioV, kv_dim, seq_len, bridge->bucket, ane_v);
    if (bridge->q_proj_uses_cpu || bridge->q_query_uses_cpu || bridge->q_gate_uses_cpu || bridge->q_gate_cpu_channel_count > 0) {
        float *cpu_q_proj_seq = (float *)calloc((size_t)seq_len * (q_dim * 2), sizeof(float));
        if (!cpu_q_proj_seq) goto fail;
        cpu_linear_batch(normed, seq_len, q_proj_w, d_model, q_dim * 2, cpu_q_proj_seq);
        for (int s = 0; s < seq_len; s++) {
            float *dst = ane_q + (size_t)s * (q_dim * 2);
            const float *src = cpu_q_proj_seq + (size_t)s * (q_dim * 2);
            if (bridge->q_proj_uses_cpu || bridge->q_query_uses_cpu) {
                memcpy(dst, src, (size_t)q_dim * sizeof(float));
            }
            if (bridge->q_proj_uses_cpu || bridge->q_gate_uses_cpu) {
                memcpy(dst + q_dim, src + q_dim, (size_t)q_dim * sizeof(float));
            }
        }
        if (!bridge->q_proj_uses_cpu && !bridge->q_gate_uses_cpu && bridge->q_gate_cpu_channel_count > 0) {
            apply_q_gate_cpu_channel_overrides(ane_q, cpu_q_proj_seq, seq_len, q_dim,
                                               bridge->q_gate_cpu_channels, bridge->q_gate_cpu_channel_count);
        }
        free(cpu_q_proj_seq);
    }
    if (bridge->v_proj_uses_cpu) {
        memcpy(ane_v, cpu_v, (size_t)seq_len * kv_dim * sizeof(float));
    } else if (bridge->v_proj_cpu_channel_count > 0) {
        apply_v_proj_cpu_channel_overrides(ane_v, cpu_v, seq_len, kv_dim, head_dim, n_head / n_kv_head,
                                           bridge->v_proj_cpu_channels, bridge->v_proj_cpu_channel_count);
    }

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

    for (int i = 0; i < seq_len * d_model; i++) {
        cpu_hidden_attn[i] = hidden_in[i] + cpu_attn[i];
        ane_hidden_attn[i] = hidden_in[i] + ane_attn[i];
    }

    if (!apply_cpu_ffn(cpu_hidden_attn, seq_len, d_model, d_ff, post_ln, gate_proj, up_proj, down_proj, cpu_final)) {
        goto fail;
    }
    ioFfnIn = make_cpu_seq_input_surface(ane_hidden_attn, seq_len, bridge->bucket, d_model);
    ioHidden = make_f32_surface(d_model * bridge->bucket, 0.0f);
    IOSurfaceRef insFFN[] = {ioFfnIn};
    IOSurfaceRef outsFFN[] = {ioHidden};
    if (!orion_eval(bridge->prog_ffn, insFFN, 1, outsFFN, 1)) {
        goto fail;
    }
    read_ane_surface_prefix(ioHidden, d_model, seq_len, bridge->bucket, ane_final);

    float cpu_hidden_attn_pair_logits[2] = {0.0f, 0.0f};
    float ane_hidden_attn_pair_logits[2] = {0.0f, 0.0f};
    float cpu_final_pair_logits[2] = {0.0f, 0.0f};
    float ane_final_pair_logits[2] = {0.0f, 0.0f};
    orion_qwen_cpu_rmsnorm(cpu_hidden_attn + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_last);
    orion_qwen_cpu_rmsnorm(ane_hidden_attn + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, ane_last);
    if (!selected_token_logits(blob_dir, lm_head_name, cpu_last, d_model, pair_ids, 2, cpu_hidden_attn_pair_logits) ||
        !selected_token_logits(blob_dir, lm_head_name, ane_last, d_model, pair_ids, 2, ane_hidden_attn_pair_logits)) {
        goto fail;
    }
    orion_qwen_cpu_rmsnorm(cpu_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_last);
    orion_qwen_cpu_rmsnorm(ane_final + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, ane_last);
    if (!selected_token_logits(blob_dir, lm_head_name, cpu_last, d_model, pair_ids, 2, cpu_final_pair_logits) ||
        !selected_token_logits(blob_dir, lm_head_name, ane_last, d_model, pair_ids, 2, ane_final_pair_logits)) {
        goto fail;
    }

    out->cpu_hidden_attn_pair_gap = (double)cpu_hidden_attn_pair_logits[0] - (double)cpu_hidden_attn_pair_logits[1];
    out->ane_hidden_attn_pair_gap = (double)ane_hidden_attn_pair_logits[0] - (double)ane_hidden_attn_pair_logits[1];
    out->cpu_final_pair_gap = (double)cpu_final_pair_logits[0] - (double)cpu_final_pair_logits[1];
    out->ane_final_pair_gap = (double)ane_final_pair_logits[0] - (double)ane_final_pair_logits[1];
    out->cpu_hidden_attn_pref_token = (out->cpu_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->ane_hidden_attn_pref_token = (out->ane_hidden_attn_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->cpu_final_pref_token = (out->cpu_final_pair_gap >= 0.0) ? candidate_a : candidate_b;
    out->ane_final_pref_token = (out->ane_final_pair_gap >= 0.0) ? candidate_a : candidate_b;

    if (ioInQ) CFRelease(ioInQ);
    if (ioInKV) CFRelease(ioInKV);
    if (ioQ) CFRelease(ioQ);
    if (ioK) CFRelease(ioK);
    if (ioV) CFRelease(ioV);
    if (ioFfnIn) CFRelease(ioFfnIn);
    if (ioHidden) CFRelease(ioHidden);
    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj); free(o_proj); free(q_norm); free(k_norm);
    free(final_norm); free(normed); free(cpu_q); free(cpu_k); free(cpu_v); free(ane_q); free(ane_k); free(ane_v);
    free(cpu_attn); free(ane_attn); free(cpu_hidden_attn); free(ane_hidden_attn); free(cpu_final); free(ane_final);
    free(cpu_last); free(ane_last);
    free(q_proj_w); free(k_proj_w); free(v_proj_w);
    return 1;

fail:
    if (ioInQ) CFRelease(ioInQ);
    if (ioInKV) CFRelease(ioInKV);
    if (ioQ) CFRelease(ioQ);
    if (ioK) CFRelease(ioK);
    if (ioV) CFRelease(ioV);
    if (ioFfnIn) CFRelease(ioFfnIn);
    if (ioHidden) CFRelease(ioHidden);
    free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj); free(o_proj); free(q_norm); free(k_norm);
    free(final_norm); free(normed); free(cpu_q); free(cpu_k); free(cpu_v); free(ane_q); free(ane_k); free(ane_v);
    free(cpu_attn); free(ane_attn); free(cpu_hidden_attn); free(ane_hidden_attn); free(cpu_final); free(ane_final);
    free(cpu_last); free(ane_last);
    free(q_proj_w); free(k_proj_w); free(v_proj_w);
    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s <blob_dir> <tokenizer_dir> [single|all_full] [prompt] [sample_vocab] [candidate_a] [candidate_b] [sensitivity_layer]\n", argv[0]);
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
        const char *prompt = (argc >= 5) ? argv[4] : "사진";
        const int sample_vocab = (argc >= 6) ? atoi(argv[5]) : 4096;
        const int candidate_a = (argc >= 7) ? atoi(argv[6]) : -1;
        const int candidate_b = (argc >= 8) ? atoi(argv[7]) : -1;
        const int pair_enabled = (candidate_a >= 0 && candidate_b >= 0);
        const int sensitivity_layer = (argc >= 9) ? atoi(argv[8]) : -1;
        int gate_attr_report_topk = (argc >= 10) ? atoi(argv[9]) : 32;
        const char *trace_dims_csv = (argc >= 11) ? argv[10] : "";
        int trace_dims[ORION_TRACE_DIM_MAX] = {0};
        int trace_dim_count = parse_dim_list(trace_dims_csv, trace_dims, ORION_TRACE_DIM_MAX, manifest->d_model);
        if (gate_attr_report_topk < 0) gate_attr_report_topk = 0;
        if (gate_attr_report_topk > ORION_GATE_ATTR_REPORT_MAX) gate_attr_report_topk = ORION_GATE_ATTR_REPORT_MAX;
        if (strcmp(mode, "single") != 0 && strcmp(mode, "all_full") != 0) {
            fprintf(stderr, "FAIL: unsupported mode '%s'\n", mode);
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
        const int bucket = 32;
        const int total = seq_len * d_model;
        NSString *blobDir = [NSString stringWithUTF8String:blob_dir];

        OrionQwen35AneBridge *bridges = calloc((size_t)manifest->n_layer, sizeof(OrionQwen35AneBridge));
        unsigned char *bridge_mask = calloc((size_t)manifest->n_layer, sizeof(unsigned char));
        OrionQwen35AneBridge sensitivity_bridge = {0};
        OrionQwen35AneBridge *sensitivity_bridge_ptr = NULL;
        int sensitivity_bridge_owned = 0;
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
                if (!bridge_init(&bridges[layer], blobDir, layer, bucket, manifest, qkv_input_mode, q_proj_uses_cpu, k_proj_uses_cpu, v_proj_uses_cpu, q_query_uses_cpu, q_gate_uses_cpu, ffn_uses_cpu)) {
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
            if (!bridge_init(&bridges[bridged_layer], blobDir, bridged_layer, bucket, manifest, qkv_input_mode, q_proj_uses_cpu, k_proj_uses_cpu, v_proj_uses_cpu, q_query_uses_cpu, q_gate_uses_cpu, ffn_uses_cpu)) {
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

        if (pair_enabled && sensitivity_layer >= 0) {
            if (sensitivity_layer >= manifest->n_layer) {
                fprintf(stderr, "FAIL: sensitivity layer %d out of range\n", sensitivity_layer);
                for (int i = 0; i < manifest->n_layer; i++) bridge_release(&bridges[i]);
                free(bridges);
                free(bridge_mask);
                orion_qwen35_manifest_free(manifest);
                orion_gpt2_tokenizer_free(tok);
                return 1;
            }

            char sensitivity_q_path[2048];
            snprintf(sensitivity_q_path, sizeof(sensitivity_q_path), "%s/layer%d/self_attn_q_proj.bin", blob_dir, sensitivity_layer);
            if (file_exists(sensitivity_q_path)) {
                if (bridge_mask[sensitivity_layer]) {
                    sensitivity_bridge_ptr = &bridges[sensitivity_layer];
                } else {
                    if (!bridge_init(&sensitivity_bridge, blobDir, sensitivity_layer, bucket, manifest, qkv_input_mode, q_proj_uses_cpu, k_proj_uses_cpu, v_proj_uses_cpu, q_query_uses_cpu, q_gate_uses_cpu, ffn_uses_cpu)) {
                        fprintf(stderr, "FAIL: sensitivity bridge_init failed for layer %d\n", sensitivity_layer);
                        for (int i = 0; i < manifest->n_layer; i++) bridge_release(&bridges[i]);
                        free(bridges);
                        free(bridge_mask);
                        orion_qwen35_manifest_free(manifest);
                        orion_gpt2_tokenizer_free(tok);
                        return 1;
                    }
                    sensitivity_bridge_ptr = &sensitivity_bridge;
                    sensitivity_bridge_owned = 1;
                }
            }
        }

        float *cpu_curr = (float *)calloc((size_t)total, sizeof(float));
        float *cpu_next = (float *)calloc((size_t)total, sizeof(float));
        float *hybrid_curr = (float *)calloc((size_t)total, sizeof(float));
        float *hybrid_next = (float *)calloc((size_t)total, sizeof(float));
        float *sensitivity_cpu_in = (float *)calloc((size_t)total, sizeof(float));
        float *sensitivity_hybrid_in = (float *)calloc((size_t)total, sizeof(float));
        float *cpu_last = (float *)calloc((size_t)d_model, sizeof(float));
        float *hybrid_last = (float *)calloc((size_t)d_model, sizeof(float));
        float *final_norm = NULL;
        if (!cpu_curr || !cpu_next || !hybrid_curr || !hybrid_next || !sensitivity_cpu_in || !sensitivity_hybrid_in ||
            !cpu_last || !hybrid_last) {
            fprintf(stderr, "FAIL: hidden allocation failed\n");
            free(cpu_curr); free(cpu_next); free(hybrid_curr); free(hybrid_next); free(sensitivity_cpu_in); free(sensitivity_hybrid_in); free(cpu_last); free(hybrid_last);
            for (int i = 0; i < manifest->n_layer; i++) bridge_release(&bridges[i]);
            free(bridges);
            free(bridge_mask);
            orion_qwen35_manifest_free(manifest);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        if (!load_embeddings(blob_dir, manifest, token_ids, seq_len, cpu_curr) ||
            !load_embeddings(blob_dir, manifest, token_ids, seq_len, hybrid_curr)) {
            fprintf(stderr, "FAIL: embedding load failed\n");
            goto fail;
        }

        printf("TRACE: qwen35 9b hybrid layer diff\n");
        printf("  prompt=%s\n", prompt);
        printf("  prompt_len=%d\n", seq_len);
        printf("  bridge_mode=%s\n", mode);
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
        printf("  candidate_a=%d\n", candidate_a);
        printf("  candidate_b=%d\n", candidate_b);
        printf("  sensitivity_layer=%d\n", sensitivity_layer);
        printf("  gate_attr_report_topk=%d\n", gate_attr_report_topk);
        printf("  trace_dim_count=%d\n", trace_dim_count);
        for (int i = 0; i < trace_dim_count; i++) {
            printf("  trace_dim_config_rank=%d dim=%d\n", i + 1, trace_dims[i]);
        }

        int first_diverged_layer = -1;
        double first_diverged_max = 0.0;
        int worst_layer = -1;
        double worst_mean = 0.0;
        double worst_max = 0.0;
        int first_pair_pref_diverged_layer = -1;
        int worst_pair_gap_layer = -1;
        double worst_pair_gap_abs_diff = 0.0;
        int first_tail_pref_diverged_layer = -1;
        int sensitivity_captured = 0;
        const char *sensitivity_stage_kind = "none";

        char final_norm_path[2048];
        snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
        final_norm = orion_read_blob_f32_exact(final_norm_path, d_model);
        if (!final_norm) {
            fprintf(stderr, "FAIL: final_norm load failed\n");
            goto fail;
        }

        const char *lm_head_name = manifest->tie_word_embeddings ? "embed_tokens.bin" : "lm_head.bin";

        for (int layer_idx = 0; layer_idx < manifest->n_layer; layer_idx++) {
            if (layer_idx == sensitivity_layer) {
                memcpy(sensitivity_cpu_in, cpu_curr, (size_t)total * sizeof(float));
                memcpy(sensitivity_hybrid_in, hybrid_curr, (size_t)total * sizeof(float));
                sensitivity_captured = 1;
            }
            if (!apply_cpu_layer(blob_dir, manifest, layer_idx, cpu_curr, seq_len, cpu_next)) {
                fprintf(stderr, "FAIL: cpu layer apply failed at layer %d\n", layer_idx);
                goto fail;
            }
            if (!apply_hybrid_layer(blob_dir, manifest, bridges, bridge_mask, layer_idx, hybrid_curr, seq_len, hybrid_next)) {
                fprintf(stderr, "FAIL: hybrid layer apply failed at layer %d\n", layer_idx);
                goto fail;
            }

            double mean_diff = mean_abs_diff(cpu_next, hybrid_next, total);
            double max_diff = max_abs_diff(cpu_next, hybrid_next, total);
            double cpu_abs = abs_sum(cpu_next, total);
            double hybrid_abs = abs_sum(hybrid_next, total);

            char full_q_path[2048];
            snprintf(full_q_path, sizeof(full_q_path), "%s/layer%d/self_attn_q_proj.bin", blob_dir, layer_idx);
            const char *layer_type = file_exists(full_q_path) ? "full_attention" : "linear_attention";
            int bridged = bridge_mask[layer_idx] ? 1 : 0;

            printf("layer=%d type=%s bridged=%d mean_abs_diff=%.6f max_abs_diff=%.6f cpu_abs_sum=%.6f hybrid_abs_sum=%.6f\n",
                   layer_idx, layer_type, bridged, mean_diff, max_diff, cpu_abs, hybrid_abs);
            for (int i = 0; i < trace_dim_count; i++) {
                int dim = trace_dims[i];
                const double cpu_val = (double)cpu_next[(seq_len - 1) * d_model + dim];
                const double hybrid_val = (double)hybrid_next[(seq_len - 1) * d_model + dim];
                printf("trace_dim_layer=%d dim=%d cpu=%.6f hybrid=%.6f diff=%.6f\n",
                       layer_idx, dim, cpu_val, hybrid_val, fabs(cpu_val - hybrid_val));
            }

            if (first_diverged_layer < 0 && max_diff > 1e-6) {
                first_diverged_layer = layer_idx;
                first_diverged_max = max_diff;
            }
            if (worst_layer < 0 || max_diff > worst_max) {
                worst_layer = layer_idx;
                worst_mean = mean_diff;
                worst_max = max_diff;
            }

            if (pair_enabled) {
                float cpu_pair_logits[2] = {0.0f, 0.0f};
                float hybrid_pair_logits[2] = {0.0f, 0.0f};
                const int pair_ids[2] = {candidate_a, candidate_b};
                orion_qwen_cpu_rmsnorm(cpu_next + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_last);
                orion_qwen_cpu_rmsnorm(hybrid_next + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, hybrid_last);
                if (!selected_token_logits(blob_dir, lm_head_name, cpu_last, d_model, pair_ids, 2, cpu_pair_logits) ||
                    !selected_token_logits(blob_dir, lm_head_name, hybrid_last, d_model, pair_ids, 2, hybrid_pair_logits)) {
                    fprintf(stderr, "FAIL: selected pair logits failed at layer %d\n", layer_idx);
                    goto fail;
                }

                double cpu_pair_gap = (double)cpu_pair_logits[0] - (double)cpu_pair_logits[1];
                double hybrid_pair_gap = (double)hybrid_pair_logits[0] - (double)hybrid_pair_logits[1];
                int cpu_pref_token = (cpu_pair_gap >= 0.0) ? candidate_a : candidate_b;
                int hybrid_pref_token = (hybrid_pair_gap >= 0.0) ? candidate_a : candidate_b;
                double pair_gap_abs_diff = fabs(cpu_pair_gap - hybrid_pair_gap);

                printf("pair_layer=%d cpu_cand_a_logit=%.6f cpu_cand_b_logit=%.6f cpu_gap=%.6f cpu_pref_token=%d "
                       "hybrid_cand_a_logit=%.6f hybrid_cand_b_logit=%.6f hybrid_gap=%.6f hybrid_pref_token=%d pair_gap_abs_diff=%.6f\n",
                       layer_idx,
                       cpu_pair_logits[0], cpu_pair_logits[1], cpu_pair_gap, cpu_pref_token,
                       hybrid_pair_logits[0], hybrid_pair_logits[1], hybrid_pair_gap, hybrid_pref_token,
                       pair_gap_abs_diff);

                if (first_pair_pref_diverged_layer < 0 && cpu_pref_token != hybrid_pref_token) {
                    first_pair_pref_diverged_layer = layer_idx;
                }
                if (worst_pair_gap_layer < 0 || pair_gap_abs_diff > worst_pair_gap_abs_diff) {
                    worst_pair_gap_layer = layer_idx;
                    worst_pair_gap_abs_diff = pair_gap_abs_diff;
                }
            }

            if (pair_enabled) {
                OrionTailPairReplay cpu_tail = {0};
                OrionTailPairReplay hybrid_tail = {0};
                if (!run_cpu_tail_pair_replay(blob_dir, manifest, layer_idx, cpu_curr, seq_len, sample_vocab, candidate_a, candidate_b, &cpu_tail) ||
                    !run_cpu_tail_pair_replay(blob_dir, manifest, layer_idx, hybrid_curr, seq_len, sample_vocab, candidate_a, candidate_b, &hybrid_tail)) {
                    fprintf(stderr, "FAIL: cpu tail pair replay failed at layer %d\n", layer_idx);
                    goto fail;
                }
                printf("tail_pair_layer=%d cpu_input_final_pair_gap=%.6f hybrid_input_final_pair_gap=%.6f "
                       "cpu_input_final_pref_token=%d hybrid_input_final_pref_token=%d "
                       "cpu_input_top_id=%d hybrid_input_top_id=%d\n",
                       layer_idx,
                       cpu_tail.final_pair_gap, hybrid_tail.final_pair_gap,
                       cpu_tail.final_pref_token, hybrid_tail.final_pref_token,
                       cpu_tail.top_id, hybrid_tail.top_id);
                if (first_tail_pref_diverged_layer < 0 && cpu_tail.final_pref_token != hybrid_tail.final_pref_token) {
                    first_tail_pref_diverged_layer = layer_idx;
                }
            }

            float *tmp = cpu_curr; cpu_curr = cpu_next; cpu_next = tmp;
            tmp = hybrid_curr; hybrid_curr = hybrid_next; hybrid_next = tmp;
        }

        double sensitivity_input_mean_abs_diff = 0.0;
        double sensitivity_input_max_abs_diff = 0.0;
        OrionStagePairSensitivity sensitivity_cpu_input = {0};
        OrionStagePairSensitivity sensitivity_hybrid_input = {0};
        OrionLinearStageSensitivity linear_sensitivity = {0};
        OrionFullInputStageSensitivity full_input_sensitivity = {0};
        if (pair_enabled && sensitivity_layer >= 0) {
            if (!sensitivity_captured) {
                fprintf(stderr, "FAIL: sensitivity layer %d was not captured\n", sensitivity_layer);
                goto fail;
            }
            sensitivity_input_mean_abs_diff = mean_abs_diff(sensitivity_cpu_in, sensitivity_hybrid_in, total);
            sensitivity_input_max_abs_diff = max_abs_diff(sensitivity_cpu_in, sensitivity_hybrid_in, total);
            char sensitivity_q_path[2048];
            snprintf(sensitivity_q_path, sizeof(sensitivity_q_path), "%s/layer%d/self_attn_q_proj.bin", blob_dir, sensitivity_layer);
            if (file_exists(sensitivity_q_path)) {
                sensitivity_stage_kind = "full_attention";
                if (!sensitivity_bridge_ptr ||
                    !run_same_input_stage_pair_compare(blob_dir, manifest, sensitivity_bridge_ptr, sensitivity_cpu_in, seq_len, candidate_a, candidate_b, &sensitivity_cpu_input) ||
                    !run_same_input_stage_pair_compare(blob_dir, manifest, sensitivity_bridge_ptr, sensitivity_hybrid_in, seq_len, candidate_a, candidate_b, &sensitivity_hybrid_input) ||
                    !run_full_input_stage_compare(blob_dir, manifest, sensitivity_layer, sensitivity_cpu_in, sensitivity_hybrid_in,
                                                  seq_len, candidate_a, candidate_b, gate_attr_report_topk, &full_input_sensitivity)) {
                    fprintf(stderr, "FAIL: sensitivity stage pair compare failed at layer %d\n", sensitivity_layer);
                    goto fail;
                }
            } else {
                sensitivity_stage_kind = "linear_attention";
                if (!run_linear_input_stage_compare(blob_dir, manifest, sensitivity_layer, sensitivity_cpu_in, sensitivity_hybrid_in,
                                                    seq_len, candidate_a, candidate_b, &linear_sensitivity)) {
                    fprintf(stderr, "FAIL: linear sensitivity stage compare failed at layer %d\n", sensitivity_layer);
                    goto fail;
                }
            }
        }

        orion_qwen_cpu_rmsnorm(cpu_curr + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, cpu_last);
        orion_qwen_cpu_rmsnorm(hybrid_curr + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, hybrid_last);

        int cpu_top_id = -1;
        int hybrid_top_id = -1;
        float cpu_top_logit = -INFINITY;
        float hybrid_top_logit = -INFINITY;
        if (!sampled_topk_logits(blob_dir, lm_head_name, cpu_last, d_model, sample_vocab, &cpu_top_id, &cpu_top_logit) ||
            !sampled_topk_logits(blob_dir, lm_head_name, hybrid_last, d_model, sample_vocab, &hybrid_top_id, &hybrid_top_logit)) {
            fprintf(stderr, "FAIL: topk logits failed\n");
            goto fail;
        }

        double final_mean = mean_abs_diff(cpu_last, hybrid_last, d_model);
        double final_max = max_abs_diff(cpu_last, hybrid_last, d_model);

        printf("  first_diverged_layer=%d\n", first_diverged_layer);
        printf("  first_diverged_max_abs_diff=%.6f\n", first_diverged_max);
        printf("  worst_layer=%d\n", worst_layer);
        printf("  worst_layer_mean_abs_diff=%.6f\n", worst_mean);
        printf("  worst_layer_max_abs_diff=%.6f\n", worst_max);
        printf("  final_norm_mean_abs_diff=%.6f\n", final_mean);
        printf("  final_norm_max_abs_diff=%.6f\n", final_max);
        printf("  first_pair_pref_diverged_layer=%d\n", first_pair_pref_diverged_layer);
        printf("  worst_pair_gap_layer=%d\n", worst_pair_gap_layer);
        printf("  worst_pair_gap_abs_diff=%.6f\n", worst_pair_gap_abs_diff);
        printf("  first_tail_pref_diverged_layer=%d\n", first_tail_pref_diverged_layer);
        if (pair_enabled && sensitivity_layer >= 0) {
            printf("  sensitivity_stage_kind=%s\n", sensitivity_stage_kind);
            printf("  sensitivity_input_mean_abs_diff=%.6f\n", sensitivity_input_mean_abs_diff);
            printf("  sensitivity_input_max_abs_diff=%.6f\n", sensitivity_input_max_abs_diff);
            if (strcmp(sensitivity_stage_kind, "full_attention") == 0) {
                printf("  cpu_input_cpu_hidden_attn_pair_gap=%.6f\n", sensitivity_cpu_input.cpu_hidden_attn_pair_gap);
                printf("  cpu_input_ane_hidden_attn_pair_gap=%.6f\n", sensitivity_cpu_input.ane_hidden_attn_pair_gap);
                printf("  cpu_input_cpu_final_pair_gap=%.6f\n", sensitivity_cpu_input.cpu_final_pair_gap);
                printf("  cpu_input_ane_final_pair_gap=%.6f\n", sensitivity_cpu_input.ane_final_pair_gap);
                printf("  cpu_input_cpu_hidden_attn_pref_token=%d\n", sensitivity_cpu_input.cpu_hidden_attn_pref_token);
                printf("  cpu_input_ane_hidden_attn_pref_token=%d\n", sensitivity_cpu_input.ane_hidden_attn_pref_token);
                printf("  cpu_input_cpu_final_pref_token=%d\n", sensitivity_cpu_input.cpu_final_pref_token);
                printf("  cpu_input_ane_final_pref_token=%d\n", sensitivity_cpu_input.ane_final_pref_token);
                printf("  hybrid_input_cpu_hidden_attn_pair_gap=%.6f\n", sensitivity_hybrid_input.cpu_hidden_attn_pair_gap);
                printf("  hybrid_input_ane_hidden_attn_pair_gap=%.6f\n", sensitivity_hybrid_input.ane_hidden_attn_pair_gap);
                printf("  hybrid_input_cpu_final_pair_gap=%.6f\n", sensitivity_hybrid_input.cpu_final_pair_gap);
                printf("  hybrid_input_ane_final_pair_gap=%.6f\n", sensitivity_hybrid_input.ane_final_pair_gap);
                printf("  hybrid_input_cpu_hidden_attn_pref_token=%d\n", sensitivity_hybrid_input.cpu_hidden_attn_pref_token);
                printf("  hybrid_input_ane_hidden_attn_pref_token=%d\n", sensitivity_hybrid_input.ane_hidden_attn_pref_token);
                printf("  hybrid_input_cpu_final_pref_token=%d\n", sensitivity_hybrid_input.cpu_final_pref_token);
                printf("  hybrid_input_ane_final_pref_token=%d\n", sensitivity_hybrid_input.ane_final_pref_token);
                printf("  full_input_normed_mean_abs_diff=%.6f\n", full_input_sensitivity.normed_mean_abs_diff);
                printf("  full_input_normed_max_abs_diff=%.6f\n", full_input_sensitivity.normed_max_abs_diff);
                printf("  full_input_q_proj_mean_abs_diff=%.6f\n", full_input_sensitivity.q_proj_mean_abs_diff);
                printf("  full_input_q_proj_max_abs_diff=%.6f\n", full_input_sensitivity.q_proj_max_abs_diff);
                printf("  full_input_q_gate_mean_abs_diff=%.6f\n", full_input_sensitivity.q_gate_mean_abs_diff);
                printf("  full_input_q_gate_max_abs_diff=%.6f\n", full_input_sensitivity.q_gate_max_abs_diff);
                printf("  full_input_k_proj_mean_abs_diff=%.6f\n", full_input_sensitivity.k_proj_mean_abs_diff);
                printf("  full_input_k_proj_max_abs_diff=%.6f\n", full_input_sensitivity.k_proj_max_abs_diff);
                printf("  full_input_v_proj_mean_abs_diff=%.6f\n", full_input_sensitivity.v_proj_mean_abs_diff);
                printf("  full_input_v_proj_max_abs_diff=%.6f\n", full_input_sensitivity.v_proj_max_abs_diff);
                printf("  full_input_attn_gate_sigmoid_mean_abs_diff=%.6f\n", full_input_sensitivity.attn_gate_sigmoid_mean_abs_diff);
                printf("  full_input_attn_gate_sigmoid_max_abs_diff=%.6f\n", full_input_sensitivity.attn_gate_sigmoid_max_abs_diff);
                printf("  full_input_attn_context_mean_abs_diff=%.6f\n", full_input_sensitivity.attn_context_mean_abs_diff);
                printf("  full_input_attn_context_max_abs_diff=%.6f\n", full_input_sensitivity.attn_context_max_abs_diff);
                printf("  full_input_attn_gated_context_mean_abs_diff=%.6f\n", full_input_sensitivity.attn_gated_context_mean_abs_diff);
                printf("  full_input_attn_gated_context_max_abs_diff=%.6f\n", full_input_sensitivity.attn_gated_context_max_abs_diff);
                printf("  full_input_attn_out_mean_abs_diff=%.6f\n", full_input_sensitivity.attn_out_mean_abs_diff);
                printf("  full_input_attn_out_max_abs_diff=%.6f\n", full_input_sensitivity.attn_out_max_abs_diff);
                printf("  full_input_attn_qgate_only_mean_abs_diff=%.6f\n", full_input_sensitivity.attn_qgate_only_mean_abs_diff);
                printf("  full_input_attn_qgate_only_max_abs_diff=%.6f\n", full_input_sensitivity.attn_qgate_only_max_abs_diff);
                printf("  full_input_attn_v_only_mean_abs_diff=%.6f\n", full_input_sensitivity.attn_v_only_mean_abs_diff);
                printf("  full_input_attn_v_only_max_abs_diff=%.6f\n", full_input_sensitivity.attn_v_only_max_abs_diff);
                printf("  full_input_attn_context_only_mean_abs_diff=%.6f\n", full_input_sensitivity.attn_context_only_mean_abs_diff);
                printf("  full_input_attn_context_only_max_abs_diff=%.6f\n", full_input_sensitivity.attn_context_only_max_abs_diff);
                printf("  full_input_attn_sigmoid_only_mean_abs_diff=%.6f\n", full_input_sensitivity.attn_sigmoid_only_mean_abs_diff);
                printf("  full_input_attn_sigmoid_only_max_abs_diff=%.6f\n", full_input_sensitivity.attn_sigmoid_only_max_abs_diff);
                printf("  full_input_attn_gated_context_only_mean_abs_diff=%.6f\n", full_input_sensitivity.attn_gated_context_only_mean_abs_diff);
                printf("  full_input_attn_gated_context_only_max_abs_diff=%.6f\n", full_input_sensitivity.attn_gated_context_only_max_abs_diff);
                printf("  full_input_hidden_attn_mean_abs_diff=%.6f\n", full_input_sensitivity.hidden_attn_mean_abs_diff);
                printf("  full_input_hidden_attn_max_abs_diff=%.6f\n", full_input_sensitivity.hidden_attn_max_abs_diff);
                printf("  full_input_ffn_rms_mean_abs_diff=%.6f\n", full_input_sensitivity.ffn_rms_mean_abs_diff);
                printf("  full_input_ffn_rms_max_abs_diff=%.6f\n", full_input_sensitivity.ffn_rms_max_abs_diff);
                printf("  full_input_ffn_gate_mean_abs_diff=%.6f\n", full_input_sensitivity.ffn_gate_mean_abs_diff);
                printf("  full_input_ffn_gate_max_abs_diff=%.6f\n", full_input_sensitivity.ffn_gate_max_abs_diff);
                printf("  full_input_ffn_up_mean_abs_diff=%.6f\n", full_input_sensitivity.ffn_up_mean_abs_diff);
                printf("  full_input_ffn_up_max_abs_diff=%.6f\n", full_input_sensitivity.ffn_up_max_abs_diff);
                printf("  full_input_ffn_silu_mean_abs_diff=%.6f\n", full_input_sensitivity.ffn_silu_mean_abs_diff);
                printf("  full_input_ffn_silu_max_abs_diff=%.6f\n", full_input_sensitivity.ffn_silu_max_abs_diff);
                printf("  full_input_ffn_hidden_mean_abs_diff=%.6f\n", full_input_sensitivity.ffn_hidden_mean_abs_diff);
                printf("  full_input_ffn_hidden_max_abs_diff=%.6f\n", full_input_sensitivity.ffn_hidden_max_abs_diff);
                printf("  full_input_ffn_down_mean_abs_diff=%.6f\n", full_input_sensitivity.ffn_down_mean_abs_diff);
                printf("  full_input_ffn_down_max_abs_diff=%.6f\n", full_input_sensitivity.ffn_down_max_abs_diff);
                printf("  full_input_final_mean_abs_diff=%.6f\n", full_input_sensitivity.final_mean_abs_diff);
                printf("  full_input_final_max_abs_diff=%.6f\n", full_input_sensitivity.final_max_abs_diff);
                printf("  full_input_cpu_hidden_attn_pair_gap=%.6f\n", full_input_sensitivity.cpu_hidden_attn_pair_gap);
                printf("  full_input_hybrid_hidden_attn_pair_gap=%.6f\n", full_input_sensitivity.hybrid_hidden_attn_pair_gap);
                printf("  full_input_cpu_res_hybrid_qgate_hidden_attn_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_qgate_hidden_attn_pair_gap);
                printf("  full_input_cpu_res_hybrid_v_hidden_attn_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_v_hidden_attn_pair_gap);
                printf("  full_input_cpu_res_hybrid_context_hidden_attn_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_context_hidden_attn_pair_gap);
                printf("  full_input_cpu_res_hybrid_sigmoid_hidden_attn_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_sigmoid_hidden_attn_pair_gap);
                printf("  full_input_cpu_res_hybrid_gated_context_hidden_attn_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_gated_context_hidden_attn_pair_gap);
                printf("  full_input_cpu_final_pair_gap=%.6f\n", full_input_sensitivity.cpu_final_pair_gap);
                printf("  full_input_hybrid_final_pair_gap=%.6f\n", full_input_sensitivity.hybrid_final_pair_gap);
                printf("  full_input_cpu_res_hybrid_qgate_final_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_qgate_final_pair_gap);
                printf("  full_input_cpu_res_hybrid_v_final_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_v_final_pair_gap);
                printf("  full_input_cpu_res_hybrid_context_final_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_context_final_pair_gap);
                printf("  full_input_cpu_res_hybrid_sigmoid_final_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_sigmoid_final_pair_gap);
                printf("  full_input_cpu_res_hybrid_gated_context_final_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_gated_context_final_pair_gap);
                printf("  full_input_cpu_hidden_attn_pref_token=%d\n", full_input_sensitivity.cpu_hidden_attn_pref_token);
                printf("  full_input_hybrid_hidden_attn_pref_token=%d\n", full_input_sensitivity.hybrid_hidden_attn_pref_token);
                printf("  full_input_cpu_res_hybrid_qgate_hidden_attn_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_qgate_hidden_attn_pref_token);
                printf("  full_input_cpu_res_hybrid_v_hidden_attn_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_v_hidden_attn_pref_token);
                printf("  full_input_cpu_res_hybrid_context_hidden_attn_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_context_hidden_attn_pref_token);
                printf("  full_input_cpu_res_hybrid_sigmoid_hidden_attn_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_sigmoid_hidden_attn_pref_token);
                printf("  full_input_cpu_res_hybrid_gated_context_hidden_attn_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_gated_context_hidden_attn_pref_token);
                printf("  full_input_cpu_final_pref_token=%d\n", full_input_sensitivity.cpu_final_pref_token);
                printf("  full_input_hybrid_final_pref_token=%d\n", full_input_sensitivity.hybrid_final_pref_token);
                printf("  full_input_cpu_res_hybrid_qgate_final_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_qgate_final_pref_token);
                printf("  full_input_cpu_res_hybrid_v_final_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_v_final_pref_token);
                printf("  full_input_cpu_res_hybrid_context_final_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_context_final_pref_token);
                printf("  full_input_cpu_res_hybrid_sigmoid_final_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_sigmoid_final_pref_token);
                printf("  full_input_cpu_res_hybrid_gated_context_final_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_gated_context_final_pref_token);
                printf("  full_input_attn_v_attr_count=%d\n", full_input_sensitivity.attn_v_attr_count);
                for (int rank = 0; rank < full_input_sensitivity.attn_v_attr_count; rank++) {
                    printf("  full_input_attn_v_attr_rank=%d dim=%d cpu_norm=%.6f hybrid_norm=%.6f delta_norm=%.6f pair_delta=%.6f abs_contrib=%.6f signed_contrib=%.6f\n",
                           rank + 1,
                           full_input_sensitivity.attn_v_attr_dims[rank],
                           full_input_sensitivity.attn_v_attr_cpu_norm[rank],
                           full_input_sensitivity.attn_v_attr_hybrid_norm[rank],
                           full_input_sensitivity.attn_v_attr_delta_norm[rank],
                           full_input_sensitivity.attn_v_attr_pair_delta[rank],
                           full_input_sensitivity.attn_v_attr_abs_contrib[rank],
                           full_input_sensitivity.attn_v_attr_signed_contrib[rank]);
                }
                printf("  full_input_attn_gated_context_attr_count=%d\n", full_input_sensitivity.attn_gated_context_attr_count);
                for (int rank = 0; rank < full_input_sensitivity.attn_gated_context_attr_count; rank++) {
                    printf("  full_input_attn_gated_context_attr_rank=%d dim=%d cpu_norm=%.6f hybrid_norm=%.6f delta_norm=%.6f pair_delta=%.6f abs_contrib=%.6f signed_contrib=%.6f\n",
                           rank + 1,
                           full_input_sensitivity.attn_gated_context_attr_dims[rank],
                           full_input_sensitivity.attn_gated_context_attr_cpu_norm[rank],
                           full_input_sensitivity.attn_gated_context_attr_hybrid_norm[rank],
                           full_input_sensitivity.attn_gated_context_attr_delta_norm[rank],
                           full_input_sensitivity.attn_gated_context_attr_pair_delta[rank],
                           full_input_sensitivity.attn_gated_context_attr_abs_contrib[rank],
                           full_input_sensitivity.attn_gated_context_attr_signed_contrib[rank]);
                }
                printf("  full_input_ffn_down_attr_count=%d\n", full_input_sensitivity.ffn_down_attr_count);
                for (int rank = 0; rank < full_input_sensitivity.ffn_down_attr_count; rank++) {
                    printf("  full_input_ffn_down_attr_rank=%d dim=%d cpu_norm=%.6f hybrid_norm=%.6f delta_norm=%.6f pair_delta=%.6f abs_contrib=%.6f signed_contrib=%.6f\n",
                           rank + 1,
                           full_input_sensitivity.ffn_down_attr_dims[rank],
                           full_input_sensitivity.ffn_down_attr_cpu_norm[rank],
                           full_input_sensitivity.ffn_down_attr_hybrid_norm[rank],
                           full_input_sensitivity.ffn_down_attr_delta_norm[rank],
                           full_input_sensitivity.ffn_down_attr_pair_delta[rank],
                           full_input_sensitivity.ffn_down_attr_abs_contrib[rank],
                           full_input_sensitivity.ffn_down_attr_signed_contrib[rank]);
                }
                printf("  full_input_cpu_res_hybrid_down_final_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_down_final_pair_gap);
                printf("  full_input_hybrid_res_cpu_down_final_pair_gap=%.6f\n", full_input_sensitivity.hybrid_res_cpu_down_final_pair_gap);
                printf("  full_input_cpu_res_hybrid_down_final_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_down_final_pref_token);
                printf("  full_input_hybrid_res_cpu_down_final_pref_token=%d\n", full_input_sensitivity.hybrid_res_cpu_down_final_pref_token);
                printf("  full_input_cpu_res_cpu_silu_hybrid_up_final_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_cpu_silu_hybrid_up_final_pair_gap);
                printf("  full_input_cpu_res_hybrid_silu_cpu_up_final_pair_gap=%.6f\n", full_input_sensitivity.cpu_res_hybrid_silu_cpu_up_final_pair_gap);
                printf("  full_input_cpu_res_cpu_silu_hybrid_up_final_pref_token=%d\n", full_input_sensitivity.cpu_res_cpu_silu_hybrid_up_final_pref_token);
                printf("  full_input_cpu_res_hybrid_silu_cpu_up_final_pref_token=%d\n", full_input_sensitivity.cpu_res_hybrid_silu_cpu_up_final_pref_token);
                printf("  full_input_gate_attr_total_channels=%d\n", full_input_sensitivity.gate_attr_total_channels);
                printf("  full_input_gate_attr_report_count=%d\n", full_input_sensitivity.gate_attr_report_count);
                printf("  full_input_gate_attr_first_cumulative_flip_rank=%d\n", full_input_sensitivity.gate_attr_first_cumulative_flip_rank);
                printf("  full_input_gate_attr_flip_channel=%d\n", full_input_sensitivity.gate_attr_flip_channel);
                printf("  full_input_gate_attr_flip_delta_hidden=%.6f\n", full_input_sensitivity.gate_attr_flip_delta_hidden);
                printf("  full_input_gate_attr_flip_cumulative_pair_gap=%.6f\n", full_input_sensitivity.gate_attr_flip_cumulative_pair_gap);
                printf("  full_input_gate_attr_all_cumulative_pair_gap=%.6f\n", full_input_sensitivity.gate_attr_all_cumulative_pair_gap);
                printf("  full_input_gate_input_attr_channel_limit=%d\n", full_input_sensitivity.gate_input_attr_channel_limit);
                printf("  full_input_gate_input_attr_count=%d\n", full_input_sensitivity.gate_input_attr_count);
                for (int rank = 0; rank < full_input_sensitivity.gate_attr_report_count; rank++) {
                    printf("  full_input_gate_attr_rank=%d channel=%d cpu_gate=%.6f hybrid_gate=%.6f delta_gate=%.6f cpu_silu=%.6f hybrid_silu=%.6f delta_silu=%.6f cpu_up=%.6f delta_hidden=%.6f single_pair_gap=%.6f single_pref_token=%d cumulative_pair_gap=%.6f cumulative_pref_token=%d\n",
                           rank + 1,
                           full_input_sensitivity.gate_attr_channels[rank],
                           full_input_sensitivity.gate_attr_cpu_gate[rank],
                           full_input_sensitivity.gate_attr_hybrid_gate[rank],
                           full_input_sensitivity.gate_attr_delta_gate[rank],
                           full_input_sensitivity.gate_attr_cpu_silu[rank],
                           full_input_sensitivity.gate_attr_hybrid_silu[rank],
                           full_input_sensitivity.gate_attr_delta_silu[rank],
                           full_input_sensitivity.gate_attr_cpu_up[rank],
                           full_input_sensitivity.gate_attr_delta_hidden[rank],
                           full_input_sensitivity.gate_attr_single_pair_gap[rank],
                           full_input_sensitivity.gate_attr_single_pref_token[rank],
                           full_input_sensitivity.gate_attr_cumulative_pair_gap[rank],
                           full_input_sensitivity.gate_attr_cumulative_pref_token[rank]);
                }
                for (int rank = 0; rank < full_input_sensitivity.gate_input_attr_count; rank++) {
                    printf("  full_input_gate_input_attr_rank=%d dim=%d delta_ffn_rms=%.6f abs_contrib=%.6f signed_contrib=%.6f\n",
                           rank + 1,
                           full_input_sensitivity.gate_input_attr_dims[rank],
                           full_input_sensitivity.gate_input_attr_delta_ffn_rms[rank],
                           full_input_sensitivity.gate_input_attr_abs_contrib[rank],
                           full_input_sensitivity.gate_input_attr_signed_contrib[rank]);
                }
            } else if (strcmp(sensitivity_stage_kind, "linear_attention") == 0) {
                printf("  linear_normed_mean_abs_diff=%.6f\n", linear_sensitivity.normed_mean_abs_diff);
                printf("  linear_normed_max_abs_diff=%.6f\n", linear_sensitivity.normed_max_abs_diff);
                printf("  linear_mixed_linear_mean_abs_diff=%.6f\n", linear_sensitivity.mixed_linear_mean_abs_diff);
                printf("  linear_mixed_linear_max_abs_diff=%.6f\n", linear_sensitivity.mixed_linear_max_abs_diff);
                printf("  linear_mixed_conv_mean_abs_diff=%.6f\n", linear_sensitivity.mixed_conv_mean_abs_diff);
                printf("  linear_mixed_conv_max_abs_diff=%.6f\n", linear_sensitivity.mixed_conv_max_abs_diff);
                printf("  linear_query_mean_abs_diff=%.6f\n", linear_sensitivity.query_mean_abs_diff);
                printf("  linear_query_max_abs_diff=%.6f\n", linear_sensitivity.query_max_abs_diff);
                printf("  linear_key_mean_abs_diff=%.6f\n", linear_sensitivity.key_mean_abs_diff);
                printf("  linear_key_max_abs_diff=%.6f\n", linear_sensitivity.key_max_abs_diff);
                printf("  linear_value_mean_abs_diff=%.6f\n", linear_sensitivity.value_mean_abs_diff);
                printf("  linear_value_max_abs_diff=%.6f\n", linear_sensitivity.value_max_abs_diff);
                printf("  linear_z_mean_abs_diff=%.6f\n", linear_sensitivity.z_mean_abs_diff);
                printf("  linear_z_max_abs_diff=%.6f\n", linear_sensitivity.z_max_abs_diff);
                printf("  linear_beta_mean_abs_diff=%.6f\n", linear_sensitivity.beta_mean_abs_diff);
                printf("  linear_beta_max_abs_diff=%.6f\n", linear_sensitivity.beta_max_abs_diff);
                printf("  linear_g_mean_abs_diff=%.6f\n", linear_sensitivity.g_mean_abs_diff);
                printf("  linear_g_max_abs_diff=%.6f\n", linear_sensitivity.g_max_abs_diff);
                printf("  linear_core_pre_mean_abs_diff=%.6f\n", linear_sensitivity.core_pre_mean_abs_diff);
                printf("  linear_core_pre_max_abs_diff=%.6f\n", linear_sensitivity.core_pre_max_abs_diff);
                printf("  linear_core_mean_abs_diff=%.6f\n", linear_sensitivity.core_mean_abs_diff);
                printf("  linear_core_max_abs_diff=%.6f\n", linear_sensitivity.core_max_abs_diff);
                printf("  linear_attn_out_mean_abs_diff=%.6f\n", linear_sensitivity.attn_out_mean_abs_diff);
                printf("  linear_attn_out_max_abs_diff=%.6f\n", linear_sensitivity.attn_out_max_abs_diff);
                printf("  linear_hidden_attn_mean_abs_diff=%.6f\n", linear_sensitivity.hidden_attn_mean_abs_diff);
                printf("  linear_hidden_attn_max_abs_diff=%.6f\n", linear_sensitivity.hidden_attn_max_abs_diff);
                printf("  linear_final_mean_abs_diff=%.6f\n", linear_sensitivity.final_mean_abs_diff);
                printf("  linear_final_max_abs_diff=%.6f\n", linear_sensitivity.final_max_abs_diff);
                printf("  linear_cpu_hidden_attn_pair_gap=%.6f\n", linear_sensitivity.cpu_hidden_attn_pair_gap);
                printf("  linear_hybrid_hidden_attn_pair_gap=%.6f\n", linear_sensitivity.hybrid_hidden_attn_pair_gap);
                printf("  linear_cpu_final_pair_gap=%.6f\n", linear_sensitivity.cpu_final_pair_gap);
                printf("  linear_hybrid_final_pair_gap=%.6f\n", linear_sensitivity.hybrid_final_pair_gap);
                printf("  linear_cpu_hidden_attn_pref_token=%d\n", linear_sensitivity.cpu_hidden_attn_pref_token);
                printf("  linear_hybrid_hidden_attn_pref_token=%d\n", linear_sensitivity.hybrid_hidden_attn_pref_token);
                printf("  linear_cpu_final_pref_token=%d\n", linear_sensitivity.cpu_final_pref_token);
                printf("  linear_hybrid_final_pref_token=%d\n", linear_sensitivity.hybrid_final_pref_token);
            }
        }
        printf("  cpu_top_id=%d\n", cpu_top_id);
        printf("  cpu_top_logit=%.6f\n", cpu_top_logit);
        printf("  hybrid_top_id=%d\n", hybrid_top_id);
        printf("  hybrid_top_logit=%.6f\n", hybrid_top_logit);
        printf("PASS: qwen35 9b hybrid layer diff trace\n");

        free(final_norm);
        free(cpu_curr); free(cpu_next); free(hybrid_curr); free(hybrid_next); free(sensitivity_cpu_in); free(sensitivity_hybrid_in); free(cpu_last); free(hybrid_last);
        if (sensitivity_bridge_owned) bridge_release(&sensitivity_bridge);
        for (int i = 0; i < manifest->n_layer; i++) bridge_release(&bridges[i]);
        free(bridges);
        free(bridge_mask);
        orion_qwen35_manifest_free(manifest);
        orion_gpt2_tokenizer_free(tok);
        return 0;

fail:
        free(final_norm);
        free(cpu_curr); free(cpu_next); free(hybrid_curr); free(hybrid_next); free(sensitivity_cpu_in); free(sensitivity_hybrid_in); free(cpu_last); free(hybrid_last);
        if (sensitivity_bridge_owned) bridge_release(&sensitivity_bridge);
        for (int i = 0; i < manifest->n_layer; i++) bridge_release(&bridges[i]);
        free(bridges);
        free(bridge_mask);
        orion_qwen35_manifest_free(manifest);
        orion_gpt2_tokenizer_free(tok);
        return 1;
    }
}
