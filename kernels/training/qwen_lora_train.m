#import "qwen_lora_train.h"
#import "../inference/qwen_cpu_ops.h"
#import "../training/stories_cpu_ops.h"
#include "compiler/frontends/qwen35_prefill.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#import "compiler/codegen.h"
#include "core/ane_runtime.h"
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <unistd.h>

static int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

static float *load_layer_exact(const char *blob_dir, int layer_idx, const char *suffix, int count) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/layer%d/%s", blob_dir, layer_idx, suffix);
    return orion_read_blob_f32_exact(path, count);
}

static int load_embed_row(const char *blob_dir, int token_id, int d_model, float *out_row) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/model/embed_tokens.bin", blob_dir);
    return orion_read_blob_row_f32(path, token_id, d_model, out_row);
}

static void expand_grouped_value(const float *v,
                                 int n_head,
                                 int n_kv_head,
                                 int head_dim,
                                 float *attn_cat) {
    int q_per_kv = n_head / n_kv_head;
    for (int h = 0; h < n_head; h++) {
        int kv_head = h / q_per_kv;
        memcpy(attn_cat + (size_t)h * head_dim,
               v + (size_t)kv_head * head_dim,
               (size_t)head_dim * sizeof(float));
    }
}

static void reduce_grouped_value_grad(const float *d_attn_cat,
                                      int n_head,
                                      int n_kv_head,
                                      int head_dim,
                                      float *d_v) {
    int q_per_kv = n_head / n_kv_head;
    memset(d_v, 0, (size_t)n_kv_head * head_dim * sizeof(float));
    for (int h = 0; h < n_head; h++) {
        int kv_head = h / q_per_kv;
        float *dst = d_v + (size_t)kv_head * head_dim;
        const float *src = d_attn_cat + (size_t)h * head_dim;
        for (int i = 0; i < head_dim; i++) dst[i] += src[i];
    }
}

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
        while (*end && isspace((unsigned char)*end)) end++;
        if (*end != '\0') continue;
        int duplicate = 0;
        for (int i = 0; i < count; i++) {
            if (out[i] == (int)value) {
                duplicate = 1;
                break;
            }
        }
        if (!duplicate) out[count++] = (int)value;
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

struct OrionQwen9BCPUTrainContext {
    int d_model;
    int d_ff;
    int q_dim;
    int kv_dim;
    float *input_ln;
    float *post_ln;
    float *q_proj;
    float *v_proj;
    float *o_proj;
    float *gate_proj;
    float *up_proj;
    float *down_proj;
    float *final_norm_weight;
    float *normed;
    float *q_full;
    float *v_raw;
    float *attn_cat;
    float *gated;
    float *mixer;
    float *hidden_mid;
    float *post_norm;
    float *mlp_out;
    float *hidden_out;
    float *final_norm;
    float *last_hidden;
    float *d_last_hidden;
    float *d_hidden_out;
    float *d_post_norm;
    float *d_hidden_mid;
    float *d_mixer;
    float *d_gated;
    float *d_attn_cat;
    float *d_gate_half;
    float *d_q_full;
    float *d_v;
    float *d_normed_from_q;
    float *d_normed_from_v;
    float *d_normed;
    float *d_hidden_mid_from_post;
    float *d_post_ln_weight_grad;
    float *throwaway_weight_grad;
    float *throwaway_dx;
};

static void orion_qwen9b_cpu_train_context_close(OrionQwen9BCPUTrainContext *ctx) {
    if (!ctx) return;
    free(ctx->input_ln); free(ctx->post_ln); free(ctx->q_proj); free(ctx->v_proj); free(ctx->o_proj);
    free(ctx->gate_proj); free(ctx->up_proj); free(ctx->down_proj); free(ctx->final_norm_weight);
    free(ctx->normed); free(ctx->q_full); free(ctx->v_raw); free(ctx->attn_cat); free(ctx->gated);
    free(ctx->mixer); free(ctx->hidden_mid); free(ctx->post_norm); free(ctx->mlp_out); free(ctx->hidden_out);
    free(ctx->final_norm); free(ctx->last_hidden); free(ctx->d_last_hidden); free(ctx->d_hidden_out);
    free(ctx->d_post_norm); free(ctx->d_hidden_mid); free(ctx->d_mixer); free(ctx->d_gated); free(ctx->d_attn_cat);
    free(ctx->d_gate_half); free(ctx->d_q_full); free(ctx->d_v); free(ctx->d_normed_from_q); free(ctx->d_normed_from_v);
    free(ctx->d_normed); free(ctx->d_hidden_mid_from_post); free(ctx->d_post_ln_weight_grad);
    free(ctx->throwaway_weight_grad); free(ctx->throwaway_dx);
    free(ctx);
}

static OrionQwen9BCPUTrainContext *orion_qwen9b_cpu_train_context_open(const char *blob_dir,
                                                                       const OrionQwen35Manifest *manifest,
                                                                       int layer_idx) {
    if (!blob_dir || !manifest) return NULL;

    OrionQwen9BCPUTrainContext *ctx = (OrionQwen9BCPUTrainContext *)calloc(1, sizeof(OrionQwen9BCPUTrainContext));
    if (!ctx) return NULL;

    ctx->d_model = manifest->d_model;
    ctx->d_ff = manifest->d_ff;
    ctx->q_dim = manifest->n_head * manifest->head_dim;
    ctx->kv_dim = manifest->n_kv_head * manifest->head_dim;

    ctx->input_ln = load_layer_exact(blob_dir, layer_idx, "input_layernorm.bin", ctx->d_model);
    ctx->post_ln = load_layer_exact(blob_dir, layer_idx, "post_attention_layernorm.bin", ctx->d_model);
    ctx->q_proj = load_layer_exact(blob_dir, layer_idx, "self_attn_q_proj.bin", (ctx->q_dim * 2) * ctx->d_model);
    ctx->v_proj = load_layer_exact(blob_dir, layer_idx, "self_attn_v_proj.bin", ctx->kv_dim * ctx->d_model);
    ctx->o_proj = load_layer_exact(blob_dir, layer_idx, "self_attn_o_proj.bin", ctx->d_model * ctx->q_dim);
    ctx->gate_proj = load_layer_exact(blob_dir, layer_idx, "mlp_gate_proj.bin", ctx->d_ff * ctx->d_model);
    ctx->up_proj = load_layer_exact(blob_dir, layer_idx, "mlp_up_proj.bin", ctx->d_ff * ctx->d_model);
    ctx->down_proj = load_layer_exact(blob_dir, layer_idx, "mlp_down_proj.bin", ctx->d_model * ctx->d_ff);
    char final_norm_path[2048];
    snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
    ctx->final_norm_weight = orion_read_blob_f32_exact(final_norm_path, ctx->d_model);

    ctx->normed = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->q_full = (float *)calloc((size_t)(ctx->q_dim * 2), sizeof(float));
    ctx->v_raw = (float *)calloc((size_t)ctx->kv_dim, sizeof(float));
    ctx->attn_cat = (float *)calloc((size_t)ctx->q_dim, sizeof(float));
    ctx->gated = (float *)calloc((size_t)ctx->q_dim, sizeof(float));
    ctx->mixer = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->hidden_mid = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->post_norm = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->mlp_out = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->hidden_out = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->final_norm = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->last_hidden = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->d_last_hidden = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->d_hidden_out = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->d_post_norm = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->d_hidden_mid = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->d_mixer = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->d_gated = (float *)calloc((size_t)ctx->q_dim, sizeof(float));
    ctx->d_attn_cat = (float *)calloc((size_t)ctx->q_dim, sizeof(float));
    ctx->d_gate_half = (float *)calloc((size_t)ctx->q_dim, sizeof(float));
    ctx->d_q_full = (float *)calloc((size_t)(ctx->q_dim * 2), sizeof(float));
    ctx->d_v = (float *)calloc((size_t)ctx->kv_dim, sizeof(float));
    ctx->d_normed_from_q = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->d_normed_from_v = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->d_normed = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->d_hidden_mid_from_post = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->d_post_ln_weight_grad = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->throwaway_weight_grad = (float *)calloc((size_t)ctx->d_model, sizeof(float));
    ctx->throwaway_dx = (float *)calloc((size_t)ctx->d_model, sizeof(float));

    if (!ctx->input_ln || !ctx->post_ln || !ctx->q_proj || !ctx->v_proj || !ctx->o_proj ||
        !ctx->gate_proj || !ctx->up_proj || !ctx->down_proj || !ctx->final_norm_weight ||
        !ctx->normed || !ctx->q_full || !ctx->v_raw || !ctx->attn_cat || !ctx->gated ||
        !ctx->mixer || !ctx->hidden_mid || !ctx->post_norm || !ctx->mlp_out || !ctx->hidden_out ||
        !ctx->final_norm || !ctx->last_hidden || !ctx->d_last_hidden || !ctx->d_hidden_out ||
        !ctx->d_post_norm || !ctx->d_hidden_mid || !ctx->d_mixer || !ctx->d_gated ||
        !ctx->d_attn_cat || !ctx->d_gate_half || !ctx->d_q_full || !ctx->d_v ||
        !ctx->d_normed_from_q || !ctx->d_normed_from_v || !ctx->d_normed ||
        !ctx->d_hidden_mid_from_post || !ctx->d_post_ln_weight_grad ||
        !ctx->throwaway_weight_grad || !ctx->throwaway_dx) {
        orion_qwen9b_cpu_train_context_close(ctx);
        return NULL;
    }

    return ctx;
}

static void orion_qwen9b_cpu_train_context_reset(OrionQwen9BCPUTrainContext *ctx) {
    if (!ctx) return;
    memset(ctx->normed, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->q_full, 0, (size_t)(ctx->q_dim * 2) * sizeof(float));
    memset(ctx->v_raw, 0, (size_t)ctx->kv_dim * sizeof(float));
    memset(ctx->attn_cat, 0, (size_t)ctx->q_dim * sizeof(float));
    memset(ctx->gated, 0, (size_t)ctx->q_dim * sizeof(float));
    memset(ctx->mixer, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->hidden_mid, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->post_norm, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->mlp_out, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->hidden_out, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->final_norm, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->last_hidden, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->d_last_hidden, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->d_hidden_out, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->d_post_norm, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->d_hidden_mid, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->d_mixer, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->d_gated, 0, (size_t)ctx->q_dim * sizeof(float));
    memset(ctx->d_attn_cat, 0, (size_t)ctx->q_dim * sizeof(float));
    memset(ctx->d_gate_half, 0, (size_t)ctx->q_dim * sizeof(float));
    memset(ctx->d_q_full, 0, (size_t)(ctx->q_dim * 2) * sizeof(float));
    memset(ctx->d_v, 0, (size_t)ctx->kv_dim * sizeof(float));
    memset(ctx->d_normed_from_q, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->d_normed_from_v, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->d_normed, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->d_hidden_mid_from_post, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->d_post_ln_weight_grad, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->throwaway_weight_grad, 0, (size_t)ctx->d_model * sizeof(float));
    memset(ctx->throwaway_dx, 0, (size_t)ctx->d_model * sizeof(float));
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
    int layer_idx;
    int bucket;
    int d_model;
    int q_dim;
    int kv_dim;
    int qkv_input_mode;
    int q_uses_cpu_rms;
    int kv_uses_cpu_rms;
    int compile_cache_hit;
    int q_cache_hit;
    int kv_cache_hit;
    int runtime_artifact_exported;
    char compile_cache_source[64];
    char blob_dir[2048];
} OrionQwenAneTrainBridge;

static OrionQwenAneTrainBridge g_ane_train_bridge = {0};

enum {
    ORION_QKV_INPUT_MODE_ANE_RMS = 0,
    ORION_QKV_INPUT_MODE_CPU_RMS = 1,
    ORION_QKV_INPUT_MODE_KV_CPU_RMS = 2,
};

static int parse_qkv_input_mode(void) {
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

static const char *compile_cache_root_env(void) {
    const char *root = getenv("ORION_COMPILE_CACHE_DIR");
    if (root && *root) return root;
    root = getenv("COMPILE_CACHE_DIR");
    return (root && *root) ? root : NULL;
}

static const char *compile_cache_dataset_hash_env(void) {
    const char *value = getenv("ORION_COMPILE_CACHE_DATASET_HASH");
    if (value && *value) return value;
    value = getenv("DATASET_HASH");
    return (value && *value) ? value : "unknown";
}

static NSString *qkv_input_mode_name(int qkv_input_mode) {
    switch (qkv_input_mode) {
        case ORION_QKV_INPUT_MODE_CPU_RMS: return @"cpu_rms";
        case ORION_QKV_INPUT_MODE_KV_CPU_RMS: return @"kv_cpu_rms";
        default: return @"ane_rms";
    }
}

static NSString *ane_train_bridge_compile_artifact_dir(const char *blob_dir,
                                                       int layer_idx,
                                                       int bucket,
                                                       int qkv_input_mode,
                                                       const char *component) {
    const char *cache_root = compile_cache_root_env();
    if (!cache_root || !component || !*component) return nil;
    NSString *blobBase = [[NSString stringWithUTF8String:blob_dir] lastPathComponent];
    NSString *datasetHash = [NSString stringWithUTF8String:compile_cache_dataset_hash_env()];
    NSString *modeName = qkv_input_mode_name(qkv_input_mode);
    NSString *artifactName = [NSString stringWithFormat:@"qwen35_9b_lora_train_%@_%@_L%d_B%d_%@_%s",
                              blobBase,
                              datasetHash,
                              layer_idx,
                              bucket,
                              modeName,
                              component];
    return [[NSString stringWithUTF8String:cache_root] stringByAppendingPathComponent:artifactName];
}

static void ane_train_bridge_release(void) {
    if (g_ane_train_bridge.prog_q) orion_release_program(g_ane_train_bridge.prog_q);
    if (g_ane_train_bridge.prog_kv) orion_release_program(g_ane_train_bridge.prog_kv);
    memset(&g_ane_train_bridge, 0, sizeof(g_ane_train_bridge));
}

static int ane_train_bridge_ensure(const char *blob_dir,
                                   OrionQwen35Manifest *manifest,
                                   int layer_idx,
                                   int bucket) {
    if (g_ane_train_bridge.prog_q &&
        g_ane_train_bridge.prog_kv &&
        g_ane_train_bridge.layer_idx == layer_idx &&
        g_ane_train_bridge.bucket == bucket &&
        g_ane_train_bridge.qkv_input_mode == parse_qkv_input_mode() &&
        strcmp(g_ane_train_bridge.blob_dir, blob_dir) == 0) {
        return 1;
    }

    ane_train_bridge_release();
    if (!orion_ane_init()) return 0;

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
    NSString *blobDir = [NSString stringWithUTF8String:blob_dir];
    int qkv_input_mode = parse_qkv_input_mode();
    int q_uses_cpu_rms = q_uses_cpu_rms_mode(qkv_input_mode);
    int kv_uses_cpu_rms = kv_uses_cpu_rms_mode(qkv_input_mode);
    NSString *mil_q = compile_graph(
        q_uses_cpu_rms
            ? orion_frontend_qwen35_prefill_q_proj_linear_only(layer_idx, bucket, &cfg)
            : orion_frontend_qwen35_prefill_q_proj(layer_idx, bucket, &cfg)
    );
    NSString *mil_kv = compile_graph(
        kv_uses_cpu_rms
            ? orion_frontend_qwen35_prefill_kv_proj_linear_only(layer_idx, bucket, &cfg)
            : orion_frontend_qwen35_prefill_kv_proj(layer_idx, bucket, &cfg)
    );
    if (!mil_q || !mil_kv) return 0;

    NSDictionary *wdict_q = q_uses_cpu_rms ? build_qproj_linear_only_wdict(layer_idx, blobDir) : build_qproj_wdict(layer_idx, blobDir);
    NSDictionary *wdict_kv = kv_uses_cpu_rms ? build_kv_linear_only_wdict(layer_idx, blobDir) : build_kv_wdict(layer_idx, blobDir);
    NSString *artifact_q = ane_train_bridge_compile_artifact_dir(blob_dir, layer_idx, bucket, qkv_input_mode, "q");
    NSString *artifact_kv = ane_train_bridge_compile_artifact_dir(blob_dir, layer_idx, bucket, qkv_input_mode, "kv");
    int q_cache_hit = 0;
    int kv_cache_hit = 0;
    int runtime_artifact_exported = 0;

    if (artifact_q) {
        g_ane_train_bridge.prog_q = orion_program_load_artifacts(
            mil_q.UTF8String,
            wdict_q,
            artifact_q.UTF8String,
            "qwen35_9b_lora_ane_train_q"
        );
        q_cache_hit = g_ane_train_bridge.prog_q ? 1 : 0;
    }
    if (!g_ane_train_bridge.prog_q) {
        g_ane_train_bridge.prog_q = orion_compile_mil(
            mil_q.UTF8String,
            wdict_q,
            "qwen35_9b_lora_ane_train_q"
        );
        if (g_ane_train_bridge.prog_q && artifact_q &&
            orion_program_export_artifacts(g_ane_train_bridge.prog_q, artifact_q.UTF8String)) {
            runtime_artifact_exported = 1;
        }
    }

    if (artifact_kv) {
        g_ane_train_bridge.prog_kv = orion_program_load_artifacts(
            mil_kv.UTF8String,
            wdict_kv,
            artifact_kv.UTF8String,
            "qwen35_9b_lora_ane_train_kv"
        );
        kv_cache_hit = g_ane_train_bridge.prog_kv ? 1 : 0;
    }
    if (!g_ane_train_bridge.prog_kv) {
        g_ane_train_bridge.prog_kv = orion_compile_mil(
            mil_kv.UTF8String,
            wdict_kv,
            "qwen35_9b_lora_ane_train_kv"
        );
        if (g_ane_train_bridge.prog_kv && artifact_kv &&
            orion_program_export_artifacts(g_ane_train_bridge.prog_kv, artifact_kv.UTF8String)) {
            runtime_artifact_exported = 1;
        }
    }
    if (!g_ane_train_bridge.prog_q || !g_ane_train_bridge.prog_kv) {
        ane_train_bridge_release();
        return 0;
    }

    g_ane_train_bridge.layer_idx = layer_idx;
    g_ane_train_bridge.bucket = bucket;
    g_ane_train_bridge.d_model = manifest->d_model;
    g_ane_train_bridge.q_dim = manifest->n_head * manifest->head_dim;
    g_ane_train_bridge.kv_dim = manifest->n_kv_head * manifest->head_dim;
    g_ane_train_bridge.qkv_input_mode = qkv_input_mode;
    g_ane_train_bridge.q_uses_cpu_rms = q_uses_cpu_rms;
    g_ane_train_bridge.kv_uses_cpu_rms = kv_uses_cpu_rms;
    g_ane_train_bridge.q_cache_hit = q_cache_hit;
    g_ane_train_bridge.kv_cache_hit = kv_cache_hit;
    g_ane_train_bridge.compile_cache_hit = q_cache_hit && kv_cache_hit;
    g_ane_train_bridge.runtime_artifact_exported = runtime_artifact_exported;
    if (g_ane_train_bridge.compile_cache_hit) {
        strlcpy(g_ane_train_bridge.compile_cache_source, "runtime_artifact", sizeof(g_ane_train_bridge.compile_cache_source));
    } else if (q_cache_hit || kv_cache_hit) {
        strlcpy(g_ane_train_bridge.compile_cache_source, "partial_runtime_artifact", sizeof(g_ane_train_bridge.compile_cache_source));
    } else {
        strlcpy(g_ane_train_bridge.compile_cache_source, "fresh_compile", sizeof(g_ane_train_bridge.compile_cache_source));
    }
    strlcpy(g_ane_train_bridge.blob_dir, blob_dir, sizeof(g_ane_train_bridge.blob_dir));
    return 1;
}

int orion_qwen9b_lora_ane_train_bridge_last_compile_cache_hit(void) {
    return g_ane_train_bridge.compile_cache_hit;
}

int orion_qwen9b_lora_ane_train_bridge_last_q_cache_hit(void) {
    return g_ane_train_bridge.q_cache_hit;
}

int orion_qwen9b_lora_ane_train_bridge_last_kv_cache_hit(void) {
    return g_ane_train_bridge.kv_cache_hit;
}

const char *orion_qwen9b_lora_ane_train_bridge_last_compile_cache_source(void) {
    return g_ane_train_bridge.compile_cache_source[0] ? g_ane_train_bridge.compile_cache_source : "unknown";
}

int orion_qwen9b_lora_frozen_prefix_hidden(const char *blob_dir,
                                           const OrionQwen35Manifest *manifest,
                                           int input_token,
                                           float *hidden_out) {
    const int seq_len = 1;
    const int d_model = manifest->d_model;
    const int d_ff = manifest->d_ff;
    const int n_head = manifest->n_head;
    const int n_kv_head = manifest->n_kv_head;
    const int head_dim = manifest->head_dim;
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_kv_head * head_dim;

    float *hidden = (float *)calloc((size_t)d_model, sizeof(float));
    float *normed = (float *)calloc((size_t)d_model, sizeof(float));
    float *mixer_out = (float *)calloc((size_t)d_model, sizeof(float));
    float *mlp_out = (float *)calloc((size_t)d_model, sizeof(float));
    float *scratch = (float *)calloc((size_t)d_model, sizeof(float));
    int ok = 0;

    if (!hidden || !normed || !mixer_out || !mlp_out || !scratch) goto cleanup;
    if (!load_embed_row(blob_dir, input_token, d_model, hidden)) goto cleanup;

    for (int layer_idx = 0; layer_idx < manifest->n_layer - 1; layer_idx++) {
        float *input_ln = load_layer_exact(blob_dir, layer_idx, "input_layernorm.bin", d_model);
        float *post_ln = load_layer_exact(blob_dir, layer_idx, "post_attention_layernorm.bin", d_model);
        float *gate_proj = load_layer_exact(blob_dir, layer_idx, "mlp_gate_proj.bin", d_ff * d_model);
        float *up_proj = load_layer_exact(blob_dir, layer_idx, "mlp_up_proj.bin", d_ff * d_model);
        float *down_proj = load_layer_exact(blob_dir, layer_idx, "mlp_down_proj.bin", d_model * d_ff);

        orion_qwen_cpu_rmsnorm(hidden, input_ln, d_model, 1e-6f, normed);
        memset(mixer_out, 0, (size_t)d_model * sizeof(float));

        char full_q_path[2048];
        snprintf(full_q_path, sizeof(full_q_path), "%s/layer%d/self_attn_q_proj.bin", blob_dir, layer_idx);
        if (file_exists(full_q_path)) {
            float *q_proj = load_layer_exact(blob_dir, layer_idx, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
            float *k_proj = load_layer_exact(blob_dir, layer_idx, "self_attn_k_proj.bin", kv_dim * d_model);
            float *v_proj = load_layer_exact(blob_dir, layer_idx, "self_attn_v_proj.bin", kv_dim * d_model);
            float *o_proj = load_layer_exact(blob_dir, layer_idx, "self_attn_o_proj.bin", d_model * q_dim);
            float *q_norm = load_layer_exact(blob_dir, layer_idx, "self_attn_q_norm.bin", head_dim);
            float *k_norm = load_layer_exact(blob_dir, layer_idx, "self_attn_k_norm.bin", head_dim);
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

            float *in_proj_qkv = load_layer_exact(blob_dir, layer_idx, "linear_attn_in_proj_qkv.bin", qkv_rows * d_model);
            float *in_proj_z = load_layer_exact(blob_dir, layer_idx, "linear_attn_in_proj_z.bin", value_dim * d_model);
            float *in_proj_a = load_layer_exact(blob_dir, layer_idx, "linear_attn_in_proj_a.bin", num_v_heads * d_model);
            float *in_proj_b = load_layer_exact(blob_dir, layer_idx, "linear_attn_in_proj_b.bin", num_v_heads * d_model);
            float *conv1d = load_layer_exact(blob_dir, layer_idx, "linear_attn_conv1d.bin", qkv_rows * conv_kernel);
            float *dt_bias = load_layer_exact(blob_dir, layer_idx, "linear_attn_dt_bias.bin", num_v_heads);
            float *a_log = load_layer_exact(blob_dir, layer_idx, "linear_attn_a_log.bin", num_v_heads);
            float *norm_weight = load_layer_exact(blob_dir, layer_idx, "linear_attn_norm.bin", head_v_dim);
            float *out_proj = load_layer_exact(blob_dir, layer_idx, "linear_attn_out_proj.bin", d_model * value_dim);
            orion_qwen_cpu_linear_attention_recurrent_prefill(
                normed, seq_len, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
                conv1d, dt_bias, a_log, norm_weight, out_proj,
                d_model, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel,
                mixer_out
            );
            free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
            free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
        }

        for (int i = 0; i < d_model; i++) hidden[i] += mixer_out[i];
        orion_qwen_cpu_rmsnorm(hidden, post_ln, d_model, 1e-6f, scratch);
        orion_qwen_cpu_swiglu_ffn(scratch, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out);
        for (int i = 0; i < d_model; i++) hidden[i] += mlp_out[i];

        free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
    }

    memcpy(hidden_out, hidden, (size_t)d_model * sizeof(float));
    ok = 1;

cleanup:
    free(hidden);
    free(normed);
    free(mixer_out);
    free(mlp_out);
    free(scratch);
    return ok;
}

void orion_qwen9b_lora_trainer_init(OrionQwen9BLoRATrainer *trainer,
                                    const OrionQwen35Manifest *manifest,
                                    int layer_idx,
                                    int rank,
                                    float alpha,
                                    float lr,
                                    unsigned int seed) {
    memset(trainer, 0, sizeof(*trainer));
    trainer->layer_idx = layer_idx;
    trainer->lr = lr;
    trainer->beta1 = 0.9f;
    trainer->beta2 = 0.999f;
    trainer->eps = 1e-8f;
    trainer->step = 0;
    trainer->ce_ctx = NULL;
    trainer->owns_ce_ctx = 0;
    trainer->cpu_ctx = NULL;
    trainer->owns_cpu_ctx = 0;

    int q_dim = manifest->n_head * manifest->head_dim;
    int kv_dim = manifest->n_kv_head * manifest->head_dim;
    orion_qwen_lora_matrix_init(&trainer->q_proj, manifest->d_model, q_dim * 2, rank, alpha, seed ^ 0x13579BDFu);
    orion_qwen_lora_matrix_init(&trainer->v_proj, manifest->d_model, kv_dim, rank, alpha, seed ^ 0x2468ACE0u);
}

void orion_qwen9b_lora_trainer_free(OrionQwen9BLoRATrainer *trainer) {
    if (!trainer) return;
    if (trainer->owns_ce_ctx && trainer->ce_ctx) {
        orion_qwen_streaming_ce_context_close(trainer->ce_ctx);
    }
    if (trainer->owns_cpu_ctx && trainer->cpu_ctx) {
        orion_qwen9b_cpu_train_context_close(trainer->cpu_ctx);
    }
    orion_qwen_lora_matrix_free(&trainer->q_proj);
    orion_qwen_lora_matrix_free(&trainer->v_proj);
    memset(trainer, 0, sizeof(*trainer));
}

int orion_qwen9b_lora_trainer_attach_ce_context(OrionQwen9BLoRATrainer *trainer,
                                                const char *embed_blob_path,
                                                const OrionQwen35Manifest *manifest) {
    if (!trainer || !embed_blob_path || !manifest) return 0;
    if (trainer->owns_ce_ctx && trainer->ce_ctx) {
        orion_qwen_streaming_ce_context_close(trainer->ce_ctx);
    }
    trainer->ce_ctx = orion_qwen_streaming_ce_context_open(embed_blob_path, manifest->d_model, manifest->vocab);
    trainer->owns_ce_ctx = trainer->ce_ctx ? 1 : 0;
    return trainer->ce_ctx != NULL;
}

int orion_qwen9b_lora_trainer_attach_cpu_train_context(OrionQwen9BLoRATrainer *trainer,
                                                       const char *blob_dir,
                                                       const OrionQwen35Manifest *manifest) {
    if (!trainer || !blob_dir || !manifest) return 0;
    if (trainer->owns_cpu_ctx && trainer->cpu_ctx) {
        orion_qwen9b_cpu_train_context_close(trainer->cpu_ctx);
    }
    trainer->cpu_ctx = orion_qwen9b_cpu_train_context_open(blob_dir, manifest, trainer->layer_idx);
    trainer->owns_cpu_ctx = trainer->cpu_ctx ? 1 : 0;
    return trainer->cpu_ctx != NULL;
}

void orion_qwen9b_lora_trainer_zero_grad(OrionQwen9BLoRATrainer *trainer) {
    orion_qwen_lora_matrix_zero_grad(&trainer->q_proj);
    orion_qwen_lora_matrix_zero_grad(&trainer->v_proj);
}

void orion_qwen9b_lora_trainer_scale_grad(OrionQwen9BLoRATrainer *trainer,
                                          float scale) {
    orion_qwen_lora_matrix_scale_grad(&trainer->q_proj, scale);
    orion_qwen_lora_matrix_scale_grad(&trainer->v_proj, scale);
}

void orion_qwen9b_lora_trainer_step(OrionQwen9BLoRATrainer *trainer) {
    trainer->step += 1;
    orion_cpu_adam_step(trainer->q_proj.a, trainer->q_proj.da, trainer->q_proj.ma, trainer->q_proj.va,
                        trainer->q_proj.rank * trainer->q_proj.in_dim, trainer->lr, trainer->beta1, trainer->beta2, trainer->eps, trainer->step);
    orion_cpu_adam_step(trainer->q_proj.b, trainer->q_proj.db, trainer->q_proj.mb, trainer->q_proj.vb,
                        trainer->q_proj.out_dim * trainer->q_proj.rank, trainer->lr, trainer->beta1, trainer->beta2, trainer->eps, trainer->step);
    orion_cpu_adam_step(trainer->v_proj.a, trainer->v_proj.da, trainer->v_proj.ma, trainer->v_proj.va,
                        trainer->v_proj.rank * trainer->v_proj.in_dim, trainer->lr, trainer->beta1, trainer->beta2, trainer->eps, trainer->step);
    orion_cpu_adam_step(trainer->v_proj.b, trainer->v_proj.db, trainer->v_proj.mb, trainer->v_proj.vb,
                        trainer->v_proj.out_dim * trainer->v_proj.rank, trainer->lr, trainer->beta1, trainer->beta2, trainer->eps, trainer->step);
    orion_qwen9b_lora_trainer_zero_grad(trainer);
}

static int ensure_dir(const char *path) {
    NSString *dir = [NSString stringWithUTF8String:path];
    NSError *error = nil;
    return [[NSFileManager defaultManager] createDirectoryAtPath:dir
                                     withIntermediateDirectories:YES
                                                      attributes:nil
                                                           error:&error];
}

int orion_qwen9b_lora_trainer_save(const OrionQwen9BLoRATrainer *trainer,
                                   const char *out_dir) {
    if (!ensure_dir(out_dir)) return 0;

    NSString *dir = [NSString stringWithUTF8String:out_dir];
    NSString *configPath = [dir stringByAppendingPathComponent:@"adapter_config.json"];
    NSString *statePath = [dir stringByAppendingPathComponent:@"trainer_state.json"];
    NSString *weightPath = [dir stringByAppendingPathComponent:@"adapter_weights.bin"];

    NSDictionary *config = @{
        @"model_name": @"CRPG_Q3.5",
        @"base_model": @"Qwen3.5-9B",
        @"targets": @[@"q_proj", @"v_proj"],
        @"rank": @(trainer->q_proj.rank),
        @"alpha": @(trainer->q_proj.alpha),
        @"dropout": @0.0,
        @"layer_idx": @(trainer->layer_idx)
    };
    NSDictionary *state = @{
        @"step": @(trainer->step),
        @"lr": @(trainer->lr),
        @"beta1": @(trainer->beta1),
        @"beta2": @(trainer->beta2),
        @"eps": @(trainer->eps)
    };
    NSData *configData = [NSJSONSerialization dataWithJSONObject:config options:NSJSONWritingPrettyPrinted error:nil];
    NSData *stateData = [NSJSONSerialization dataWithJSONObject:state options:NSJSONWritingPrettyPrinted error:nil];
    if (![configData writeToFile:configPath atomically:YES]) return 0;
    if (![stateData writeToFile:statePath atomically:YES]) return 0;

    FILE *f = fopen(weightPath.UTF8String, "wb");
    if (!f) return 0;
    fwrite(trainer->q_proj.a, sizeof(float), (size_t)trainer->q_proj.rank * trainer->q_proj.in_dim, f);
    fwrite(trainer->q_proj.b, sizeof(float), (size_t)trainer->q_proj.out_dim * trainer->q_proj.rank, f);
    fwrite(trainer->v_proj.a, sizeof(float), (size_t)trainer->v_proj.rank * trainer->v_proj.in_dim, f);
    fwrite(trainer->v_proj.b, sizeof(float), (size_t)trainer->v_proj.out_dim * trainer->v_proj.rank, f);
    fclose(f);
    return 1;
}

int orion_qwen9b_lora_trainer_load(OrionQwen9BLoRATrainer *trainer,
                                   const char *in_dir) {
    NSString *dir = [NSString stringWithUTF8String:in_dir];
    NSString *weightPath = [dir stringByAppendingPathComponent:@"adapter_weights.bin"];
    NSString *statePath = [dir stringByAppendingPathComponent:@"trainer_state.json"];
    FILE *f = fopen(weightPath.UTF8String, "rb");
    if (!f) return 0;
    size_t q_a = (size_t)trainer->q_proj.rank * trainer->q_proj.in_dim;
    size_t q_b = (size_t)trainer->q_proj.out_dim * trainer->q_proj.rank;
    size_t v_a = (size_t)trainer->v_proj.rank * trainer->v_proj.in_dim;
    size_t v_b = (size_t)trainer->v_proj.out_dim * trainer->v_proj.rank;
    int ok = fread(trainer->q_proj.a, sizeof(float), q_a, f) == q_a &&
             fread(trainer->q_proj.b, sizeof(float), q_b, f) == q_b &&
             fread(trainer->v_proj.a, sizeof(float), v_a, f) == v_a &&
             fread(trainer->v_proj.b, sizeof(float), v_b, f) == v_b;
    fclose(f);
    if (!ok) return 0;

    NSData *stateData = [NSData dataWithContentsOfFile:statePath];
    if (!stateData) return 0;
    NSDictionary *state = [NSJSONSerialization JSONObjectWithData:stateData options:0 error:nil];
    trainer->step = [state[@"step"] intValue];
    trainer->lr = [state[@"lr"] floatValue];
    trainer->beta1 = [state[@"beta1"] floatValue];
    trainer->beta2 = [state[@"beta2"] floatValue];
    trainer->eps = [state[@"eps"] floatValue];
    return 1;
}

int orion_qwen9b_lora_trainer_compare(const OrionQwen9BLoRATrainer *lhs,
                                      const OrionQwen9BLoRATrainer *rhs,
                                      float atol) {
    if (lhs->layer_idx != rhs->layer_idx || lhs->step != rhs->step) return 0;
    size_t q_a = (size_t)lhs->q_proj.rank * lhs->q_proj.in_dim;
    size_t q_b = (size_t)lhs->q_proj.out_dim * lhs->q_proj.rank;
    size_t v_a = (size_t)lhs->v_proj.rank * lhs->v_proj.in_dim;
    size_t v_b = (size_t)lhs->v_proj.out_dim * lhs->v_proj.rank;
    const float *pairs[] = { lhs->q_proj.a, rhs->q_proj.a, lhs->q_proj.b, rhs->q_proj.b,
                             lhs->v_proj.a, rhs->v_proj.a, lhs->v_proj.b, rhs->v_proj.b };
    const size_t counts[] = { q_a, q_b, v_a, v_b };
    for (int block = 0; block < 4; block++) {
        const float *a = pairs[block * 2 + 0];
        const float *b = pairs[block * 2 + 1];
        for (size_t i = 0; i < counts[block]; i++) {
            if (fabsf(a[i] - b[i]) > atol) return 0;
        }
    }
    return 1;
}

static int orion_qwen9b_lora_train_with_hidden_internal(const char *blob_dir,
                                                        const OrionQwen35Manifest *manifest,
                                                        OrionQwen9BLoRATrainer *trainer,
                                                        const float *hidden_in,
                                                        int target_token,
                                                        int apply_step,
                                                        OrionQwen9BLoRASmokeResult *out_result) {
    const int d_model = manifest->d_model;
    const int d_ff = manifest->d_ff;
    const int n_head = manifest->n_head;
    const int n_kv_head = manifest->n_kv_head;
    const int head_dim = manifest->head_dim;
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_kv_head * head_dim;
    OrionQwen9BCPUTrainContext *cpu_ctx = trainer->cpu_ctx;

    if (cpu_ctx) {
        orion_qwen9b_cpu_train_context_reset(cpu_ctx);
    }

    float *normed = cpu_ctx ? cpu_ctx->normed : (float *)calloc((size_t)d_model, sizeof(float));
    float *q_full = cpu_ctx ? cpu_ctx->q_full : (float *)calloc((size_t)(q_dim * 2), sizeof(float));
    float *v_raw = cpu_ctx ? cpu_ctx->v_raw : (float *)calloc((size_t)kv_dim, sizeof(float));
    float *attn_cat = cpu_ctx ? cpu_ctx->attn_cat : (float *)calloc((size_t)q_dim, sizeof(float));
    float *gated = cpu_ctx ? cpu_ctx->gated : (float *)calloc((size_t)q_dim, sizeof(float));
    float *mixer = cpu_ctx ? cpu_ctx->mixer : (float *)calloc((size_t)d_model, sizeof(float));
    float *hidden_mid = cpu_ctx ? cpu_ctx->hidden_mid : (float *)calloc((size_t)d_model, sizeof(float));
    float *post_norm = cpu_ctx ? cpu_ctx->post_norm : (float *)calloc((size_t)d_model, sizeof(float));
    float *mlp_out = cpu_ctx ? cpu_ctx->mlp_out : (float *)calloc((size_t)d_model, sizeof(float));
    float *hidden_out = cpu_ctx ? cpu_ctx->hidden_out : (float *)calloc((size_t)d_model, sizeof(float));
    float *final_norm = cpu_ctx ? cpu_ctx->final_norm : (float *)calloc((size_t)d_model, sizeof(float));
    float *last_hidden = cpu_ctx ? cpu_ctx->last_hidden : (float *)calloc((size_t)d_model, sizeof(float));
    float *d_last_hidden = cpu_ctx ? cpu_ctx->d_last_hidden : (float *)calloc((size_t)d_model, sizeof(float));
    float *d_hidden_out = cpu_ctx ? cpu_ctx->d_hidden_out : (float *)calloc((size_t)d_model, sizeof(float));
    float *d_post_norm = cpu_ctx ? cpu_ctx->d_post_norm : (float *)calloc((size_t)d_model, sizeof(float));
    float *d_hidden_mid = cpu_ctx ? cpu_ctx->d_hidden_mid : (float *)calloc((size_t)d_model, sizeof(float));
    float *d_mixer = cpu_ctx ? cpu_ctx->d_mixer : (float *)calloc((size_t)d_model, sizeof(float));
    float *d_gated = cpu_ctx ? cpu_ctx->d_gated : (float *)calloc((size_t)q_dim, sizeof(float));
    float *d_attn_cat = cpu_ctx ? cpu_ctx->d_attn_cat : (float *)calloc((size_t)q_dim, sizeof(float));
    float *d_gate_half = cpu_ctx ? cpu_ctx->d_gate_half : (float *)calloc((size_t)q_dim, sizeof(float));
    float *d_q_full = cpu_ctx ? cpu_ctx->d_q_full : (float *)calloc((size_t)(q_dim * 2), sizeof(float));
    float *d_v = cpu_ctx ? cpu_ctx->d_v : (float *)calloc((size_t)kv_dim, sizeof(float));
    float *d_normed_from_q = cpu_ctx ? cpu_ctx->d_normed_from_q : (float *)calloc((size_t)d_model, sizeof(float));
    float *d_normed_from_v = cpu_ctx ? cpu_ctx->d_normed_from_v : (float *)calloc((size_t)d_model, sizeof(float));
    float *d_normed = cpu_ctx ? cpu_ctx->d_normed : (float *)calloc((size_t)d_model, sizeof(float));
    float *d_hidden_mid_from_post = cpu_ctx ? cpu_ctx->d_hidden_mid_from_post : NULL;
    float *d_post_ln_weight_grad = cpu_ctx ? cpu_ctx->d_post_ln_weight_grad : NULL;

    float *input_ln = cpu_ctx ? cpu_ctx->input_ln : load_layer_exact(blob_dir, trainer->layer_idx, "input_layernorm.bin", d_model);
    float *post_ln = cpu_ctx ? cpu_ctx->post_ln : load_layer_exact(blob_dir, trainer->layer_idx, "post_attention_layernorm.bin", d_model);
    float *q_proj = cpu_ctx ? cpu_ctx->q_proj : load_layer_exact(blob_dir, trainer->layer_idx, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
    float *v_proj = cpu_ctx ? cpu_ctx->v_proj : load_layer_exact(blob_dir, trainer->layer_idx, "self_attn_v_proj.bin", kv_dim * d_model);
    float *o_proj = cpu_ctx ? cpu_ctx->o_proj : load_layer_exact(blob_dir, trainer->layer_idx, "self_attn_o_proj.bin", d_model * q_dim);
    float *gate_proj = cpu_ctx ? cpu_ctx->gate_proj : load_layer_exact(blob_dir, trainer->layer_idx, "mlp_gate_proj.bin", d_ff * d_model);
    float *up_proj = cpu_ctx ? cpu_ctx->up_proj : load_layer_exact(blob_dir, trainer->layer_idx, "mlp_up_proj.bin", d_ff * d_model);
    float *down_proj = cpu_ctx ? cpu_ctx->down_proj : load_layer_exact(blob_dir, trainer->layer_idx, "mlp_down_proj.bin", d_model * d_ff);

    char final_norm_path[2048], embed_path[2048];
    snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
    snprintf(embed_path, sizeof(embed_path), "%s/model/embed_tokens.bin", blob_dir);
    float *final_norm_weight = cpu_ctx ? cpu_ctx->final_norm_weight : orion_read_blob_f32_exact(final_norm_path, d_model);

    if (!cpu_ctx && (!normed || !q_full || !v_raw || !attn_cat || !gated || !mixer || !hidden_mid ||
        !post_norm || !mlp_out || !hidden_out || !final_norm || !last_hidden || !d_last_hidden ||
        !d_hidden_out || !d_post_norm || !d_hidden_mid || !d_mixer || !d_gated || !d_attn_cat ||
        !d_gate_half || !d_q_full || !d_v || !d_normed_from_q || !d_normed_from_v || !d_normed ||
        !input_ln || !post_ln || !q_proj || !v_proj || !o_proj || !gate_proj || !up_proj ||
        !down_proj || !final_norm_weight)) goto fail;

    orion_qwen_cpu_rmsnorm(hidden_in, input_ln, d_model, 1e-6f, normed);

    orion_qwen_lora_linear_forward(normed, q_proj, &trainer->q_proj, q_full);
    orion_qwen_lora_linear_forward(normed, v_proj, &trainer->v_proj, v_raw);

    expand_grouped_value(v_raw, n_head, n_kv_head, head_dim, attn_cat);
    for (int i = 0; i < q_dim; i++) {
        float gate = 1.0f / (1.0f + expf(-q_full[q_dim + i]));
        gated[i] = attn_cat[i] * gate;
    }
    cblas_sgemv(CblasRowMajor, CblasTrans,
                d_model, q_dim,
                1.0f, o_proj, q_dim, gated, 1,
                0.0f, mixer, 1);

    for (int i = 0; i < d_model; i++) hidden_mid[i] = hidden_in[i] + mixer[i];
    orion_qwen_cpu_rmsnorm(hidden_mid, post_ln, d_model, 1e-6f, post_norm);
    orion_qwen_cpu_swiglu_ffn(post_norm, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out);
    for (int i = 0; i < d_model; i++) hidden_out[i] = hidden_mid[i] + mlp_out[i];
    orion_qwen_cpu_rmsnorm(hidden_out, final_norm_weight, d_model, 1e-6f, last_hidden);

    float loss = trainer->ce_ctx
        ? orion_qwen_streaming_ce_tied_embedding_ctx(trainer->ce_ctx, last_hidden, target_token, d_last_hidden)
        : orion_qwen_cpu_streaming_ce_tied_embedding(embed_path, last_hidden, d_model, manifest->vocab, target_token, d_last_hidden);
    if (!isfinite(loss)) goto fail;

    orion_cpu_rmsnorm_bwd(d_hidden_out, final_norm, d_last_hidden, hidden_out, final_norm_weight, d_model, 1, 1e-6f);

    orion_qwen_cpu_swiglu_ffn_bwd(post_norm, gate_proj, up_proj, down_proj, d_model, d_ff, d_hidden_out, d_post_norm);
    if (!cpu_ctx) {
        d_hidden_mid_from_post = (float *)calloc((size_t)d_model, sizeof(float));
        d_post_ln_weight_grad = (float *)calloc((size_t)d_model, sizeof(float));
    }
    if (!d_hidden_mid_from_post || !d_post_ln_weight_grad) goto fail;
    orion_cpu_rmsnorm_bwd(d_hidden_mid_from_post, d_post_ln_weight_grad, d_post_norm, hidden_mid, post_ln, d_model, 1, 1e-6f);
    for (int i = 0; i < d_model; i++) d_hidden_mid[i] = d_hidden_out[i] + d_hidden_mid_from_post[i];

    memcpy(d_mixer, d_hidden_mid, (size_t)d_model * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                d_model, q_dim,
                1.0f, o_proj, q_dim, d_mixer, 1,
                0.0f, d_gated, 1);
    for (int i = 0; i < q_dim; i++) {
        float gate_pre = q_full[q_dim + i];
        float gate = 1.0f / (1.0f + expf(-gate_pre));
        d_attn_cat[i] = d_gated[i] * gate;
        d_gate_half[i] = d_gated[i] * attn_cat[i] * gate * (1.0f - gate);
        d_q_full[q_dim + i] = d_gate_half[i];
    }
    reduce_grouped_value_grad(d_attn_cat, n_head, n_kv_head, head_dim, d_v);

    orion_qwen_lora_linear_backward(normed, q_proj, &trainer->q_proj, d_q_full, d_normed_from_q);
    orion_qwen_lora_linear_backward(normed, v_proj, &trainer->v_proj, d_v, d_normed_from_v);
    for (int i = 0; i < d_model; i++) d_normed[i] = d_normed_from_q[i] + d_normed_from_v[i];

    {
        float *throwaway_weight_grad = cpu_ctx ? cpu_ctx->throwaway_weight_grad : (float *)calloc((size_t)d_model, sizeof(float));
        float *throwaway_dx = cpu_ctx ? cpu_ctx->throwaway_dx : (float *)calloc((size_t)d_model, sizeof(float));
        if (!throwaway_weight_grad || !throwaway_dx) {
            if (!cpu_ctx) {
                free(throwaway_weight_grad);
                free(throwaway_dx);
            }
            goto fail;
        }
        orion_cpu_rmsnorm_bwd(throwaway_dx, throwaway_weight_grad, d_normed, hidden_in, input_ln, d_model, 1, 1e-6f);
        if (!cpu_ctx) {
            free(throwaway_weight_grad);
            free(throwaway_dx);
        }
    }

    float q_grad_abs_sum = orion_qwen_lora_grad_abs_sum(&trainer->q_proj);
    float v_grad_abs_sum = orion_qwen_lora_grad_abs_sum(&trainer->v_proj);

    if (apply_step) orion_qwen9b_lora_trainer_step(trainer);

    memset(out_result, 0, sizeof(*out_result));
    out_result->loss = loss;
    out_result->q_grad_abs_sum = q_grad_abs_sum;
    out_result->v_grad_abs_sum = v_grad_abs_sum;
    out_result->q_param_abs_sum = orion_qwen_lora_abs_sum(trainer->q_proj.a, trainer->q_proj.rank * trainer->q_proj.in_dim) +
                                  orion_qwen_lora_abs_sum(trainer->q_proj.b, trainer->q_proj.out_dim * trainer->q_proj.rank);
    out_result->v_param_abs_sum = orion_qwen_lora_abs_sum(trainer->v_proj.a, trainer->v_proj.rank * trainer->v_proj.in_dim) +
                                  orion_qwen_lora_abs_sum(trainer->v_proj.b, trainer->v_proj.out_dim * trainer->v_proj.rank);
    out_result->predicted_token = -1;

    if (!cpu_ctx) {
        free(normed); free(q_full); free(v_raw); free(attn_cat); free(gated); free(mixer);
        free(hidden_mid); free(post_norm); free(mlp_out); free(hidden_out); free(final_norm); free(last_hidden);
        free(d_last_hidden); free(d_hidden_out); free(d_post_norm); free(d_hidden_mid); free(d_mixer);
        free(d_gated); free(d_attn_cat); free(d_gate_half); free(d_q_full); free(d_v); free(d_normed_from_q);
        free(d_normed_from_v); free(d_normed);
        free(d_hidden_mid_from_post); free(d_post_ln_weight_grad);
        free(input_ln); free(post_ln); free(q_proj); free(v_proj); free(o_proj); free(gate_proj); free(up_proj); free(down_proj);
        free(final_norm_weight);
    }
    return 1;

fail:
    if (!cpu_ctx) {
        free(normed); free(q_full); free(v_raw); free(attn_cat); free(gated); free(mixer);
        free(hidden_mid); free(post_norm); free(mlp_out); free(hidden_out); free(final_norm); free(last_hidden);
        free(d_last_hidden); free(d_hidden_out); free(d_post_norm); free(d_hidden_mid); free(d_mixer);
        free(d_gated); free(d_attn_cat); free(d_gate_half); free(d_q_full); free(d_v); free(d_normed_from_q);
        free(d_normed_from_v); free(d_normed);
        free(d_hidden_mid_from_post); free(d_post_ln_weight_grad);
        free(input_ln); free(post_ln); free(q_proj); free(v_proj); free(o_proj); free(gate_proj); free(up_proj); free(down_proj);
        free(final_norm_weight);
    }
    return 0;
}

static int orion_qwen9b_lora_train_single_internal(const char *blob_dir,
                                                   OrionQwen9BLoRATrainer *trainer,
                                                   int input_token,
                                                   int target_token,
                                                   int apply_step,
                                                   OrionQwen9BLoRASmokeResult *out_result) {
    OrionQwen35Manifest *manifest = orion_qwen35_manifest_load(blob_dir);
    if (!manifest) return 0;

    int ok = 0;
    float *hidden_in = (float *)calloc((size_t)manifest->d_model, sizeof(float));
    if (!hidden_in) goto done;
    if (!orion_qwen9b_lora_frozen_prefix_hidden(blob_dir, manifest, input_token, hidden_in)) goto done;
    ok = orion_qwen9b_lora_train_with_hidden_internal(blob_dir, manifest, trainer, hidden_in, target_token, apply_step, out_result);

done:
    free(hidden_in);
    orion_qwen35_manifest_free(manifest);
    return ok;
}

int orion_qwen9b_lora_train_smoke1(const char *blob_dir,
                                   OrionQwen9BLoRATrainer *trainer,
                                   int input_token,
                                   int target_token,
                                   OrionQwen9BLoRASmokeResult *out_result) {
    return orion_qwen9b_lora_train_single_internal(blob_dir, trainer, input_token, target_token, 1, out_result);
}

int orion_qwen9b_lora_train_accumulate1(const char *blob_dir,
                                        OrionQwen9BLoRATrainer *trainer,
                                        int input_token,
                                        int target_token,
                                        OrionQwen9BLoRASmokeResult *out_result) {
    return orion_qwen9b_lora_train_single_internal(blob_dir, trainer, input_token, target_token, 0, out_result);
}

int orion_qwen9b_lora_train_accumulate_hidden1(const char *blob_dir,
                                               const OrionQwen35Manifest *manifest,
                                               OrionQwen9BLoRATrainer *trainer,
                                               const float *hidden_in,
                                               int target_token,
                                               OrionQwen9BLoRASmokeResult *out_result) {
    if (!manifest || !hidden_in) return 0;
    return orion_qwen9b_lora_train_with_hidden_internal(blob_dir, manifest, trainer, hidden_in, target_token, 0, out_result);
}

int orion_qwen9b_lora_train_hidden_batch(const char *blob_dir,
                                         const OrionQwen35Manifest *manifest,
                                         OrionQwen9BLoRATrainer *trainer,
                                         const float *const *hidden_batch,
                                         const int *target_tokens,
                                         int item_count,
                                         OrionQwen9BLoRABatchResult *out_result) {
    if (!blob_dir || !manifest || !trainer || !hidden_batch || !target_tokens || !out_result || item_count <= 0) {
        return 0;
    }

    orion_qwen9b_lora_trainer_zero_grad(trainer);

    double loss_sum = 0.0;
    float loss_first = NAN;
    float loss_last = NAN;
    float loss_min = INFINITY;
    float loss_max = -INFINITY;
    OrionQwen9BLoRASmokeResult step_result;

    for (int idx = 0; idx < item_count; idx++) {
        const float *hidden_in = hidden_batch[idx];
        if (!hidden_in) return 0;
        if (!orion_qwen9b_lora_train_with_hidden_internal(blob_dir, manifest, trainer, hidden_in,
                                                          target_tokens[idx], 0, &step_result)) {
            return 0;
        }
        if (!isfinite(step_result.loss)) return 0;
        if (idx == 0) loss_first = step_result.loss;
        loss_last = step_result.loss;
        if (step_result.loss < loss_min) loss_min = step_result.loss;
        if (step_result.loss > loss_max) loss_max = step_result.loss;
        loss_sum += step_result.loss;
    }

    orion_qwen9b_lora_trainer_scale_grad(trainer, 1.0f / (float)item_count);
    double q_grad_last = orion_qwen_lora_grad_abs_sum(&trainer->q_proj);
    double v_grad_last = orion_qwen_lora_grad_abs_sum(&trainer->v_proj);
    orion_qwen9b_lora_trainer_step(trainer);
    double q_param_last = orion_qwen_lora_abs_sum(trainer->q_proj.a, trainer->q_proj.rank * trainer->q_proj.in_dim) +
                          orion_qwen_lora_abs_sum(trainer->q_proj.b, trainer->q_proj.out_dim * trainer->q_proj.rank);
    double v_param_last = orion_qwen_lora_abs_sum(trainer->v_proj.a, trainer->v_proj.rank * trainer->v_proj.in_dim) +
                          orion_qwen_lora_abs_sum(trainer->v_proj.b, trainer->v_proj.out_dim * trainer->v_proj.rank);

    memset(out_result, 0, sizeof(*out_result));
    out_result->items_completed = item_count;
    out_result->loss_sum = loss_sum;
    out_result->loss_first = loss_first;
    out_result->loss_last = loss_last;
    out_result->loss_min = loss_min;
    out_result->loss_max = loss_max;
    out_result->loss_avg = (float)(loss_sum / (double)item_count);
    out_result->q_grad_abs_sum_last = q_grad_last;
    out_result->v_grad_abs_sum_last = v_grad_last;
    out_result->q_param_abs_sum_last = q_param_last;
    out_result->v_param_abs_sum_last = v_param_last;
    out_result->predicted_token_last = step_result.predicted_token;
    return 1;
}

int orion_qwen9b_lora_train_smoke1_ane_qv_base(const char *blob_dir,
                                               OrionQwen9BLoRATrainer *trainer,
                                               int input_token,
                                               int target_token,
                                               OrionQwen9BLoRASmokeResult *out_result) {
    OrionQwen35Manifest *manifest = orion_qwen35_manifest_load(blob_dir);
    if (!manifest) return 0;

    const int bucket = 32;
    const int seq_len = 1;
    const int d_model = manifest->d_model;
    const int d_ff = manifest->d_ff;
    const int n_head = manifest->n_head;
    const int n_kv_head = manifest->n_kv_head;
    const int head_dim = manifest->head_dim;
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_kv_head * head_dim;
    const int q_proj_uses_cpu = use_cpu_q_proj_override();
    const int q_query_uses_cpu = use_cpu_q_query_override();
    const int q_gate_uses_cpu = use_cpu_q_gate_override();
    const int v_proj_uses_cpu = use_cpu_v_proj_override();
    int q_gate_cpu_channels[32] = {0};
    const int q_gate_cpu_channel_count = load_q_gate_cpu_channel_override(q_gate_cpu_channels, 32, q_dim);

    float *hidden_in = (float *)calloc((size_t)d_model, sizeof(float));
    float *normed = (float *)calloc((size_t)d_model, sizeof(float));
    float *base_q = (float *)calloc((size_t)(q_dim * 2), sizeof(float));
    float *base_v = (float *)calloc((size_t)kv_dim, sizeof(float));
    float *cpu_q = (q_proj_uses_cpu || q_query_uses_cpu || q_gate_uses_cpu || q_gate_cpu_channel_count > 0)
        ? (float *)calloc((size_t)(q_dim * 2), sizeof(float))
        : NULL;
    float *cpu_v = v_proj_uses_cpu ? (float *)calloc((size_t)kv_dim, sizeof(float)) : NULL;
    float *delta_q = (float *)calloc((size_t)(q_dim * 2), sizeof(float));
    float *delta_v = (float *)calloc((size_t)kv_dim, sizeof(float));
    float *q_full = (float *)calloc((size_t)(q_dim * 2), sizeof(float));
    float *v_raw = (float *)calloc((size_t)kv_dim, sizeof(float));
    float *attn_cat = (float *)calloc((size_t)q_dim, sizeof(float));
    float *gated = (float *)calloc((size_t)q_dim, sizeof(float));
    float *mixer = (float *)calloc((size_t)d_model, sizeof(float));
    float *hidden_mid = (float *)calloc((size_t)d_model, sizeof(float));
    float *post_norm = (float *)calloc((size_t)d_model, sizeof(float));
    float *mlp_out = (float *)calloc((size_t)d_model, sizeof(float));
    float *hidden_out = (float *)calloc((size_t)d_model, sizeof(float));
    float *final_norm = (float *)calloc((size_t)d_model, sizeof(float));
    float *last_hidden = (float *)calloc((size_t)d_model, sizeof(float));
    float *d_last_hidden = (float *)calloc((size_t)d_model, sizeof(float));
    float *d_hidden_out = (float *)calloc((size_t)d_model, sizeof(float));
    float *d_post_norm = (float *)calloc((size_t)d_model, sizeof(float));
    float *d_hidden_mid = (float *)calloc((size_t)d_model, sizeof(float));
    float *d_mixer = (float *)calloc((size_t)d_model, sizeof(float));
    float *d_gated = (float *)calloc((size_t)q_dim, sizeof(float));
    float *d_attn_cat = (float *)calloc((size_t)q_dim, sizeof(float));
    float *d_gate_half = (float *)calloc((size_t)q_dim, sizeof(float));
    float *d_q_full = (float *)calloc((size_t)(q_dim * 2), sizeof(float));
    float *d_v = (float *)calloc((size_t)kv_dim, sizeof(float));
    float *d_normed_from_q = (float *)calloc((size_t)d_model, sizeof(float));
    float *d_normed_from_v = (float *)calloc((size_t)d_model, sizeof(float));
    float *d_normed = (float *)calloc((size_t)d_model, sizeof(float));
    float *d_hidden_mid_from_post = NULL;
    float *d_post_ln_weight_grad = NULL;
    float *input_ln = load_layer_exact(blob_dir, trainer->layer_idx, "input_layernorm.bin", d_model);
    float *post_ln = load_layer_exact(blob_dir, trainer->layer_idx, "post_attention_layernorm.bin", d_model);
    float *q_proj = load_layer_exact(blob_dir, trainer->layer_idx, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
    float *v_proj = load_layer_exact(blob_dir, trainer->layer_idx, "self_attn_v_proj.bin", kv_dim * d_model);
    float *o_proj = load_layer_exact(blob_dir, trainer->layer_idx, "self_attn_o_proj.bin", d_model * q_dim);
    float *gate_proj = load_layer_exact(blob_dir, trainer->layer_idx, "mlp_gate_proj.bin", d_ff * d_model);
    float *up_proj = load_layer_exact(blob_dir, trainer->layer_idx, "mlp_up_proj.bin", d_ff * d_model);
    float *down_proj = load_layer_exact(blob_dir, trainer->layer_idx, "mlp_down_proj.bin", d_model * d_ff);
    char final_norm_path[2048], embed_path[2048];
    snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
    snprintf(embed_path, sizeof(embed_path), "%s/model/embed_tokens.bin", blob_dir);
    float *final_norm_weight = orion_read_blob_f32_exact(final_norm_path, d_model);
    IOSurfaceRef ioIn = NULL;
    IOSurfaceRef ioNormedIn = NULL;
    IOSurfaceRef ioQ = NULL;
    IOSurfaceRef ioK = NULL;
    IOSurfaceRef ioV = NULL;
    int ok = 0;
    float loss = NAN;

    if (!hidden_in || !normed || !base_q || !base_v || !delta_q || !delta_v || !q_full || !v_raw ||
        !attn_cat || !gated || !mixer || !hidden_mid || !post_norm || !mlp_out || !hidden_out ||
        !final_norm || !last_hidden || !d_last_hidden || !d_hidden_out || !d_post_norm ||
        !d_hidden_mid || !d_mixer || !d_gated || !d_attn_cat || !d_gate_half || !d_q_full ||
        !d_v || !d_normed_from_q || !d_normed_from_v || !d_normed || !input_ln || !post_ln ||
        !q_proj || !v_proj || !o_proj || !gate_proj || !up_proj || !down_proj || !final_norm_weight ||
        ((q_proj_uses_cpu || q_query_uses_cpu || q_gate_uses_cpu || q_gate_cpu_channel_count > 0) && !cpu_q) ||
        (v_proj_uses_cpu && !cpu_v)) goto fail;

    if (!orion_qwen9b_lora_frozen_prefix_hidden(blob_dir, manifest, input_token, hidden_in)) goto fail;
    orion_qwen_cpu_rmsnorm(hidden_in, input_ln, d_model, 1e-6f, normed);
    if (!ane_train_bridge_ensure(blob_dir, manifest, trainer->layer_idx, bucket)) goto fail;

    ioIn = make_cpu_seq_input_surface(hidden_in, seq_len, bucket, d_model);
    ioNormedIn = make_cpu_seq_input_surface(normed, seq_len, bucket, d_model);
    ioQ = make_f32_surface((q_dim * 2) * bucket, 0.0f);
    ioK = make_f32_surface(kv_dim * bucket, 0.0f);
    ioV = make_f32_surface(kv_dim * bucket, 0.0f);
    if (!ioIn || !ioNormedIn || !ioQ || !ioK || !ioV) goto fail;

    IOSurfaceRef q_ins[] = { g_ane_train_bridge.q_uses_cpu_rms ? ioNormedIn : ioIn };
    IOSurfaceRef kv_ins[] = { g_ane_train_bridge.kv_uses_cpu_rms ? ioNormedIn : ioIn };
    IOSurfaceRef outsQ[] = { ioQ };
    IOSurfaceRef outsKV[] = { ioK, ioV };
    if (!orion_eval(g_ane_train_bridge.prog_q, q_ins, 1, outsQ, 1)) goto fail;
    if (!orion_eval(g_ane_train_bridge.prog_kv, kv_ins, 1, outsKV, 2)) goto fail;

    read_ane_surface_prefix(ioQ, q_dim * 2, seq_len, bucket, base_q);
    read_ane_surface_prefix(ioV, kv_dim, seq_len, bucket, base_v);
    if (cpu_q) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    q_dim * 2, d_model,
                    1.0f, q_proj, d_model, normed, 1,
                    0.0f, cpu_q, 1);
        if (q_proj_uses_cpu || q_query_uses_cpu) {
            memcpy(base_q, cpu_q, (size_t)q_dim * sizeof(float));
        }
        if (q_proj_uses_cpu || q_gate_uses_cpu) {
            memcpy(base_q + q_dim, cpu_q + q_dim, (size_t)q_dim * sizeof(float));
        } else {
            for (int i = 0; i < q_gate_cpu_channel_count; i++) {
                int channel = q_gate_cpu_channels[i];
                base_q[q_dim + channel] = cpu_q[q_dim + channel];
            }
        }
    }
    if (cpu_v) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    kv_dim, d_model,
                    1.0f, v_proj, d_model, normed, 1,
                    0.0f, cpu_v, 1);
        memcpy(base_v, cpu_v, (size_t)kv_dim * sizeof(float));
    }
    orion_qwen_lora_linear_delta_forward(normed, &trainer->q_proj, delta_q);
    orion_qwen_lora_linear_delta_forward(normed, &trainer->v_proj, delta_v);
    for (int i = 0; i < q_dim * 2; i++) q_full[i] = base_q[i] + delta_q[i];
    for (int i = 0; i < kv_dim; i++) v_raw[i] = base_v[i] + delta_v[i];

    expand_grouped_value(v_raw, n_head, n_kv_head, head_dim, attn_cat);
    for (int i = 0; i < q_dim; i++) {
        float gate = 1.0f / (1.0f + expf(-q_full[q_dim + i]));
        gated[i] = attn_cat[i] * gate;
    }
    cblas_sgemv(CblasRowMajor, CblasTrans,
                d_model, q_dim,
                1.0f, o_proj, q_dim, gated, 1,
                0.0f, mixer, 1);

    for (int i = 0; i < d_model; i++) hidden_mid[i] = hidden_in[i] + mixer[i];
    orion_qwen_cpu_rmsnorm(hidden_mid, post_ln, d_model, 1e-6f, post_norm);
    orion_qwen_cpu_swiglu_ffn(post_norm, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out);
    for (int i = 0; i < d_model; i++) hidden_out[i] = hidden_mid[i] + mlp_out[i];
    orion_qwen_cpu_rmsnorm(hidden_out, final_norm_weight, d_model, 1e-6f, last_hidden);

    loss = trainer->ce_ctx
        ? orion_qwen_streaming_ce_tied_embedding_ctx(trainer->ce_ctx, last_hidden, target_token, d_last_hidden)
        : orion_qwen_cpu_streaming_ce_tied_embedding(embed_path, last_hidden, d_model, manifest->vocab, target_token, d_last_hidden);
    if (!isfinite(loss)) goto fail;

    orion_cpu_rmsnorm_bwd(d_hidden_out, final_norm, d_last_hidden, hidden_out, final_norm_weight, d_model, 1, 1e-6f);
    orion_qwen_cpu_swiglu_ffn_bwd(post_norm, gate_proj, up_proj, down_proj, d_model, d_ff, d_hidden_out, d_post_norm);
    d_hidden_mid_from_post = (float *)calloc((size_t)d_model, sizeof(float));
    d_post_ln_weight_grad = (float *)calloc((size_t)d_model, sizeof(float));
    orion_cpu_rmsnorm_bwd(d_hidden_mid_from_post, d_post_ln_weight_grad, d_post_norm, hidden_mid, post_ln, d_model, 1, 1e-6f);
    for (int i = 0; i < d_model; i++) d_hidden_mid[i] = d_hidden_out[i] + d_hidden_mid_from_post[i];

    memcpy(d_mixer, d_hidden_mid, (size_t)d_model * sizeof(float));
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                d_model, q_dim,
                1.0f, o_proj, q_dim, d_mixer, 1,
                0.0f, d_gated, 1);
    for (int i = 0; i < q_dim; i++) {
        float gate_pre = q_full[q_dim + i];
        float gate = 1.0f / (1.0f + expf(-gate_pre));
        d_attn_cat[i] = d_gated[i] * gate;
        d_gate_half[i] = d_gated[i] * attn_cat[i] * gate * (1.0f - gate);
        d_q_full[q_dim + i] = d_gate_half[i];
    }
    reduce_grouped_value_grad(d_attn_cat, n_head, n_kv_head, head_dim, d_v);

    orion_qwen_lora_linear_backward(normed, q_proj, &trainer->q_proj, d_q_full, d_normed_from_q);
    orion_qwen_lora_linear_backward(normed, v_proj, &trainer->v_proj, d_v, d_normed_from_v);
    for (int i = 0; i < d_model; i++) d_normed[i] = d_normed_from_q[i] + d_normed_from_v[i];

    {
        float *throwaway_weight_grad = (float *)calloc((size_t)d_model, sizeof(float));
        float *throwaway_dx = (float *)calloc((size_t)d_model, sizeof(float));
        orion_cpu_rmsnorm_bwd(throwaway_dx, throwaway_weight_grad, d_normed, hidden_in, input_ln, d_model, 1, 1e-6f);
        free(throwaway_weight_grad);
        free(throwaway_dx);
    }

    memset(out_result, 0, sizeof(*out_result));
    out_result->loss = loss;
    out_result->q_grad_abs_sum = orion_qwen_lora_grad_abs_sum(&trainer->q_proj);
    out_result->v_grad_abs_sum = orion_qwen_lora_grad_abs_sum(&trainer->v_proj);
    orion_qwen9b_lora_trainer_step(trainer);
    out_result->q_param_abs_sum = orion_qwen_lora_abs_sum(trainer->q_proj.a, trainer->q_proj.rank * trainer->q_proj.in_dim) +
                                  orion_qwen_lora_abs_sum(trainer->q_proj.b, trainer->q_proj.out_dim * trainer->q_proj.rank);
    out_result->v_param_abs_sum = orion_qwen_lora_abs_sum(trainer->v_proj.a, trainer->v_proj.rank * trainer->v_proj.in_dim) +
                                  orion_qwen_lora_abs_sum(trainer->v_proj.b, trainer->v_proj.out_dim * trainer->v_proj.rank);
    out_result->predicted_token = -1;
    ok = 1;

fail:
    if (ioIn) CFRelease(ioIn);
    if (ioNormedIn) CFRelease(ioNormedIn);
    if (ioQ) CFRelease(ioQ);
    if (ioK) CFRelease(ioK);
    if (ioV) CFRelease(ioV);
    free(hidden_in); free(normed); free(base_q); free(base_v); free(cpu_q); free(cpu_v); free(delta_q); free(delta_v);
    free(q_full); free(v_raw); free(attn_cat); free(gated); free(mixer); free(hidden_mid);
    free(post_norm); free(mlp_out); free(hidden_out); free(final_norm); free(last_hidden);
    free(d_last_hidden); free(d_hidden_out); free(d_post_norm); free(d_hidden_mid); free(d_mixer);
    free(d_gated); free(d_attn_cat); free(d_gate_half); free(d_q_full); free(d_v); free(d_normed_from_q);
    free(d_normed_from_v); free(d_normed); free(d_hidden_mid_from_post); free(d_post_ln_weight_grad);
    free(input_ln); free(post_ln); free(q_proj); free(v_proj); free(o_proj); free(gate_proj); free(up_proj); free(down_proj);
    free(final_norm_weight);
    orion_qwen35_manifest_free(manifest);
    return ok;
}
