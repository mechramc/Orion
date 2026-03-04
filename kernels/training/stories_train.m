#import "stories_train.h"
#import "../../compiler/kernel_adapter.h"
#include "../../compiler/frontends/stories_train.h"
#import "../../core/kernel.h"
#import <math.h>
#import <stdlib.h>
#import <string.h>
#import <dispatch/dispatch.h>

// T072: Single training step for Stories110M on ANE.
// Forward (ANE) → Loss (CPU) → Backward (ANE) → dW (CPU)

#pragma mark - Internal Helpers

static float* alloc_zeros(int count) {
    return (float *)calloc(count, sizeof(float));
}

// Read fp16 IOSurface as fp32 into buffer. Layout: ANE [1,C,1,S] → CPU [S,C] (transposed).
// ANE stores channel-first: data[c*S + s]. CPU wants row-major [seq, dim]: data[s*C + c].
// Clamps NaN/Inf to fp16 range on readback to prevent overflow cascade through layers.
static void io_read_transpose(IOSurfaceRef surf, float* out, int channels, int seq) {
    int count = channels * seq;
    float *tmp = (float *)malloc(count * sizeof(float));
    orion_tensor_read_f32(surf, tmp, count);
    // ANE layout: [C, S] (channel-first) → CPU layout: [S, C] (row-major)
    for (int c = 0; c < channels; c++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float v = tmp[c * seq + s2];
            if (isnan(v) || v > 65504.0f) v = 65504.0f;
            else if (v < -65504.0f) v = -65504.0f;
            out[s2 * channels + c] = v;
        }
    }
    free(tmp);
}

// Write fp32 CPU [S,C] data as fp16 to ANE IOSurface [1,C,1,S] (transposed).
// Clamps values to fp16 range [-65504, 65504] to prevent Inf from entering ANE.
static void io_write_transpose(IOSurfaceRef surf, const float* in, int channels, int seq) {
    int count = channels * seq;
    float *tmp = (float *)malloc(count * sizeof(float));
    // CPU layout: [S, C] → ANE layout: [C, S] with fp16 clamping
    for (int s2 = 0; s2 < seq; s2++) {
        for (int c = 0; c < channels; c++) {
            float v = in[s2 * channels + c];
            if (v > 65504.0f) v = 65504.0f;
            else if (v < -65504.0f) v = -65504.0f;
            tmp[c * seq + s2] = v;
        }
    }
    orion_tensor_write_f32(surf, tmp, count);
    free(tmp);
}

// Write fp32 weights as fp16 BLOBFILE to disk.
static bool save_blob_f32(const char* path, const float* data, int n_elements) {
    int data_bytes = n_elements * (int)sizeof(_Float16);
    int total = 128 + data_bytes;

    uint8_t *buf = (uint8_t *)calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = data_bytes;
    *(uint32_t *)(buf + 80) = 128;

    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < n_elements; i++) {
        float v = data[i];
        if (isnan(v)) v = 0.0f;
        else if (v > 65504.0f) v = 65504.0f;
        else if (v < -65504.0f) v = -65504.0f;
        fp16[i] = (_Float16)v;
    }

    NSString *dir = [@(path) stringByDeletingLastPathComponent];
    [[NSFileManager defaultManager] createDirectoryAtPath:dir
                              withIntermediateDirectories:YES attributes:nil error:nil];
    NSData *nsdata = [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
    return [nsdata writeToFile:@(path) atomically:YES];
}

// Save all layer weights (forward + transposed) to disk for recompilation.
static void save_layer_weights(const OrionLayerWeights* w, int layer_idx,
                                const OrionModelConfig* cfg, const char* base) {
    int d = cfg->d_model, h = cfg->hidden_dim;
    char path[512];

    // Forward weights
    snprintf(path, sizeof(path), "%s/layer%d/rms1.bin", base, layer_idx);
    save_blob_f32(path, w->rms_att, d);
    snprintf(path, sizeof(path), "%s/layer%d/wq.bin", base, layer_idx);
    save_blob_f32(path, w->wq, d*d);
    snprintf(path, sizeof(path), "%s/layer%d/wk.bin", base, layer_idx);
    save_blob_f32(path, w->wk, d*d);
    snprintf(path, sizeof(path), "%s/layer%d/wv.bin", base, layer_idx);
    save_blob_f32(path, w->wv, d*d);
    snprintf(path, sizeof(path), "%s/layer%d/wo.bin", base, layer_idx);
    save_blob_f32(path, w->wo, d*d);
    snprintf(path, sizeof(path), "%s/layer%d/rms2.bin", base, layer_idx);
    save_blob_f32(path, w->rms_ffn, d);
    snprintf(path, sizeof(path), "%s/layer%d/w1.bin", base, layer_idx);
    save_blob_f32(path, w->w1, h*d);
    snprintf(path, sizeof(path), "%s/layer%d/w3.bin", base, layer_idx);
    save_blob_f32(path, w->w3, h*d);
    snprintf(path, sizeof(path), "%s/layer%d/w2.bin", base, layer_idx);
    save_blob_f32(path, w->w2, d*h);

    // Transposed weights for backward kernels
    // For simplicity, we write the same weights transposed in-memory.
    // The MIL kernels reference e.g. w2t.bin which is W2 transposed.
    float *tmp = (float *)malloc(h * d * sizeof(float));

    // W2 is [d, h], W2^T is [h, d]
    for (int r = 0; r < d; r++)
        for (int c = 0; c < h; c++)
            tmp[c * d + r] = w->w2[r * h + c];
    snprintf(path, sizeof(path), "%s/layer%d/w2t.bin", base, layer_idx);
    save_blob_f32(path, tmp, h * d);

    // W1 is [h, d], W1^T is [d, h]
    for (int r = 0; r < h; r++)
        for (int c = 0; c < d; c++)
            tmp[c * h + r] = w->w1[r * d + c];
    snprintf(path, sizeof(path), "%s/layer%d/w1t.bin", base, layer_idx);
    save_blob_f32(path, tmp, d * h);

    // W3 is [h, d], W3^T is [d, h]
    for (int r = 0; r < h; r++)
        for (int c = 0; c < d; c++)
            tmp[c * h + r] = w->w3[r * d + c];
    snprintf(path, sizeof(path), "%s/layer%d/w3t.bin", base, layer_idx);
    save_blob_f32(path, tmp, d * h);

    free(tmp);
    tmp = (float *)malloc(d * d * sizeof(float));

    // Wo^T
    for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++)
            tmp[c * d + r] = w->wo[r * d + c];
    snprintf(path, sizeof(path), "%s/layer%d/wot.bin", base, layer_idx);
    save_blob_f32(path, tmp, d * d);

    // Wq^T
    for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++)
            tmp[c * d + r] = w->wq[r * d + c];
    snprintf(path, sizeof(path), "%s/layer%d/wqt.bin", base, layer_idx);
    save_blob_f32(path, tmp, d * d);

    // Wk^T
    for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++)
            tmp[c * d + r] = w->wk[r * d + c];
    snprintf(path, sizeof(path), "%s/layer%d/wkt.bin", base, layer_idx);
    save_blob_f32(path, tmp, d * d);

    // Wv^T
    for (int r = 0; r < d; r++)
        for (int c = 0; c < d; c++)
            tmp[c * d + r] = w->wv[r * d + c];
    snprintf(path, sizeof(path), "%s/layer%d/wvt.bin", base, layer_idx);
    save_blob_f32(path, tmp, d * d);

    free(tmp);
}

// Build weight dict for a layer's kernel from disk blobs.
// Each entry: key="@model_path/rel_path" → {offset:0, data:NSData}
static NSDictionary* load_weight_dict(NSString* base, NSArray<NSString*>* rel_paths) {
    NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
    NSFileManager *fm = [NSFileManager defaultManager];
    for (NSString *rel in rel_paths) {
        NSString *full = [base stringByAppendingPathComponent:rel];
        NSData *data = [fm contentsAtPath:full];
        if (!data) {
            NSLog(@"WARNING: Missing weight blob: %@", full);
            continue;
        }
        NSString *key = [NSString stringWithFormat:@"@model_path/%@", rel];
        wdict[key] = @{@"offset": @0, @"data": data};
    }
    return wdict;
}

// Load fp16 blob file into fp32 array. Returns number of elements.
static int load_blob_f32(const char* path, float* out, int max_elements) {
    NSData *data = [[NSFileManager defaultManager]
        contentsAtPath:@(path)];
    if (!data) return 0;

    // BLOBFILE: 128 byte header, then fp16 data
    const uint8_t *bytes = (const uint8_t *)data.bytes;
    if (data.length < 128) return 0;

    int data_bytes = *(const uint32_t *)(bytes + 72);
    int n_elements = data_bytes / sizeof(_Float16);
    if (n_elements > max_elements) n_elements = max_elements;

    int data_offset = *(const uint32_t *)(bytes + 80);
    const _Float16 *fp16 = (const _Float16 *)(bytes + data_offset);
    for (int i = 0; i < n_elements; i++) {
        float v = (float)fp16[i];
        // Sanitize: replace NaN/Inf from corrupted BLOBFILEs with 0
        if (isnan(v) || isinf(v)) v = 0.0f;
        out[i] = v;
    }
    return n_elements;
}

#pragma mark - OrionKernel Adapters for Training

// Compiler MIL gen adapters — training kernels ignore bucket
static NSString* compiler_fwd_attn_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    (void)bucket;
    return orion_kernel_adapter_generate_mil_2arg(orion_frontend_fwd_attn, layer_idx, cfg);
}

static NSString* compiler_fwd_ffn_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    (void)bucket;
    return orion_kernel_adapter_generate_mil_2arg(orion_frontend_fwd_ffn, layer_idx, cfg);
}

static NSString* compiler_ffn_bwd_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    (void)bucket;
    return orion_kernel_adapter_generate_mil_2arg(orion_frontend_ffn_bwd, layer_idx, cfg);
}

static NSString* compiler_sdpa_bwd1_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    (void)bucket;
    return orion_kernel_adapter_generate_mil_2arg(orion_frontend_sdpa_bwd1, layer_idx, cfg);
}

static NSString* compiler_sdpa_bwd2_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    (void)bucket;
    return orion_kernel_adapter_generate_mil_2arg(orion_frontend_sdpa_bwd2, layer_idx, cfg);
}

static NSString* compiler_qkv_bwd_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    (void)bucket;
    return orion_kernel_adapter_generate_mil_2arg(orion_frontend_qkv_bwd, layer_idx, cfg);
}

// Weight dict adapters
static NSDictionary* wdict_fwd_attn_adapter(int layer_idx, int bucket, const char* blob_dir) {
    (void)bucket;
    NSString *base = @(blob_dir);
    NSString *L = [NSString stringWithFormat:@"layer%d", layer_idx];
    return load_weight_dict(base, @[
        [L stringByAppendingPathComponent:@"rms1.bin"],
        [L stringByAppendingPathComponent:@"wq.bin"],
        [L stringByAppendingPathComponent:@"wk.bin"],
        [L stringByAppendingPathComponent:@"wv.bin"],
        [L stringByAppendingPathComponent:@"wo.bin"],
        [NSString stringWithFormat:@"masks/causal_%d.bin", kStories110M.max_seq]
    ]);
}

static NSDictionary* wdict_fwd_ffn_adapter(int layer_idx, int bucket, const char* blob_dir) {
    (void)bucket;
    NSString *base = @(blob_dir);
    NSString *L = [NSString stringWithFormat:@"layer%d", layer_idx];
    return load_weight_dict(base, @[
        [L stringByAppendingPathComponent:@"rms2.bin"],
        [L stringByAppendingPathComponent:@"w1.bin"],
        [L stringByAppendingPathComponent:@"w3.bin"],
        [L stringByAppendingPathComponent:@"w2.bin"]
    ]);
}

static NSDictionary* wdict_ffn_bwd_adapter(int layer_idx, int bucket, const char* blob_dir) {
    (void)bucket;
    NSString *base = @(blob_dir);
    NSString *L = [NSString stringWithFormat:@"layer%d", layer_idx];
    return load_weight_dict(base, @[
        [L stringByAppendingPathComponent:@"w2t.bin"],
        [L stringByAppendingPathComponent:@"w1t.bin"],
        [L stringByAppendingPathComponent:@"w3t.bin"]
    ]);
}

static NSDictionary* wdict_sdpa_bwd1_adapter(int layer_idx, int bucket, const char* blob_dir) {
    (void)bucket;
    NSString *base = @(blob_dir);
    NSString *L = [NSString stringWithFormat:@"layer%d", layer_idx];
    return load_weight_dict(base, @[
        [L stringByAppendingPathComponent:@"wot.bin"],
        [NSString stringWithFormat:@"masks/causal_%d.bin", kStories110M.max_seq]
    ]);
}

static NSDictionary* wdict_qkv_bwd_adapter(int layer_idx, int bucket, const char* blob_dir) {
    (void)bucket;
    NSString *base = @(blob_dir);
    NSString *L = [NSString stringWithFormat:@"layer%d", layer_idx];
    return load_weight_dict(base, @[
        [L stringByAppendingPathComponent:@"wqt.bin"],
        [L stringByAppendingPathComponent:@"wkt.bin"],
        [L stringByAppendingPathComponent:@"wvt.bin"]
    ]);
}

#pragma mark - Compile Layer Kernels

static bool compile_layer(OrionLayerKernels* kern, int layer_idx,
                           const OrionModelConfig* cfg, NSString* weight_dir) {
    // Training compiles directly — no cache (weights change every recompile)
    int s = cfg->max_seq;
    NSString *base = weight_dir;

    NSString *mil;
    NSDictionary *wdict;

    mil = compiler_fwd_attn_adapter(layer_idx, s, cfg);
    wdict = wdict_fwd_attn_adapter(layer_idx, s, base.UTF8String);
    kern->fwd_attn = orion_compile_mil(mil.UTF8String, wdict, "fwdAttn");
    if (!kern->fwd_attn) return false;

    mil = compiler_fwd_ffn_adapter(layer_idx, s, cfg);
    wdict = wdict_fwd_ffn_adapter(layer_idx, s, base.UTF8String);
    kern->fwd_ffn = orion_compile_mil(mil.UTF8String, wdict, "fwdFFN");
    if (!kern->fwd_ffn) return false;

    mil = compiler_ffn_bwd_adapter(layer_idx, s, cfg);
    wdict = wdict_ffn_bwd_adapter(layer_idx, s, base.UTF8String);
    kern->ffn_bwd = orion_compile_mil(mil.UTF8String, wdict, "ffnBwd");
    if (!kern->ffn_bwd) return false;

    mil = compiler_sdpa_bwd1_adapter(layer_idx, s, cfg);
    wdict = wdict_sdpa_bwd1_adapter(layer_idx, s, base.UTF8String);
    kern->sdpa_bwd1 = orion_compile_mil(mil.UTF8String, wdict, "sdpaBwd1");
    if (!kern->sdpa_bwd1) return false;

    mil = compiler_sdpa_bwd2_adapter(layer_idx, s, cfg);
    kern->sdpa_bwd2 = orion_compile_mil(mil.UTF8String, @{}, "sdpaBwd2");
    if (!kern->sdpa_bwd2) return false;

    mil = compiler_qkv_bwd_adapter(layer_idx, s, cfg);
    wdict = wdict_qkv_bwd_adapter(layer_idx, s, base.UTF8String);
    kern->qkv_bwd = orion_compile_mil(mil.UTF8String, wdict, "qkvBwd");
    if (!kern->qkv_bwd) return false;

    return true;
}

#pragma mark - Allocate Per-Layer IO

static void alloc_layer_io(OrionLayerIO* io, const OrionModelConfig* cfg) {
    int d = cfg->d_model, h = cfg->hidden_dim, s = cfg->max_seq;
    int nh = cfg->n_head;
    int sc_ch = nh * s;

    // ANE CONSTRAINT: All output IOSurfaces for a multi-output program must be
    // the same allocation size. Use max channel count per kernel for outputs.

    // Forward inputs
    io->fwd_attn_in = orion_tensor_create(d, s);
    io->fwd_ffn_in  = orion_tensor_create(d, s);

    // fwdAttn: 6 outputs, all [d, s] — naturally uniform
    for (int i = 0; i < 6; i++) io->fwd_attn_out[i] = orion_tensor_create(d, s);

    // fwdFFN: 5 outputs — max(d, h) = h for uniform sizing
    for (int i = 0; i < 5; i++) io->fwd_ffn_out[i] = orion_tensor_create(h, s);

    // Backward inputs (concatenated for slice_by_index in kernel)
    io->ffn_bwd_in   = orion_tensor_create(d + 2*h, s);
    io->sdpa_bwd1_in = orion_tensor_create(4*d, s);
    io->sdpa_bwd2_in = orion_tensor_create(2*sc_ch + 2*d, s);
    io->qkv_bwd_in   = orion_tensor_create(3*d, s);

    // ffnBwd: 3 outputs — max(d, h) = h for uniform sizing
    for (int i = 0; i < 3; i++) io->ffn_bwd_out[i] = orion_tensor_create(h, s);

    // sdpaBwd1: 3 outputs — max(d, sc_ch) = sc_ch for uniform sizing
    int sdpa1_max = (sc_ch > d) ? sc_ch : d;
    for (int i = 0; i < 3; i++) io->sdpa_bwd1_out[i] = orion_tensor_create(sdpa1_max, s);

    // sdpaBwd2: 2 outputs — both [d, s], naturally uniform
    io->sdpa_bwd2_out[0] = orion_tensor_create(d, s);
    io->sdpa_bwd2_out[1] = orion_tensor_create(d, s);

    io->qkv_bwd_out = orion_tensor_create(d, s);
}

static void free_layer_io(OrionLayerIO* io) {
    if (io->fwd_attn_in) CFRelease(io->fwd_attn_in);
    if (io->fwd_ffn_in) CFRelease(io->fwd_ffn_in);
    for (int i = 0; i < 6; i++) if (io->fwd_attn_out[i]) CFRelease(io->fwd_attn_out[i]);
    for (int i = 0; i < 5; i++) if (io->fwd_ffn_out[i]) CFRelease(io->fwd_ffn_out[i]);
    if (io->ffn_bwd_in) CFRelease(io->ffn_bwd_in);
    if (io->sdpa_bwd1_in) CFRelease(io->sdpa_bwd1_in);
    if (io->sdpa_bwd2_in) CFRelease(io->sdpa_bwd2_in);
    if (io->qkv_bwd_in) CFRelease(io->qkv_bwd_in);
    for (int i = 0; i < 3; i++) if (io->ffn_bwd_out[i]) CFRelease(io->ffn_bwd_out[i]);
    for (int i = 0; i < 3; i++) if (io->sdpa_bwd1_out[i]) CFRelease(io->sdpa_bwd1_out[i]);
    for (int i = 0; i < 2; i++) if (io->sdpa_bwd2_out[i]) CFRelease(io->sdpa_bwd2_out[i]);
    if (io->qkv_bwd_out) CFRelease(io->qkv_bwd_out);
}

#pragma mark - Alloc / Free Weights, Grads, Adam

static void alloc_layer_grads(OrionLayerGrads* g, const OrionModelConfig* cfg) {
    int d = cfg->d_model, h = cfg->hidden_dim;
    g->drms_att = alloc_zeros(d);
    g->dwq = alloc_zeros(d*d);
    g->dwk = alloc_zeros(d*d);
    g->dwv = alloc_zeros(d*d);
    g->dwo = alloc_zeros(d*d);
    g->drms_ffn = alloc_zeros(d);
    g->dw1 = alloc_zeros(h*d);
    g->dw3 = alloc_zeros(h*d);
    g->dw2 = alloc_zeros(d*h);
}

static void zero_layer_grads(OrionLayerGrads* g, const OrionModelConfig* cfg) {
    int d = cfg->d_model, h = cfg->hidden_dim;
    memset(g->drms_att, 0, d * sizeof(float));
    memset(g->dwq, 0, d*d * sizeof(float));
    memset(g->dwk, 0, d*d * sizeof(float));
    memset(g->dwv, 0, d*d * sizeof(float));
    memset(g->dwo, 0, d*d * sizeof(float));
    memset(g->drms_ffn, 0, d * sizeof(float));
    memset(g->dw1, 0, h*d * sizeof(float));
    memset(g->dw3, 0, h*d * sizeof(float));
    memset(g->dw2, 0, d*h * sizeof(float));
}

static void free_layer_grads(OrionLayerGrads* g) {
    free(g->drms_att); free(g->dwq); free(g->dwk); free(g->dwv); free(g->dwo);
    free(g->drms_ffn); free(g->dw1); free(g->dw3); free(g->dw2);
}

static void alloc_layer_adam(OrionLayerAdam* a, const OrionModelConfig* cfg) {
    int d = cfg->d_model, h = cfg->hidden_dim;
    a->m_rms_att = alloc_zeros(d); a->v_rms_att = alloc_zeros(d);
    a->m_wq = alloc_zeros(d*d); a->v_wq = alloc_zeros(d*d);
    a->m_wk = alloc_zeros(d*d); a->v_wk = alloc_zeros(d*d);
    a->m_wv = alloc_zeros(d*d); a->v_wv = alloc_zeros(d*d);
    a->m_wo = alloc_zeros(d*d); a->v_wo = alloc_zeros(d*d);
    a->m_rms_ffn = alloc_zeros(d); a->v_rms_ffn = alloc_zeros(d);
    a->m_w1 = alloc_zeros(h*d); a->v_w1 = alloc_zeros(h*d);
    a->m_w3 = alloc_zeros(h*d); a->v_w3 = alloc_zeros(h*d);
    a->m_w2 = alloc_zeros(d*h); a->v_w2 = alloc_zeros(d*h);
}

static void free_layer_adam(OrionLayerAdam* a) {
    free(a->m_rms_att); free(a->v_rms_att);
    free(a->m_wq); free(a->v_wq); free(a->m_wk); free(a->v_wk);
    free(a->m_wv); free(a->v_wv); free(a->m_wo); free(a->v_wo);
    free(a->m_rms_ffn); free(a->v_rms_ffn);
    free(a->m_w1); free(a->v_w1); free(a->m_w3); free(a->v_w3);
    free(a->m_w2); free(a->v_w2);
}

static void alloc_layer_weights(OrionLayerWeights* w, const OrionModelConfig* cfg) {
    int d = cfg->d_model, h = cfg->hidden_dim;
    w->rms_att = alloc_zeros(d);
    w->wq = alloc_zeros(d*d);
    w->wk = alloc_zeros(d*d);
    w->wv = alloc_zeros(d*d);
    w->wo = alloc_zeros(d*d);
    w->rms_ffn = alloc_zeros(d);
    w->w1 = alloc_zeros(h*d);
    w->w3 = alloc_zeros(h*d);
    w->w2 = alloc_zeros(d*h);
}

static void free_layer_weights(OrionLayerWeights* w) {
    free(w->rms_att); free(w->wq); free(w->wk); free(w->wv); free(w->wo);
    free(w->rms_ffn); free(w->w1); free(w->w3); free(w->w2);
}

static void load_layer_weights(OrionLayerWeights* w, int layer_idx,
                                const OrionModelConfig* cfg, const char* base) {
    int d = cfg->d_model, h = cfg->hidden_dim;
    char path[512];

    snprintf(path, sizeof(path), "%s/layer%d/rms1.bin", base, layer_idx);
    load_blob_f32(path, w->rms_att, d);
    snprintf(path, sizeof(path), "%s/layer%d/wq.bin", base, layer_idx);
    load_blob_f32(path, w->wq, d*d);
    snprintf(path, sizeof(path), "%s/layer%d/wk.bin", base, layer_idx);
    load_blob_f32(path, w->wk, d*d);
    snprintf(path, sizeof(path), "%s/layer%d/wv.bin", base, layer_idx);
    load_blob_f32(path, w->wv, d*d);
    snprintf(path, sizeof(path), "%s/layer%d/wo.bin", base, layer_idx);
    load_blob_f32(path, w->wo, d*d);
    snprintf(path, sizeof(path), "%s/layer%d/rms2.bin", base, layer_idx);
    load_blob_f32(path, w->rms_ffn, d);
    snprintf(path, sizeof(path), "%s/layer%d/w1.bin", base, layer_idx);
    load_blob_f32(path, w->w1, h*d);
    snprintf(path, sizeof(path), "%s/layer%d/w3.bin", base, layer_idx);
    load_blob_f32(path, w->w3, h*d);
    snprintf(path, sizeof(path), "%s/layer%d/w2.bin", base, layer_idx);
    load_blob_f32(path, w->w2, d*h);
}

#pragma mark - Create / Free Trainer

/// Shared allocation and weight loading. If do_compile is true, compiles ANE programs.
/// If false, skips compilation (deferred mode for resume path).
static OrionTrainer* trainer_alloc_and_load(const OrionModelConfig* cfg,
                                             const char* weight_path,
                                             bool do_compile) {
    @autoreleasepool {
        if (!orion_ane_init()) return NULL;

        int nl = cfg->n_layer;
        int d = cfg->d_model;
        int s = cfg->max_seq;
        int v = cfg->vocab;

        OrionTrainer *t = (OrionTrainer *)calloc(1, sizeof(OrionTrainer));
        t->cfg = cfg;
        t->n_layers = nl;
        t->lr = 3e-4f;
        t->beta1 = 0.9f;
        t->beta2 = 0.999f;
        t->eps = 1e-8f;
        t->adam_t = 0;

        // Allocate per-layer arrays
        t->kernels = (OrionLayerKernels *)calloc(nl, sizeof(OrionLayerKernels));
        t->weights = (OrionLayerWeights *)calloc(nl, sizeof(OrionLayerWeights));
        t->grads   = (OrionLayerGrads *)calloc(nl, sizeof(OrionLayerGrads));
        t->adam    = (OrionLayerAdam *)calloc(nl, sizeof(OrionLayerAdam));
        t->io      = (OrionLayerIO *)calloc(nl, sizeof(OrionLayerIO));

        NSString *base = @(weight_path);

        // Compile (if requested) and load weights per layer
        for (int L = 0; L < nl; L++) {
            if (do_compile) {
                if (!compile_layer(&t->kernels[L], L, cfg, base)) {
                    NSLog(@"Failed to compile layer %d kernels", L);
                    orion_trainer_free(t);
                    return NULL;
                }
            }
            alloc_layer_io(&t->io[L], cfg);
            alloc_layer_weights(&t->weights[L], cfg);
            alloc_layer_grads(&t->grads[L], cfg);
            alloc_layer_adam(&t->adam[L], cfg);
            load_layer_weights(&t->weights[L], L, cfg, weight_path);
        }

        // Embedding + final RMSNorm
        t->embed = alloc_zeros(v * d);
        t->rms_final = alloc_zeros(d);
        t->dembed = alloc_zeros(v * d);
        t->drms_final = alloc_zeros(d);
        t->m_embed = alloc_zeros(v * d);
        t->v_embed = alloc_zeros(v * d);
        t->m_rms_final = alloc_zeros(d);
        t->v_rms_final = alloc_zeros(d);

        char path[512];
        snprintf(path, sizeof(path), "%s/embed.bin", weight_path);
        load_blob_f32(path, t->embed, v * d);
        snprintf(path, sizeof(path), "%s/rms_final.bin", weight_path);
        load_blob_f32(path, t->rms_final, d);

        // CPU activation buffers
        t->act_x  = alloc_zeros((nl + 1) * s * d);
        t->act_x2 = alloc_zeros(nl * s * d);
        t->logits  = alloc_zeros(s * v);
        t->dlogits = alloc_zeros(s * v);

        return t;
    }
}

OrionTrainer* orion_trainer_create(const OrionModelConfig* cfg, const char* weight_path) {
    return trainer_alloc_and_load(cfg, weight_path, true);
}

OrionTrainer* orion_trainer_create_deferred(const OrionModelConfig* cfg, const char* weight_path) {
    return trainer_alloc_and_load(cfg, weight_path, false);
}

void orion_trainer_free(OrionTrainer* t) {
    if (!t) return;
    for (int L = 0; L < t->n_layers; L++) {
        if (t->kernels[L].fwd_attn)  orion_release_program(t->kernels[L].fwd_attn);
        if (t->kernels[L].fwd_ffn)   orion_release_program(t->kernels[L].fwd_ffn);
        if (t->kernels[L].ffn_bwd)   orion_release_program(t->kernels[L].ffn_bwd);
        if (t->kernels[L].sdpa_bwd1) orion_release_program(t->kernels[L].sdpa_bwd1);
        if (t->kernels[L].sdpa_bwd2) orion_release_program(t->kernels[L].sdpa_bwd2);
        if (t->kernels[L].qkv_bwd)   orion_release_program(t->kernels[L].qkv_bwd);
        free_layer_io(&t->io[L]);
        free_layer_weights(&t->weights[L]);
        free_layer_grads(&t->grads[L]);
        free_layer_adam(&t->adam[L]);
    }
    free(t->kernels); free(t->weights); free(t->grads); free(t->adam); free(t->io);
    free(t->embed); free(t->rms_final);
    free(t->dembed); free(t->drms_final);
    free(t->m_embed); free(t->v_embed);
    free(t->m_rms_final); free(t->v_rms_final);
    free(t->act_x); free(t->act_x2);
    free(t->logits); free(t->dlogits);
    free(t);
}

#pragma mark - Zero Grads

void orion_trainer_zero_grads(OrionTrainer* t) {
    for (int L = 0; L < t->n_layers; L++) {
        zero_layer_grads(&t->grads[L], t->cfg);
    }
    memset(t->dembed, 0, t->cfg->vocab * t->cfg->d_model * sizeof(float));
    memset(t->drms_final, 0, t->cfg->d_model * sizeof(float));
}

#pragma mark - Scale Grads

static void scale_buffer(float* buf, int count, float scale) {
    vDSP_vsmul(buf, 1, &scale, buf, 1, count);
}

void orion_trainer_scale_grads(OrionTrainer* t, float scale) {
    int d = t->cfg->d_model, h = t->cfg->hidden_dim, v = t->cfg->vocab;
    for (int L = 0; L < t->n_layers; L++) {
        OrionLayerGrads *g = &t->grads[L];
        scale_buffer(g->drms_att, d, scale);
        scale_buffer(g->dwq, d*d, scale);
        scale_buffer(g->dwk, d*d, scale);
        scale_buffer(g->dwv, d*d, scale);
        scale_buffer(g->dwo, d*d, scale);
        scale_buffer(g->drms_ffn, d, scale);
        scale_buffer(g->dw1, h*d, scale);
        scale_buffer(g->dw3, h*d, scale);
        scale_buffer(g->dw2, d*h, scale);
    }
    scale_buffer(t->dembed, v * d, scale);
    scale_buffer(t->drms_final, d, scale);
}

#pragma mark - Training Step

float orion_train_step(OrionTrainer* trainer,
                        const int* input_tokens, const int* target_tokens) {
    @autoreleasepool {
        const OrionModelConfig *cfg = trainer->cfg;
        int d  = cfg->d_model;
        int h  = cfg->hidden_dim;
        int s  = cfg->max_seq;
        int v  = cfg->vocab;
        int nl = trainer->n_layers;
        int nh = cfg->n_head;
        int sc_ch = nh * s;

        // =========================================================
        // 1. EMBEDDING (CPU)
        // =========================================================
        float *x0 = trainer->act_x;  // layer 0 input: [s, d]
        orion_cpu_embedding(x0, trainer->embed, input_tokens, d, s);

        // =========================================================
        // 2. FORWARD PASS (ANE per-layer)
        // =========================================================
        for (int L = 0; L < nl; L++) {
            OrionLayerIO *io = &trainer->io[L];
            OrionLayerKernels *kern = &trainer->kernels[L];
            float *x_in  = trainer->act_x + L * s * d;       // layer input
            float *x_out = trainer->act_x + (L+1) * s * d;   // next layer input
            float *x2    = trainer->act_x2 + L * s * d;      // intermediate

            // Write layer input to ANE (CPU [s,d] → ANE [d,s])
            io_write_transpose(io->fwd_attn_in, x_in, d, s);

            // fwdAttn: x → [wo_out, q_out, k_out, v_out, attn_out, rms1_out]
            IOSurfaceRef attn_in[] = {io->fwd_attn_in};
            bool ok = orion_eval(kern->fwd_attn, attn_in, 1, io->fwd_attn_out, 6);
            if (!ok) { NSLog(@"fwdAttn eval failed layer %d", L); return -1.0f; }

            // Residual: x2 = x_in + wo_out  (CPU)
            io_read_transpose(io->fwd_attn_out[0], x2, d, s);  // wo_out
            for (int i = 0; i < s * d; i++) x2[i] += x_in[i];

            // Write x2 to FFN input
            io_write_transpose(io->fwd_ffn_in, x2, d, s);

            // fwdFFN: x2 → [w2_out, w1_out, w3_out, gate, rms2_out]
            IOSurfaceRef ffn_in[] = {io->fwd_ffn_in};
            ok = orion_eval(kern->fwd_ffn, ffn_in, 1, io->fwd_ffn_out, 5);
            if (!ok) { NSLog(@"fwdFFN eval failed layer %d", L); return -1.0f; }

            // Residual: x_out = x2 + w2_out  (CPU)
            io_read_transpose(io->fwd_ffn_out[0], x_out, d, s);  // w2_out
            for (int i = 0; i < s * d; i++) x_out[i] += x2[i];

        }

        // =========================================================
        // 3. FINAL RMSNORM + CLASSIFIER + LOSS (CPU)
        // =========================================================
        float *x_final = trainer->act_x + nl * s * d;
        float *x_normed = (float *)malloc(s * d * sizeof(float));

        // RMSNorm final (per position)
        for (int t = 0; t < s; t++) {
            orion_cpu_rmsnorm(x_normed + t*d, x_final + t*d, trainer->rms_final, d, 1e-5f);
        }

        // Classifier: logits = x_normed @ embed^T  → [s, v]
        // x_normed [s, d] @ embed^T [d, v] → logits [s, v]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    s, v, d, 1.0f,
                    x_normed, d, trainer->embed, d,
                    0.0f, trainer->logits, v);

        // Cross-entropy loss + gradient
        float loss = orion_cpu_cross_entropy(trainer->dlogits, trainer->logits,
                                              target_tokens, v, s);

        // =========================================================
        // 4. BACKWARD: CLASSIFIER + FINAL RMSNORM (CPU)
        // =========================================================
        // dlogits is [s, v] (softmax - one_hot, averaged)
        // dx_final = dlogits @ embed → [s, d]
        float *dy = (float *)malloc(s * d * sizeof(float));
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    s, d, v, 1.0f,
                    trainer->dlogits, v, trainer->embed, d,
                    0.0f, dy, d);

        // dW_embed += x_normed^T @ dlogits → [d, v] → stored as [v, d]
        // Actually: dembed[i] += sum over positions where dlogits refers to embed[i]
        // More precisely: dembed += dlogits^T @ x_normed → [v, d]
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    v, d, s, 1.0f,
                    trainer->dlogits, v, x_normed, d,
                    1.0f, trainer->dembed, d);

        // RMSNorm final backward
        float *dx_rms_final = (float *)malloc(s * d * sizeof(float));
        orion_cpu_rmsnorm_bwd(dx_rms_final, trainer->drms_final,
                               dy, x_final, trainer->rms_final, d, s, 1e-5f);
        free(dy);
        dy = dx_rms_final;  // dy now points to gradient flowing to last layer

        free(x_normed);

        // =========================================================
        // 5. BACKWARD PASS (ANE per-layer, reverse order)
        //    T073: dW cblas_sgemm dispatched async to GCD serial queue.
        //    Deferred wait before return for maximum ANE/CPU overlap.
        // =========================================================
        dispatch_queue_t dw_q = dispatch_queue_create("orion.dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        for (int L = nl - 1; L >= 0; L--) {
            OrionLayerIO *io = &trainer->io[L];
            OrionLayerKernels *kern = &trainer->kernels[L];
            OrionLayerGrads *gr = &trainer->grads[L];
            OrionLayerWeights *wt = &trainer->weights[L];
            float *x_in = trainer->act_x + L * s * d;
            float *x2   = trainer->act_x2 + L * s * d;

            // --- FFN Backward ---
            // Assemble input: [dy(d), h1(h), h3(h)] into ffn_bwd_in
            io_write_transpose(io->ffn_bwd_in, dy, d, s);
            orion_tensor_copy_into(io->ffn_bwd_in, d,
                                    io->fwd_ffn_out[1], h, s);  // w1_out = h1
            orion_tensor_copy_into(io->ffn_bwd_in, d + h,
                                    io->fwd_ffn_out[2], h, s);  // w3_out = h3

            IOSurfaceRef ffn_bwd_in[] = {io->ffn_bwd_in};
            bool ok = orion_eval(kern->ffn_bwd, ffn_bwd_in, 1, io->ffn_bwd_out, 3);
            if (!ok) { NSLog(@"ffnBwd eval failed layer %d", L); free(dy); return -1.0f; }

            // Read dx_ffn for RMSNorm backward
            float *dx_ffn = (float *)malloc(s * d * sizeof(float));
            io_read_transpose(io->ffn_bwd_out[0], dx_ffn, d, s);

            // dW for FFN — async dispatch to GCD serial queue
            {
                // Capture buffers (heap copy for block safety)
                float *c_dh1 = (float *)malloc(s * h * sizeof(float));
                float *c_dh3 = (float *)malloc(s * h * sizeof(float));
                float *c_xn  = (float *)malloc(s * d * sizeof(float));
                float *c_gate = (float *)malloc(s * h * sizeof(float));
                float *c_dy  = (float *)malloc(s * d * sizeof(float));
                io_read_transpose(io->ffn_bwd_out[1], c_dh1, h, s);
                io_read_transpose(io->ffn_bwd_out[2], c_dh3, h, s);
                io_read_transpose(io->fwd_ffn_out[4], c_xn, d, s);   // rms2_out
                io_read_transpose(io->fwd_ffn_out[3], c_gate, h, s);  // gate
                memcpy(c_dy, dy, s * d * sizeof(float));

                int c_s = s, c_d = d, c_h = h;
                float *g_dw2 = gr->dw2, *g_dw1 = gr->dw1, *g_dw3 = gr->dw3;
                dispatch_group_async(dw_grp, dw_q, ^{
                    orion_cpu_dw_accum(g_dw2, c_gate, c_dy, c_s, c_h, c_d);
                    orion_cpu_dw_accum(g_dw1, c_xn, c_dh1, c_s, c_d, c_h);
                    orion_cpu_dw_accum(g_dw3, c_xn, c_dh3, c_s, c_d, c_h);
                    free(c_dh1); free(c_dh3); free(c_xn); free(c_gate); free(c_dy);
                });
            }

            // RMSNorm2 backward (CPU)
            float *dx2 = (float *)malloc(s * d * sizeof(float));
            orion_cpu_rmsnorm_bwd(dx2, gr->drms_ffn, dx_ffn, x2, wt->rms_ffn, d, s, 1e-5f);
            free(dx_ffn);
            // Residual merge: dx2 += dy
            for (int i = 0; i < s * d; i++) dx2[i] += dy[i];

            // --- Attention Backward ---
            // sdpaBwd1 input: [qf(d), kf(d), vf(d), dx2(d)]
            orion_tensor_copy_into(io->sdpa_bwd1_in, 0,
                                    io->fwd_attn_out[1], d, s);  // q_out
            orion_tensor_copy_into(io->sdpa_bwd1_in, d,
                                    io->fwd_attn_out[2], d, s);  // k_out
            orion_tensor_copy_into(io->sdpa_bwd1_in, 2*d,
                                    io->fwd_attn_out[3], d, s);  // v_out
            {
                IOSurfaceRef tmp = orion_tensor_create(d, s);
                io_write_transpose(tmp, dx2, d, s);
                orion_tensor_copy_into(io->sdpa_bwd1_in, 3*d, tmp, d, s);
                CFRelease(tmp);
            }

            IOSurfaceRef sdpa1_in[] = {io->sdpa_bwd1_in};
            ok = orion_eval(kern->sdpa_bwd1, sdpa1_in, 1, io->sdpa_bwd1_out, 3);
            if (!ok) { NSLog(@"sdpaBwd1 eval failed layer %d", L); free(dy); free(dx2); return -1.0f; }

            // sdpaBwd2 input: [pf(sc_ch), dpf(sc_ch), qf(d), kf(d)]
            orion_tensor_copy_into(io->sdpa_bwd2_in, 0,
                                    io->sdpa_bwd1_out[1], sc_ch, s);  // pf
            orion_tensor_copy_into(io->sdpa_bwd2_in, sc_ch,
                                    io->sdpa_bwd1_out[2], sc_ch, s);  // dpf
            orion_tensor_copy_into(io->sdpa_bwd2_in, 2*sc_ch,
                                    io->fwd_attn_out[1], d, s);       // q_out
            orion_tensor_copy_into(io->sdpa_bwd2_in, 2*sc_ch + d,
                                    io->fwd_attn_out[2], d, s);       // k_out

            IOSurfaceRef sdpa2_in[] = {io->sdpa_bwd2_in};
            ok = orion_eval(kern->sdpa_bwd2, sdpa2_in, 1, io->sdpa_bwd2_out, 2);
            if (!ok) { NSLog(@"sdpaBwd2 eval failed layer %d", L); free(dy); free(dx2); return -1.0f; }

            // dW for attention — async dispatch to GCD serial queue
            {
                float *c_xn  = (float *)malloc(s * d * sizeof(float));
                float *c_dq  = (float *)malloc(s * d * sizeof(float));
                float *c_dk  = (float *)malloc(s * d * sizeof(float));
                float *c_dv  = (float *)malloc(s * d * sizeof(float));
                float *c_ao  = (float *)malloc(s * d * sizeof(float));
                float *c_dx2 = (float *)malloc(s * d * sizeof(float));
                io_read_transpose(io->fwd_attn_out[5], c_xn, d, s);  // rms1_out
                io_read_transpose(io->sdpa_bwd2_out[0], c_dq, d, s); // dqf
                io_read_transpose(io->sdpa_bwd2_out[1], c_dk, d, s); // dkf
                io_read_transpose(io->sdpa_bwd1_out[0], c_dv, d, s); // dvf
                io_read_transpose(io->fwd_attn_out[4], c_ao, d, s);  // attn_out
                memcpy(c_dx2, dx2, s * d * sizeof(float));

                int c_s = s, c_d = d;
                float *g_dwq = gr->dwq, *g_dwk = gr->dwk;
                float *g_dwv = gr->dwv, *g_dwo = gr->dwo;
                dispatch_group_async(dw_grp, dw_q, ^{
                    orion_cpu_dw_accum(g_dwq, c_xn, c_dq, c_s, c_d, c_d);
                    orion_cpu_dw_accum(g_dwk, c_xn, c_dk, c_s, c_d, c_d);
                    orion_cpu_dw_accum(g_dwv, c_xn, c_dv, c_s, c_d, c_d);
                    orion_cpu_dw_accum(g_dwo, c_ao, c_dx2, c_s, c_d, c_d);
                    free(c_xn); free(c_dq); free(c_dk); free(c_dv);
                    free(c_ao); free(c_dx2);
                });
            }

            // qkvBwd input: [dq(d), dk(d), dv(d)]
            orion_tensor_copy_into(io->qkv_bwd_in, 0,
                                    io->sdpa_bwd2_out[0], d, s);  // dqf
            orion_tensor_copy_into(io->qkv_bwd_in, d,
                                    io->sdpa_bwd2_out[1], d, s);  // dkf
            orion_tensor_copy_into(io->qkv_bwd_in, 2*d,
                                    io->sdpa_bwd1_out[0], d, s);  // dvf

            IOSurfaceRef qkv_in[] = {io->qkv_bwd_in};
            IOSurfaceRef qkv_out[] = {io->qkv_bwd_out};
            ok = orion_eval(kern->qkv_bwd, qkv_in, 1, qkv_out, 1);
            if (!ok) { NSLog(@"qkvBwd eval failed layer %d", L); free(dy); free(dx2); return -1.0f; }

            // RMSNorm1 backward (CPU)
            float *dx_attn = (float *)malloc(s * d * sizeof(float));
            io_read_transpose(io->qkv_bwd_out, dx_attn, d, s);

            float *dx_rms1 = (float *)malloc(s * d * sizeof(float));
            orion_cpu_rmsnorm_bwd(dx_rms1, gr->drms_att, dx_attn, x_in, wt->rms_att, d, s, 1e-5f);
            free(dx_attn);

            // Gradient for previous layer: dy = dx_rms1 + dx2
            free(dy);
            dy = (float *)malloc(s * d * sizeof(float));
            for (int i = 0; i < s * d; i++) dy[i] = dx_rms1[i] + dx2[i];
            free(dx_rms1);
            free(dx2);
        }

        // =========================================================
        // 6. EMBEDDING BACKWARD (CPU)
        // =========================================================
        orion_cpu_embedding_bwd(trainer->dembed, dy, input_tokens, d, s);
        free(dy);

        // Deferred wait: all async dW sgemms must complete before return
        dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);

        return loss;
    }
}

#pragma mark - Adam Update

void orion_trainer_adam_update(OrionTrainer* t) {
    int d = t->cfg->d_model, h = t->cfg->hidden_dim, v = t->cfg->vocab;
    t->adam_t++;

    for (int L = 0; L < t->n_layers; L++) {
        OrionLayerWeights *w = &t->weights[L];
        OrionLayerGrads *g = &t->grads[L];
        OrionLayerAdam *a = &t->adam[L];

        orion_cpu_adam_step(w->rms_att, g->drms_att, a->m_rms_att, a->v_rms_att,
                            d, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
        orion_cpu_adam_step(w->wq, g->dwq, a->m_wq, a->v_wq,
                            d*d, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
        orion_cpu_adam_step(w->wk, g->dwk, a->m_wk, a->v_wk,
                            d*d, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
        orion_cpu_adam_step(w->wv, g->dwv, a->m_wv, a->v_wv,
                            d*d, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
        orion_cpu_adam_step(w->wo, g->dwo, a->m_wo, a->v_wo,
                            d*d, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
        orion_cpu_adam_step(w->rms_ffn, g->drms_ffn, a->m_rms_ffn, a->v_rms_ffn,
                            d, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
        orion_cpu_adam_step(w->w1, g->dw1, a->m_w1, a->v_w1,
                            h*d, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
        orion_cpu_adam_step(w->w3, g->dw3, a->m_w3, a->v_w3,
                            h*d, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
        orion_cpu_adam_step(w->w2, g->dw2, a->m_w2, a->v_w2,
                            d*h, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
    }

    // Embedding + final RMSNorm
    orion_cpu_adam_step(t->embed, t->dembed, t->m_embed, t->v_embed,
                        v*d, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
    orion_cpu_adam_step(t->rms_final, t->drms_final, t->m_rms_final, t->v_rms_final,
                        d, t->lr, t->beta1, t->beta2, t->eps, t->adam_t);
}

#pragma mark - T078: Recompile After Adam

bool orion_trainer_recompile(OrionTrainer* t, const char* weight_path) {
    @autoreleasepool {
        NSString *base = @(weight_path);
        int nl = t->n_layers;

        // Write updated CPU weights to disk as BLOBFILE
        for (int L = 0; L < nl; L++) {
            save_layer_weights(&t->weights[L], L, t->cfg, weight_path);
        }
        // Also write embed (needed for causal mask, not weight-bearing in ANE)
        // Causal mask doesn't change, no need to rewrite.

        // Release old programs and recompile with new weights
        for (int L = 0; L < nl; L++) {
            OrionLayerKernels *kern = &t->kernels[L];

            // Release old programs
            if (kern->fwd_attn)  orion_release_program(kern->fwd_attn);
            if (kern->fwd_ffn)   orion_release_program(kern->fwd_ffn);
            if (kern->ffn_bwd)   orion_release_program(kern->ffn_bwd);
            if (kern->sdpa_bwd1) orion_release_program(kern->sdpa_bwd1);
            if (kern->sdpa_bwd2) orion_release_program(kern->sdpa_bwd2);
            if (kern->qkv_bwd)   orion_release_program(kern->qkv_bwd);

            memset(kern, 0, sizeof(OrionLayerKernels));

            // Recompile with updated weights
            if (!compile_layer(kern, L, t->cfg, base)) {
                NSLog(@"Recompile failed for layer %d", L);
                return false;
            }
        }

        return true;
    }
}

#pragma mark - T077: Compile Budget Check

bool orion_trainer_needs_restart(const OrionTrainer* t, int* budget_remaining) {
    int used = orion_compile_count();
    // 6 programs per layer; need enough budget for one full recompile
    int needed = t->n_layers * 6;
    int limit = STORIES_MAX_COMPILES;  // 100 (conservative vs ~119 hard limit)
    int remaining = limit - used;

    if (budget_remaining) *budget_remaining = remaining;
    return remaining < needed;
}
