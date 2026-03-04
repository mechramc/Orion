#import "prefill_ane.h"
#import "../../compiler/kernel_adapter.h"
#import "../../compiler/codegen.h"
#include "../../compiler/validate.h"
#include "../../compiler/pipeline.h"
#include "../../compiler/frontends/gpt2_prefill.h"
#include "../../compiler/frontends/gpt2_final.h"
#import "../../core/bucket.h"
#import "../../core/mil_builder.h"
#import "../../core/iosurface_tensor.h"
#import "../../core/ane_program_cache.h"
#import "../../core/kernel.h"
#import "../../model/configs/gpt2_124m.h"
#import <Accelerate/Accelerate.h>
#import <sys/time.h>
#import <math.h>

// T051: Prompt padding + layout conversion
// T052: ANE prefill runner
// T053: K,V extraction from ANE output into KV cache

#pragma mark - IOSurface Helpers

static IOSurfaceRef make_f32_surface(int count) {
    size_t bytes = count * sizeof(float);
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

static IOSurfaceRef make_f32_surface_data(const float *data, int count) {
    IOSurfaceRef s = make_f32_surface(count);
    IOSurfaceLock(s, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(s), data, count * sizeof(float));
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

static void read_f32_surface(IOSurfaceRef s, float *out, int count) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    memcpy(out, IOSurfaceGetBaseAddress(s), count * sizeof(float));
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

#pragma mark - Weight Dict Helpers

static void add_blob(NSMutableDictionary *dict, NSString *mil_path, NSString *file_path) {
    NSData *data = [NSData dataWithContentsOfFile:file_path];
    if (data) {
        dict[mil_path] = @{@"offset": @0, @"data": data};
    }
}

/// Build weight dict for attention layer from blob files.
static NSDictionary* build_attn_wdict(int layer_idx, int seq_len, NSString *dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];

    const char *names[] = {"ln1_g", "ln1_b", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"};
    for (int i = 0; i < 10; i++) {
        NSString *mil_path = [NSString stringWithFormat:@"@model_path/layer%d/%s.bin",
                              layer_idx, names[i]];
        NSString *file_path = [NSString stringWithFormat:@"%@/layer%d/%s.bin",
                               dir, layer_idx, names[i]];
        add_blob(dict, mil_path, file_path);
    }

    // Causal mask
    NSString *mask_path = orion_causal_mask_path(seq_len);
    NSData *mask = orion_make_causal_mask_blob(seq_len);
    dict[mask_path] = @{@"offset": @0, @"data": mask};

    return dict;
}

/// Build weight dict for FFN layer from blob files.
static NSDictionary* build_ffn_wdict(int layer_idx, NSString *dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];

    const char *names[] = {"ln2_g", "ln2_b", "wfc", "bfc", "wproj", "bproj"};
    for (int i = 0; i < 6; i++) {
        NSString *mil_path = [NSString stringWithFormat:@"@model_path/layer%d/%s.bin",
                              layer_idx, names[i]];
        NSString *file_path = [NSString stringWithFormat:@"%@/layer%d/%s.bin",
                               dir, layer_idx, names[i]];
        add_blob(dict, mil_path, file_path);
    }
    return dict;
}

/// Build weight dict for final LayerNorm from blob files.
static NSDictionary* build_final_ln_wdict(NSString *dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    add_blob(dict, @"@model_path/ln_f_g.bin",
             [dir stringByAppendingPathComponent:@"ln_f_g.bin"]);
    add_blob(dict, @"@model_path/ln_f_b.bin",
             [dir stringByAppendingPathComponent:@"ln_f_b.bin"]);
    return dict;
}

#pragma mark - OrionKernel Adapters

// Compiler MIL gen adapters — call frontend → validate → optimize → codegen
static NSString* compiler_prefill_attn_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    return orion_kernel_adapter_generate_mil(orion_frontend_gpt2_prefill_attn, layer_idx, bucket, cfg);
}

static NSString* compiler_prefill_ffn_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    return orion_kernel_adapter_generate_mil(orion_frontend_gpt2_prefill_ffn, layer_idx, bucket, cfg);
}

static NSString* compiler_final_ln_adapter(int layer_idx, int bucket, const OrionModelConfig* cfg) {
    (void)layer_idx;
    OrionGraph* g = orion_frontend_gpt2_final_ln(bucket, cfg);
    if (!g) return nil;
    OrionValidationResult vr = orion_graph_validate(g);
    if (!vr.valid) { orion_graph_free(g); return nil; }
    orion_pipeline_optimize(g);
    NSString* mil = orion_codegen_mil(g, "main");
    orion_graph_free(g);
    return mil;
}

// Weight dict adapters — convert const char* → NSString*
static NSDictionary* wdict_attn_adapter(int layer_idx, int bucket, const char* blob_dir) {
    return build_attn_wdict(layer_idx, bucket, @(blob_dir));
}

static NSDictionary* wdict_ffn_adapter(int layer_idx, int bucket, const char* blob_dir) {
    (void)bucket;
    return build_ffn_wdict(layer_idx, @(blob_dir));
}

static NSDictionary* wdict_final_ln_adapter(int layer_idx, int bucket, const char* blob_dir) {
    (void)layer_idx; (void)bucket;
    return build_final_ln_wdict(@(blob_dir));
}

// Kernel instances — using compiler-generated MIL
static const OrionKernel kPrefillAttn = {
    .name = "prefill_attn",
    .generate_mil = compiler_prefill_attn_adapter,
    .build_wdict = wdict_attn_adapter,
    .n_inputs = 1, .n_outputs = 3,
};

static const OrionKernel kPrefillFFN = {
    .name = "prefill_ffn",
    .generate_mil = compiler_prefill_ffn_adapter,
    .build_wdict = wdict_ffn_adapter,
    .n_inputs = 1, .n_outputs = 1,
};

static const OrionKernel kPrefillFinalLN = {
    .name = "prefill_final_ln",
    .generate_mil = compiler_final_ln_adapter,
    .build_wdict = wdict_final_ln_adapter,
    .n_inputs = 1, .n_outputs = 1,
};

#pragma mark - T051: Prompt Padding + Layout Conversion

IOSurfaceRef orion_prepare_ane_input(const OrionGPT2Weights* w,
                                      const int* tokens, int prompt_len,
                                      int bucket, const OrionModelConfig* cfg) {
    int d = cfg->d_model;
    int count = d * bucket;

    // Compute token + positional embeddings in CPU layout [seq, d_model]
    float *cpu_embed = (float *)calloc(bucket * d, sizeof(float));
    for (int s = 0; s < prompt_len; s++) {
        const float *tok_emb = w->wte + tokens[s] * d;
        const float *pos_emb = w->wpe + s * d;
        float *dst = cpu_embed + s * d;
        vDSP_vadd(tok_emb, 1, pos_emb, 1, dst, 1, d);
    }
    // Positions beyond prompt_len remain zero (padding)

    // Transpose CPU [seq, d_model] → ANE [d_model, bucket]
    float *ane_data = (float *)calloc(count, sizeof(float));
    for (int s = 0; s < bucket; s++) {
        for (int c = 0; c < d; c++) {
            ane_data[c * bucket + s] = cpu_embed[s * d + c];
        }
    }

    IOSurfaceRef surface = make_f32_surface_data(ane_data, count);

    free(cpu_embed);
    free(ane_data);
    return surface;
}

#pragma mark - T053: Extract K,V from ANE output to KV cache

/// Extract K,V from ANE layout [d_model, bucket] into KV cache.
/// Transposes ANE [d_model, bucket] → CPU [seq, d_model], then stores in cache.
static void extract_kv_to_cache(const float *k_ane, const float *v_ane,
                                  int layer_idx, int prompt_len, int bucket,
                                  OrionKVCache *kv, const OrionModelConfig *cfg) {
    int d = cfg->d_model;
    int nh = cfg->n_head;
    int hd = cfg->head_dim;

    // Transpose ANE [d_model, bucket] → CPU [prompt_len, d_model]
    // Only extract actual prompt positions (not padding)
    float *k_cpu = (float *)malloc(prompt_len * d * sizeof(float));
    float *v_cpu = (float *)malloc(prompt_len * d * sizeof(float));

    for (int s = 0; s < prompt_len; s++) {
        for (int c = 0; c < d; c++) {
            k_cpu[s * d + c] = k_ane[c * bucket + s];
            v_cpu[s * d + c] = v_ane[c * bucket + s];
        }
    }

    // Store into KV cache with head-split layout
    // KV cache expects [n_head, seq, head_dim] per layer
    int max_seq = kv->max_seq;
    for (int s = 0; s < prompt_len; s++) {
        for (int h = 0; h < nh; h++) {
            int cache_offset = layer_idx * (nh * max_seq * hd) + h * (max_seq * hd) + s * hd;
            int src_offset = s * d + h * hd;
            memcpy(kv->k_cache + cache_offset, k_cpu + src_offset, hd * sizeof(float));
            memcpy(kv->v_cache + cache_offset, v_cpu + src_offset, hd * sizeof(float));
        }
    }

    free(k_cpu);
    free(v_cpu);
}

#pragma mark - T052: ANE Prefill Runner

bool orion_ane_prefill(const OrionGPT2Weights* w,
                        const int* tokens, int prompt_len,
                        const OrionModelConfig* cfg,
                        const char* blob_dir,
                        OrionKVCache* kv,
                        float* logits) {
    int d = cfg->d_model;
    int n_layer = cfg->n_layer;
    int vocab = cfg->vocab;

    // Select bucket
    int bucket = orion_select_bucket(prompt_len, kGPT2Buckets, kGPT2NumBuckets);
    if (bucket < 0) {
        fprintf(stderr, "ANE prefill: prompt too long (%d tokens)\n", prompt_len);
        return false;
    }
    fprintf(stderr, "ANE prefill: %d tokens → bucket %d\n", prompt_len, bucket);

    int count = d * bucket;

    // T051: Prepare input (embed + pad + transpose)
    IOSurfaceRef ioHidden = orion_prepare_ane_input(w, tokens, prompt_len, bucket, cfg);
    if (!ioHidden) return false;

    // T084: Cache binding — "base" weights_id for standard inference
    OrionWeightsBinding wb = { .weights_id = "base", .bucket = bucket };
    int compiles_before = orion_compile_count();

    // Compile and run 12 layers of (attention + FFN)
    for (int layer = 0; layer < n_layer; layer++) {
        // --- Attention: 1 input → 3 outputs (hidden, K, V) ---
        IOSurfaceRef ioAttnOut = make_f32_surface(count);
        IOSurfaceRef ioK       = make_f32_surface(count);
        IOSurfaceRef ioV       = make_f32_surface(count);

        IOSurfaceRef attn_ins[]  = {ioHidden};
        IOSurfaceRef attn_outs[] = {ioAttnOut, ioK, ioV};
        bool ok = orion_kernel_eval(&kPrefillAttn, layer, bucket, cfg,
                                    blob_dir, &wb, attn_ins, 1, attn_outs, 3);
        if (!ok) {
            fprintf(stderr, "ANE prefill: attention L%d failed\n", layer);
            CFRelease(ioHidden); CFRelease(ioAttnOut); CFRelease(ioK); CFRelease(ioV);
            return false;
        }

        // T053: Extract K,V into cache
        float *k_data = (float *)malloc(count * sizeof(float));
        float *v_data = (float *)malloc(count * sizeof(float));
        read_f32_surface(ioK, k_data, count);
        read_f32_surface(ioV, v_data, count);
        extract_kv_to_cache(k_data, v_data, layer, prompt_len, bucket, kv, cfg);
        free(k_data); free(v_data);

        // Release input, swap attn output as new hidden
        CFRelease(ioHidden);
        CFRelease(ioK);
        CFRelease(ioV);
        ioHidden = ioAttnOut;

        // --- FFN ---
        IOSurfaceRef ioFFNOut = make_f32_surface(count);
        IOSurfaceRef ffn_ins[]  = {ioHidden};
        IOSurfaceRef ffn_outs[] = {ioFFNOut};
        ok = orion_kernel_eval(&kPrefillFFN, layer, bucket, cfg,
                               blob_dir, &wb, ffn_ins, 1, ffn_outs, 1);
        if (!ok) {
            fprintf(stderr, "ANE prefill: FFN L%d failed\n", layer);
            CFRelease(ioHidden); CFRelease(ioFFNOut);
            return false;
        }

        CFRelease(ioHidden);
        ioHidden = ioFFNOut;
    }

    // Final LayerNorm on ANE
    IOSurfaceRef ioLNOut = make_f32_surface(count);
    IOSurfaceRef ln_ins[]  = {ioHidden};
    IOSurfaceRef ln_outs[] = {ioLNOut};
    bool ok = orion_kernel_eval(&kPrefillFinalLN, -1, bucket, cfg,
                                blob_dir, &wb, ln_ins, 1, ln_outs, 1);
    CFRelease(ioHidden);

    if (!ok) {
        fprintf(stderr, "ANE prefill: final LN failed\n");
        CFRelease(ioLNOut);
        return false;
    }

    // Read final LN output (ANE layout [d_model, bucket])
    float *ln_out_ane = (float *)malloc(count * sizeof(float));
    read_f32_surface(ioLNOut, ln_out_ane, count);
    CFRelease(ioLNOut);

    // Extract last prompt position in CPU layout
    // ANE: data[c * bucket + s] → CPU: data[c] for position (prompt_len - 1)
    float *last_hidden = (float *)malloc(d * sizeof(float));
    int last_pos = prompt_len - 1;
    for (int c = 0; c < d; c++) {
        last_hidden[c] = ln_out_ane[c * bucket + last_pos];
    }
    free(ln_out_ane);

    // Logits projection on CPU: logits = last_hidden @ wte^T
    // wte is [vocab, d_model], stored as [vocab, d_model] in CPU memory
    // logits[v] = sum_c(last_hidden[c] * wte[v * d + c])
    cblas_sgemv(CblasRowMajor, CblasNoTrans, vocab, d, 1.0f,
                w->wte, d, last_hidden, 1, 0.0f, logits, 1);

    free(last_hidden);

    // Update KV cache position
    kv->current_len = prompt_len;

    int compiles_after = orion_compile_count();
    fprintf(stderr, "ANE prefill: %d new compiles (%d programs cached, %d compiles total)\n",
            compiles_after - compiles_before, orion_cache_size(), compiles_after);
    return true;
}
