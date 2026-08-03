#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "compiler/frontends/qwen35_prefill.h"
#include "compiler/pipeline.h"
#include "compiler/validate.h"
#import "compiler/codegen.h"
#include "core/ane_runtime.h"
#import "../model/weight_loader.h"
#import "../kernels/inference/qwen_cpu_ops.h"
#import "../kernels/training/stories_cpu_ops.h"
#import "../kernels/training/qwen_lora_train.h"

static float abs_sum(const float *x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)x[i]);
    return (float)total;
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

static float *load_exact(const char *blob_dir, int layer_idx, const char *suffix, int count) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/layer%d/%s", blob_dir, layer_idx, suffix);
    return orion_read_blob_f32_exact(path, count);
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

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s <blob_dir> <adapter_out_dir> [input_token] [target_token]\n", argv[0]);
            return 2;
        }

        const char *blob_dir = argv[1];
        const char *adapter_out_dir = argv[2];
        const int input_token = argc >= 4 ? atoi(argv[3]) : 27;
        const int target_token = argc >= 5 ? atoi(argv[4]) : 91;
        const int bucket = 32;
        const int seq_len = 1;

        OrionQwen35Manifest *manifest = orion_qwen35_manifest_load(blob_dir);
        if (!manifest) {
            fprintf(stderr, "FAIL: manifest load failed\n");
            return 1;
        }
        if (!orion_ane_init()) {
            fprintf(stderr, "FAIL: ane init failed\n");
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        const int layer_idx = manifest->n_layer - 1;
        const int d_model = manifest->d_model;
        const int d_ff = manifest->d_ff;
        const int n_head = manifest->n_head;
        const int n_kv_head = manifest->n_kv_head;
        const int head_dim = manifest->head_dim;
        const int q_dim = n_head * head_dim;
        const int kv_dim = n_kv_head * head_dim;
        NSString *blobDir = [NSString stringWithUTF8String:blob_dir];

        OrionQwen9BLoRATrainer trainer;
        orion_qwen9b_lora_trainer_init(&trainer, manifest, layer_idx, 8, 16.0f, 1e-3f, 1337u);

        float *hidden_in = (float *)calloc((size_t)d_model, sizeof(float));
        float *normed = (float *)calloc((size_t)d_model, sizeof(float));
        float *base_q = (float *)calloc((size_t)(q_dim * 2), sizeof(float));
        float *base_v = (float *)calloc((size_t)kv_dim, sizeof(float));
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
        float *input_ln = load_exact(blob_dir, layer_idx, "input_layernorm.bin", d_model);
        float *post_ln = load_exact(blob_dir, layer_idx, "post_attention_layernorm.bin", d_model);
        float *q_proj = load_exact(blob_dir, layer_idx, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
        float *v_proj = load_exact(blob_dir, layer_idx, "self_attn_v_proj.bin", kv_dim * d_model);
        float *o_proj = load_exact(blob_dir, layer_idx, "self_attn_o_proj.bin", d_model * q_dim);
        float *gate_proj = load_exact(blob_dir, layer_idx, "mlp_gate_proj.bin", d_ff * d_model);
        float *up_proj = load_exact(blob_dir, layer_idx, "mlp_up_proj.bin", d_ff * d_model);
        float *down_proj = load_exact(blob_dir, layer_idx, "mlp_down_proj.bin", d_model * d_ff);
        char final_norm_path[2048], embed_path[2048];
        snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
        snprintf(embed_path, sizeof(embed_path), "%s/model/embed_tokens.bin", blob_dir);
        float *final_norm_weight = orion_read_blob_f32_exact(final_norm_path, d_model);
        IOSurfaceRef ioIn = NULL;
        IOSurfaceRef ioQ = NULL;
        IOSurfaceRef ioK = NULL;
        IOSurfaceRef ioV = NULL;
        OrionProgram *progQ = NULL;
        OrionProgram *progKV = NULL;
        NSString *milQ = nil;
        NSString *milKV = nil;
        int ok = 0;
        float loss = NAN;
        float q_grad_abs_sum = 0.0f;
        float v_grad_abs_sum = 0.0f;

        if (!hidden_in || !normed || !base_q || !base_v || !delta_q || !delta_v || !q_full || !v_raw ||
            !attn_cat || !gated || !mixer || !hidden_mid || !post_norm || !mlp_out || !hidden_out ||
            !final_norm || !last_hidden || !d_last_hidden || !d_hidden_out || !d_post_norm ||
            !d_hidden_mid || !d_mixer || !d_gated || !d_attn_cat || !d_gate_half || !d_q_full ||
            !d_v || !d_normed_from_q || !d_normed_from_v || !d_normed || !input_ln || !post_ln ||
            !q_proj || !v_proj || !o_proj || !gate_proj || !up_proj || !down_proj || !final_norm_weight) {
            fprintf(stderr, "FAIL: alloc or weight load failed\n");
            goto cleanup;
        }

        if (!orion_qwen9b_lora_frozen_prefix_hidden(blob_dir, manifest, input_token, hidden_in)) {
            fprintf(stderr, "FAIL: frozen prefix hidden probe failed\n");
            goto cleanup;
        }
        orion_qwen_cpu_rmsnorm(hidden_in, input_ln, d_model, 1e-6f, normed);

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
        milQ = compile_graph(orion_frontend_qwen35_prefill_q_proj(layer_idx, bucket, &cfg));
        milKV = compile_graph(orion_frontend_qwen35_prefill_kv_proj(layer_idx, bucket, &cfg));
        if (!milQ || !milKV) {
            fprintf(stderr, "FAIL: mil graph build failed\n");
            goto cleanup;
        }
        progQ = orion_compile_mil(milQ.UTF8String, build_qproj_wdict(layer_idx, blobDir), "qwen35_9b_lora_ane_train_q");
        progKV = orion_compile_mil(milKV.UTF8String, build_kv_wdict(layer_idx, blobDir), "qwen35_9b_lora_ane_train_kv");
        if (!progQ || !progKV) {
            fprintf(stderr, "FAIL: ane compile failed\n");
            goto cleanup;
        }

        ioIn = make_cpu_seq_input_surface(hidden_in, seq_len, bucket, d_model);
        ioQ = make_f32_surface((q_dim * 2) * bucket, 0.0f);
        ioK = make_f32_surface(kv_dim * bucket, 0.0f);
        ioV = make_f32_surface(kv_dim * bucket, 0.0f);
        if (!ioIn || !ioQ || !ioK || !ioV) {
            fprintf(stderr, "FAIL: iosurface alloc failed\n");
            goto cleanup;
        }

        IOSurfaceRef ins[] = { ioIn };
        IOSurfaceRef outsQ[] = { ioQ };
        IOSurfaceRef outsKV[] = { ioK, ioV };
        if (!orion_eval(progQ, ins, 1, outsQ, 1) || !orion_eval(progKV, ins, 1, outsKV, 2)) {
            fprintf(stderr, "FAIL: ane q/v eval failed\n");
            goto cleanup;
        }

        read_ane_surface_prefix(ioQ, q_dim * 2, seq_len, bucket, base_q);
        read_ane_surface_prefix(ioV, kv_dim, seq_len, bucket, base_v);
        orion_qwen_lora_linear_delta_forward(normed, &trainer.q_proj, delta_q);
        orion_qwen_lora_linear_delta_forward(normed, &trainer.v_proj, delta_v);
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

        loss = orion_qwen_cpu_streaming_ce_tied_embedding(embed_path, last_hidden, d_model, manifest->vocab, target_token, d_last_hidden);
        if (!isfinite(loss)) {
            fprintf(stderr, "FAIL: non-finite loss\n");
            goto cleanup;
        }

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

        orion_qwen_lora_linear_backward(normed, q_proj, &trainer.q_proj, d_q_full, d_normed_from_q);
        orion_qwen_lora_linear_backward(normed, v_proj, &trainer.v_proj, d_v, d_normed_from_v);
        for (int i = 0; i < d_model; i++) d_normed[i] = d_normed_from_q[i] + d_normed_from_v[i];

        {
            float *throwaway_weight_grad = (float *)calloc((size_t)d_model, sizeof(float));
            float *throwaway_dx = (float *)calloc((size_t)d_model, sizeof(float));
            orion_cpu_rmsnorm_bwd(throwaway_dx, throwaway_weight_grad, d_normed, hidden_in, input_ln, d_model, 1, 1e-6f);
            free(throwaway_weight_grad);
            free(throwaway_dx);
        }

        q_grad_abs_sum = (float)orion_qwen_lora_grad_abs_sum(&trainer.q_proj);
        v_grad_abs_sum = (float)orion_qwen_lora_grad_abs_sum(&trainer.v_proj);
        orion_qwen9b_lora_trainer_step(&trainer);
        if (!orion_qwen9b_lora_trainer_save(&trainer, adapter_out_dir)) {
            fprintf(stderr, "FAIL: adapter save failed\n");
            goto cleanup;
        }

        printf("PASS: qwen35 9b lora ane train smoke1\n");
        printf("  blob_dir=%s\n", blob_dir);
        printf("  adapter_out_dir=%s\n", adapter_out_dir);
        printf("  input_token=%d\n", input_token);
        printf("  target_token=%d\n", target_token);
        printf("  layer_idx=%d\n", layer_idx);
        printf("  bucket=%d\n", bucket);
        printf("  step=%d\n", trainer.step);
        printf("  hidden_abs_sum=%.6f\n", abs_sum(hidden_in, d_model));
        printf("  normed_abs_sum=%.6f\n", abs_sum(normed, d_model));
        printf("  q_base_abs_sum=%.6f\n", abs_sum(base_q, q_dim * 2));
        printf("  v_base_abs_sum=%.6f\n", abs_sum(base_v, kv_dim));
        printf("  q_delta_abs_sum=%.6f\n", abs_sum(delta_q, q_dim * 2));
        printf("  v_delta_abs_sum=%.6f\n", abs_sum(delta_v, kv_dim));
        printf("  loss=%.6f\n", loss);
        printf("  q_grad_abs_sum=%.6f\n", q_grad_abs_sum);
        printf("  v_grad_abs_sum=%.6f\n", v_grad_abs_sum);
        printf("  q_param_abs_sum=%.6f\n",
               orion_qwen_lora_abs_sum(trainer.q_proj.a, trainer.q_proj.rank * trainer.q_proj.in_dim) +
               orion_qwen_lora_abs_sum(trainer.q_proj.b, trainer.q_proj.out_dim * trainer.q_proj.rank));
        printf("  v_param_abs_sum=%.6f\n",
               orion_qwen_lora_abs_sum(trainer.v_proj.a, trainer.v_proj.rank * trainer.v_proj.in_dim) +
               orion_qwen_lora_abs_sum(trainer.v_proj.b, trainer.v_proj.out_dim * trainer.v_proj.rank));
        printf("  next_blocker=%s\n", "adapter reload compare and smoke10 with ANE q/v base");
        ok = 1;

cleanup:
        if (ioIn) CFRelease(ioIn);
        if (ioQ) CFRelease(ioQ);
        if (ioK) CFRelease(ioK);
        if (ioV) CFRelease(ioV);
        if (progQ) orion_release_program(progQ);
        if (progKV) orion_release_program(progKV);
        free(hidden_in); free(normed); free(base_q); free(base_v); free(delta_q); free(delta_v);
        free(q_full); free(v_raw); free(attn_cat); free(gated); free(mixer); free(hidden_mid);
        free(post_norm); free(mlp_out); free(hidden_out); free(final_norm); free(last_hidden);
        free(d_last_hidden); free(d_hidden_out); free(d_post_norm); free(d_hidden_mid); free(d_mixer);
        free(d_gated); free(d_attn_cat); free(d_gate_half); free(d_q_full); free(d_v); free(d_normed_from_q);
        free(d_normed_from_v); free(d_normed); free(d_hidden_mid_from_post); free(d_post_ln_weight_grad);
        free(input_ln); free(post_ln); free(q_proj); free(v_proj); free(o_proj); free(gate_proj); free(up_proj); free(down_proj);
        free(final_norm_weight);
        orion_qwen9b_lora_trainer_free(&trainer);
        orion_qwen35_manifest_free(manifest);
        return ok ? 0 : 1;
    }
}
