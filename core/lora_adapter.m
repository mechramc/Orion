// lora_adapter.m — LoRA adapter loading and IOSurface management (T159)

#import "lora_adapter.h"
#import "iosurface_tensor.h"
#import <stdio.h>
#import <stdlib.h>
#import <string.h>

#pragma mark - BLOBFILE Reader

/// Read BLOBFILE (128-byte header + fp16) into an IOSurface.
/// Surface is allocated with uniform_seq for ANE compatibility,
/// but data is written packed as [channels * rank] elements.
static IOSurfaceRef read_blob_to_surface(const char* path,
                                          int channels, int rank,
                                          int uniform_seq) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "lora_adapter: cannot open %s\n", path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 128, SEEK_SET);

    long data_size = file_size - 128;
    int expected = channels * rank;
    int fp16_count = (int)(data_size / 2);

    if (fp16_count != expected) {
        fprintf(stderr, "lora_adapter: %s: expected %d elements, got %d\n",
                path, expected, fp16_count);
        fclose(f);
        return NULL;
    }

    // Read fp16, convert to fp32 for writing via orion_tensor_write_f32
    _Float16 *fp16 = (_Float16 *)malloc(data_size);
    fread(fp16, 1, data_size, f);
    fclose(f);

    float *fp32 = (float *)malloc(expected * sizeof(float));
    for (int i = 0; i < expected; i++) {
        fp32[i] = (float)fp16[i];
    }
    free(fp16);

    // Allocate surface with uniform_seq for ANE uniform alloc constraint
    IOSurfaceRef surf = orion_tensor_create(channels, uniform_seq);
    if (!surf) {
        free(fp32);
        return NULL;
    }

    // Write packed [channels * rank] data at start of surface
    orion_tensor_write_f32(surf, fp32, expected);
    free(fp32);

    return surf;
}

#pragma mark - Config Parser

/// Parse config.txt: "rank=16 alpha=16.0 targets=q,v"
static bool parse_config(const char* path, int* rank, float* alpha,
                          bool* apply_q, bool* apply_k,
                          bool* apply_v, bool* apply_o) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "lora_adapter: cannot open config %s\n", path);
        return false;
    }

    *rank = 0; *alpha = 0;
    *apply_q = *apply_k = *apply_v = *apply_o = false;

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "rank=%d", rank) == 1) continue;
        if (sscanf(line, "alpha=%f", alpha) == 1) continue;
        char targets[128];
        if (sscanf(line, "targets=%127s", targets) == 1) {
            if (strstr(targets, "q")) *apply_q = true;
            if (strstr(targets, "k")) *apply_k = true;
            if (strstr(targets, "v")) *apply_v = true;
            if (strstr(targets, "o")) *apply_o = true;
        }
    }
    fclose(f);

    if (*rank <= 0 || *alpha <= 0) {
        fprintf(stderr, "lora_adapter: invalid config rank=%d alpha=%.2f\n",
                *rank, *alpha);
        return false;
    }
    return true;
}

#pragma mark - Load

OrionLoRAAdapter* orion_lora_load(const char* path, int d_model,
                                   int n_layer, int seq) {
    char config_path[512];
    snprintf(config_path, sizeof(config_path), "%s/config.txt", path);

    int rank;
    float alpha;
    bool apply_q, apply_k, apply_v, apply_o;
    if (!parse_config(config_path, &rank, &alpha,
                       &apply_q, &apply_k, &apply_v, &apply_o)) {
        return NULL;
    }

    OrionLoRAAdapter *adapter = (OrionLoRAAdapter *)calloc(1, sizeof(OrionLoRAAdapter));
    adapter->rank = rank;
    adapter->alpha = alpha;
    adapter->apply_q = apply_q;
    adapter->apply_k = apply_k;
    adapter->apply_v = apply_v;
    adapter->apply_o = apply_o;
    adapter->n_layer = n_layer;
    adapter->d_model = d_model;
    adapter->seq = seq;
    adapter->layers = (OrionLoRALayerSurfaces *)calloc(n_layer,
                                                        sizeof(OrionLoRALayerSurfaces));

    char fpath[512];
    for (int i = 0; i < n_layer; i++) {
        OrionLoRALayerSurfaces *l = &adapter->layers[i];

#define LOAD_PAIR(proj, flag, field_A, field_B) do { \
    if (flag) { \
        snprintf(fpath, sizeof(fpath), "%s/layer%d/lora_%s_A.bin", path, i, proj); \
        l->field_A = read_blob_to_surface(fpath, d_model, rank, seq); \
        if (!l->field_A) goto fail; \
        snprintf(fpath, sizeof(fpath), "%s/layer%d/lora_%s_B.bin", path, i, proj); \
        l->field_B = read_blob_to_surface(fpath, d_model, rank, seq); \
        if (!l->field_B) goto fail; \
    } \
} while(0)

        LOAD_PAIR("q", apply_q, q_A, q_B);
        LOAD_PAIR("k", apply_k, k_A, k_B);
        LOAD_PAIR("v", apply_v, v_A, v_B);
        LOAD_PAIR("o", apply_o, o_A, o_B);

#undef LOAD_PAIR
    }

    return adapter;

fail:
    orion_lora_free(adapter);
    return NULL;
}

#pragma mark - Free

void orion_lora_free(OrionLoRAAdapter* adapter) {
    if (!adapter) return;
    if (adapter->layers) {
        for (int i = 0; i < adapter->n_layer; i++) {
            OrionLoRALayerSurfaces *l = &adapter->layers[i];
            if (l->q_A) CFRelease(l->q_A);
            if (l->q_B) CFRelease(l->q_B);
            if (l->k_A) CFRelease(l->k_A);
            if (l->k_B) CFRelease(l->k_B);
            if (l->v_A) CFRelease(l->v_A);
            if (l->v_B) CFRelease(l->v_B);
            if (l->o_A) CFRelease(l->o_A);
            if (l->o_B) CFRelease(l->o_B);
        }
        free(adapter->layers);
    }
    free(adapter);
}

#pragma mark - Zero Adapter

OrionLoRAAdapter* orion_lora_create_zero(int rank, float alpha,
                                          bool apply_q, bool apply_k,
                                          bool apply_v, bool apply_o,
                                          int d_model, int n_layer, int seq) {
    OrionLoRAAdapter *adapter = (OrionLoRAAdapter *)calloc(1, sizeof(OrionLoRAAdapter));
    adapter->rank = rank;
    adapter->alpha = alpha;
    adapter->apply_q = apply_q;
    adapter->apply_k = apply_k;
    adapter->apply_v = apply_v;
    adapter->apply_o = apply_o;
    adapter->n_layer = n_layer;
    adapter->d_model = d_model;
    adapter->seq = seq;
    adapter->layers = (OrionLoRALayerSurfaces *)calloc(n_layer,
                                                        sizeof(OrionLoRALayerSurfaces));

    // Zero-fill packed data: [d_model * rank] elements
    int packed = d_model * rank;
    float *zeros = (float *)calloc(packed, sizeof(float));

    for (int i = 0; i < n_layer; i++) {
        OrionLoRALayerSurfaces *l = &adapter->layers[i];

#define ZERO_PAIR(flag, field_A, field_B) do { \
    if (flag) { \
        l->field_A = orion_tensor_create(d_model, seq); \
        orion_tensor_write_f32(l->field_A, zeros, packed); \
        l->field_B = orion_tensor_create(d_model, seq); \
        orion_tensor_write_f32(l->field_B, zeros, packed); \
    } \
} while(0)

        ZERO_PAIR(apply_q, q_A, q_B);
        ZERO_PAIR(apply_k, k_A, k_B);
        ZERO_PAIR(apply_v, v_A, v_B);
        ZERO_PAIR(apply_o, o_A, o_B);

#undef ZERO_PAIR
    }

    free(zeros);
    return adapter;
}
