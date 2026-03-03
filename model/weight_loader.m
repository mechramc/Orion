#import "weight_loader.h"
#import <stdio.h>
#import <stdlib.h>
#import <string.h>

// T031: Load GPT-2 weights from BLOBFILE blobs into fp32 arrays.
// BLOBFILE format: 128-byte header + fp16 data.
// We read fp16 and convert to fp32 in-place.

#pragma mark - BLOBFILE Reader

/// Read a BLOBFILE and return fp32 array. Caller must free.
/// @param path     Path to .bin blob file
/// @param count    Expected number of elements (for validation)
/// @return Heap-allocated fp32 array, or NULL on error.
static float* read_blob_f32(const char* path, int count) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "weight_loader: cannot open %s\n", path);
        return NULL;
    }

    // Skip 128-byte header
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 128, SEEK_SET);

    long data_size = file_size - 128;
    int fp16_count = (int)(data_size / 2);

    if (fp16_count != count) {
        fprintf(stderr, "weight_loader: %s: expected %d elements, got %d\n",
                path, count, fp16_count);
        fclose(f);
        return NULL;
    }

    // Read fp16 data
    _Float16 *fp16 = (_Float16 *)malloc(data_size);
    fread(fp16, 1, data_size, f);
    fclose(f);

    // Convert fp16 → fp32
    float *fp32 = (float *)malloc(count * sizeof(float));
    for (int i = 0; i < count; i++) {
        fp32[i] = (float)fp16[i];
    }
    free(fp16);

    return fp32;
}

#pragma mark - Load

OrionGPT2Weights* orion_gpt2_weights_load(const char* blob_dir) {
    // GPT-2 124M config
    int n_layer = 12;
    int d_model = 768;
    int d_ff = 3072;  // 4 * d_model
    int vocab = 50257;
    int max_seq = 1024;

    OrionGPT2Weights *w = (OrionGPT2Weights *)calloc(1, sizeof(OrionGPT2Weights));
    w->n_layer = n_layer;
    w->d_model = d_model;
    w->d_ff = d_ff;
    w->vocab = vocab;
    w->max_seq = max_seq;

    char path[1024];

    // Token embedding [vocab, d_model]
    snprintf(path, sizeof(path), "%s/wte.bin", blob_dir);
    w->wte = read_blob_f32(path, vocab * d_model);
    if (!w->wte) goto fail;

    // Position embedding [max_seq, d_model]
    snprintf(path, sizeof(path), "%s/wpe.bin", blob_dir);
    w->wpe = read_blob_f32(path, max_seq * d_model);
    if (!w->wpe) goto fail;

    // Per-layer weights
    w->layers = (OrionGPT2LayerWeights *)calloc(n_layer, sizeof(OrionGPT2LayerWeights));

    for (int i = 0; i < n_layer; i++) {
        OrionGPT2LayerWeights *l = &w->layers[i];

#define LOAD(field, name, cnt) do { \
    snprintf(path, sizeof(path), "%s/layer%d/%s", blob_dir, i, name); \
    l->field = read_blob_f32(path, cnt); \
    if (!l->field) goto fail; \
} while(0)

        LOAD(ln1_g, "ln1_g.bin", d_model);
        LOAD(ln1_b, "ln1_b.bin", d_model);
        LOAD(wq,    "wq.bin",    d_model * d_model);
        LOAD(wk,    "wk.bin",    d_model * d_model);
        LOAD(wv,    "wv.bin",    d_model * d_model);
        LOAD(bq,    "bq.bin",    d_model);
        LOAD(bk,    "bk.bin",    d_model);
        LOAD(bv,    "bv.bin",    d_model);
        LOAD(wo,    "wo.bin",    d_model * d_model);
        LOAD(bo,    "bo.bin",    d_model);
        LOAD(ln2_g, "ln2_g.bin", d_model);
        LOAD(ln2_b, "ln2_b.bin", d_model);
        LOAD(wfc,   "wfc.bin",   d_ff * d_model);
        LOAD(bfc,   "bfc.bin",   d_ff);
        LOAD(wproj, "wproj.bin", d_model * d_ff);
        LOAD(bproj, "bproj.bin", d_model);

#undef LOAD
    }

    // Final layer norm
    snprintf(path, sizeof(path), "%s/ln_f_g.bin", blob_dir);
    w->ln_f_g = read_blob_f32(path, d_model);
    if (!w->ln_f_g) goto fail;

    snprintf(path, sizeof(path), "%s/ln_f_b.bin", blob_dir);
    w->ln_f_b = read_blob_f32(path, d_model);
    if (!w->ln_f_b) goto fail;

    return w;

fail:
    orion_gpt2_weights_free(w);
    return NULL;
}

#pragma mark - Free

void orion_gpt2_weights_free(OrionGPT2Weights* w) {
    if (!w) return;
    free(w->wte);
    free(w->wpe);
    free(w->ln_f_g);
    free(w->ln_f_b);
    if (w->layers) {
        for (int i = 0; i < w->n_layer; i++) {
            OrionGPT2LayerWeights *l = &w->layers[i];
            free(l->ln1_g); free(l->ln1_b);
            free(l->wq); free(l->wk); free(l->wv);
            free(l->bq); free(l->bk); free(l->bv);
            free(l->wo); free(l->bo);
            free(l->ln2_g); free(l->ln2_b);
            free(l->wfc); free(l->bfc);
            free(l->wproj); free(l->bproj);
        }
        free(w->layers);
    }
    free(w);
}
