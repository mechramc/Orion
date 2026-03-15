#import "weight_loader.h"
#import <stdio.h>
#import <stdlib.h>
#import <string.h>

// T031: Load GPT-2 weights from BLOBFILE blobs into fp32 arrays.
// BLOBFILE format: 128-byte header + fp16 data.
// We read fp16 and convert to fp32 in-place.

#pragma mark - BLOBFILE Reader

struct OrionBlobRowReader {
    FILE *file;
    int row_width;
    int row_count;
    int cache_row_start;
    int cache_row_count;
    int cache_capacity_rows;
    _Float16 *fp16_cache;
};

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

float* orion_read_blob_f32_exact(const char* path, int count) {
    return read_blob_f32(path, count);
}

int orion_blob_element_count(const char* path) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "weight_loader: cannot open %s\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fclose(f);
    if (file_size < 128) {
        fprintf(stderr, "weight_loader: %s: invalid file size %ld\n", path, file_size);
        return -1;
    }
    return (int)((file_size - 128) / 2);
}

int orion_read_blob_row_f32(const char* path, int row_index, int row_width, float* out_row) {
    if (!path || row_index < 0 || row_width <= 0 || !out_row) {
        fprintf(stderr, "weight_loader: invalid row read request\n");
        return 0;
    }

    int total_count = orion_blob_element_count(path);
    if (total_count < 0) {
        return 0;
    }
    if ((total_count % row_width) != 0) {
        fprintf(stderr, "weight_loader: %s: row_width %d does not divide %d elements\n",
                path, row_width, total_count);
        return 0;
    }

    int row_count = total_count / row_width;
    if (row_index >= row_count) {
        fprintf(stderr, "weight_loader: %s: row_index %d out of range (rows=%d)\n",
                path, row_index, row_count);
        return 0;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "weight_loader: cannot open %s\n", path);
        return 0;
    }

    long offset = 128L + ((long)row_index * (long)row_width * 2L);
    if (fseek(f, offset, SEEK_SET) != 0) {
        fprintf(stderr, "weight_loader: failed to seek %s\n", path);
        fclose(f);
        return 0;
    }

    _Float16 *fp16 = (_Float16 *)malloc((size_t)row_width * sizeof(_Float16));
    if (!fp16) {
        fclose(f);
        return 0;
    }
    size_t nread = fread(fp16, sizeof(_Float16), (size_t)row_width, f);
    fclose(f);
    if (nread != (size_t)row_width) {
        fprintf(stderr, "weight_loader: short read on %s\n", path);
        free(fp16);
        return 0;
    }

    for (int i = 0; i < row_width; i++) {
        out_row[i] = (float)fp16[i];
    }
    free(fp16);
    return 1;
}

OrionBlobRowReader* orion_blob_row_reader_open(const char* path, int row_width) {
    if (!path || row_width <= 0) {
        fprintf(stderr, "weight_loader: invalid row reader open request\n");
        return NULL;
    }

    int total_count = orion_blob_element_count(path);
    if (total_count < 0) {
        return NULL;
    }
    if ((total_count % row_width) != 0) {
        fprintf(stderr, "weight_loader: %s: row_width %d does not divide %d elements\n",
                path, row_width, total_count);
        return NULL;
    }

    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "weight_loader: cannot open %s\n", path);
        return NULL;
    }
    setvbuf(f, NULL, _IOFBF, 1 << 20);

    OrionBlobRowReader *reader = calloc(1, sizeof(OrionBlobRowReader));
    if (!reader) {
        fclose(f);
        return NULL;
    }

    const int row_bytes = row_width * (int)sizeof(_Float16);
    const int target_cache_bytes = 1 << 20;
    int cache_capacity_rows = target_cache_bytes / row_bytes;
    if (cache_capacity_rows < 1) cache_capacity_rows = 1;
    if (cache_capacity_rows > 256) cache_capacity_rows = 256;
    if (cache_capacity_rows > (total_count / row_width)) {
        cache_capacity_rows = total_count / row_width;
    }

    reader->fp16_cache = (_Float16 *)malloc((size_t)row_width * (size_t)cache_capacity_rows * sizeof(_Float16));
    if (!reader->fp16_cache) {
        fclose(f);
        free(reader);
        return NULL;
    }

    reader->file = f;
    reader->row_width = row_width;
    reader->row_count = total_count / row_width;
    reader->cache_row_start = -1;
    reader->cache_row_count = 0;
    reader->cache_capacity_rows = cache_capacity_rows;
    return reader;
}

int orion_blob_row_reader_read_f32(OrionBlobRowReader* reader, int row_index, float* out_row) {
    if (!reader || !reader->file || !out_row || row_index < 0 || row_index >= reader->row_count) {
        fprintf(stderr, "weight_loader: invalid reader row request idx=%d\n", row_index);
        return 0;
    }

    const int cache_row_end = reader->cache_row_start + reader->cache_row_count;
    if (!(reader->cache_row_start >= 0 &&
          row_index >= reader->cache_row_start &&
          row_index < cache_row_end)) {
        const int block_row_start = (row_index / reader->cache_capacity_rows) * reader->cache_capacity_rows;
        const int block_row_count = (reader->row_count - block_row_start) < reader->cache_capacity_rows
            ? (reader->row_count - block_row_start)
            : reader->cache_capacity_rows;

        long offset = 128L + ((long)block_row_start * (long)reader->row_width * 2L);
        if (fseek(reader->file, offset, SEEK_SET) != 0) {
            fprintf(stderr, "weight_loader: row reader seek failed\n");
            return 0;
        }

        const size_t nread = fread(reader->fp16_cache,
                                   sizeof(_Float16),
                                   (size_t)reader->row_width * (size_t)block_row_count,
                                   reader->file);
        if (nread != (size_t)reader->row_width * (size_t)block_row_count) {
            fprintf(stderr, "weight_loader: row reader short read\n");
            return 0;
        }

        reader->cache_row_start = block_row_start;
        reader->cache_row_count = block_row_count;
    }

    const int local_row = row_index - reader->cache_row_start;
    const _Float16 *fp16_row = reader->fp16_cache + ((size_t)local_row * (size_t)reader->row_width);
    for (int i = 0; i < reader->row_width; i++) {
        out_row[i] = (float)fp16_row[i];
    }
    return 1;
}

void orion_blob_row_reader_close(OrionBlobRowReader* reader) {
    if (!reader) return;
    if (reader->file) fclose(reader->file);
    free(reader->fp16_cache);
    free(reader);
}

#pragma mark - Qwen3.5 Manifest Loader

static int json_int(NSDictionary *dict, NSString *key) {
    id value = dict[key];
    if (![value respondsToSelector:@selector(intValue)]) {
        return 0;
    }
    return [value intValue];
}

static float json_float(NSDictionary *dict, NSString *key, float fallback) {
    id value = dict[key];
    if (![value respondsToSelector:@selector(floatValue)]) {
        return fallback;
    }
    return [value floatValue];
}

OrionQwen35Manifest* orion_qwen35_manifest_load(const char* blob_dir) {
    if (!blob_dir) {
        fprintf(stderr, "weight_loader: qwen35 manifest load got NULL blob_dir\n");
        return NULL;
    }

    @autoreleasepool {
        NSString *blobDir = [NSString stringWithUTF8String:blob_dir];
        NSString *manifestPath = [blobDir stringByAppendingPathComponent:@"manifest.json"];
        NSData *data = [NSData dataWithContentsOfFile:manifestPath];
        if (!data) {
            fprintf(stderr, "weight_loader: cannot read %s\n", manifestPath.UTF8String);
            return NULL;
        }

        NSError *error = nil;
        id root = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
        if (![root isKindOfClass:[NSDictionary class]]) {
            fprintf(stderr, "weight_loader: invalid manifest json: %s\n",
                    error.localizedDescription.UTF8String);
            return NULL;
        }

        NSDictionary *manifest = (NSDictionary *)root;
        NSString *status = manifest[@"status"];
        if (![status isEqualToString:@"PASS_QWEN35_MANIFEST_ONLY"] &&
            ![status isEqualToString:@"PASS_QWEN35_TEXT_ONLY_EXPORT"]) {
            fprintf(stderr, "weight_loader: unexpected manifest status %s\n",
                    status.UTF8String);
            return NULL;
        }

        NSArray *missing = manifest[@"missing_hf_names"];
        if ([missing isKindOfClass:[NSArray class]] && [missing count] > 0) {
            fprintf(stderr, "weight_loader: manifest still has missing tensors (%lu)\n",
                    (unsigned long)[missing count]);
            return NULL;
        }

        NSDictionary *runtime = manifest[@"runtime"];
        NSArray *entries = manifest[@"present_entries"];
        if (![runtime isKindOfClass:[NSDictionary class]] ||
            ![entries isKindOfClass:[NSArray class]]) {
            fprintf(stderr, "weight_loader: manifest missing runtime/present_entries\n");
            return NULL;
        }

        OrionQwen35Manifest *out = calloc(1, sizeof(OrionQwen35Manifest));
        out->n_layer = json_int(runtime, @"num_hidden_layers");
        out->d_model = json_int(runtime, @"hidden_size");
        out->d_ff = json_int(runtime, @"intermediate_size");
        out->n_head = json_int(runtime, @"num_attention_heads");
        out->n_kv_head = json_int(runtime, @"num_key_value_heads");
        out->head_dim = json_int(runtime, @"head_dim");
        out->vocab = json_int(runtime, @"vocab_size");
        out->max_seq = json_int(runtime, @"max_position_embeddings");
        out->tie_word_embeddings = json_int(runtime, @"tie_word_embeddings");
        NSDictionary *ropeParams = runtime[@"rope_parameters"];
        out->rope_theta = 10000000.0f;
        out->partial_rotary_factor = 0.25f;
        if ([ropeParams isKindOfClass:[NSDictionary class]]) {
            out->rope_theta = json_float(ropeParams, @"rope_theta", out->rope_theta);
            out->partial_rotary_factor = json_float(ropeParams, @"partial_rotary_factor", out->partial_rotary_factor);
        }
        out->rotary_dim = (int)(out->head_dim * out->partial_rotary_factor);
        if (out->rotary_dim % 2 != 0) {
            out->rotary_dim -= 1;
        }
        out->n_entries = (int)[entries count];
        out->blob_dir = strdup(blob_dir);
        out->manifest_path = strdup(manifestPath.UTF8String);

        NSArray *layerTypes = runtime[@"layer_types"];
        if ([layerTypes isKindOfClass:[NSArray class]]) {
            for (id value in layerTypes) {
                if (![value isKindOfClass:[NSString class]]) continue;
                NSString *layerType = (NSString *)value;
                if ([layerType isEqualToString:@"linear_attention"]) {
                    out->n_linear_layers += 1;
                } else if ([layerType isEqualToString:@"full_attention"]) {
                    out->n_full_layers += 1;
                }
            }
        }

        for (id value in entries) {
            if (![value isKindOfClass:[NSDictionary class]]) continue;
            NSDictionary *entry = (NSDictionary *)value;
            id aliasOf = entry[@"alias_of"];
            if ([aliasOf isKindOfClass:[NSString class]] && [(NSString *)aliasOf length] > 0) {
                out->n_alias_entries += 1;
            }
        }

        if (out->n_layer <= 0 || out->d_model <= 0 || out->vocab <= 0) {
            fprintf(stderr, "weight_loader: invalid qwen35 runtime dims from manifest\n");
            orion_qwen35_manifest_free(out);
            return NULL;
        }

        return out;
    }
}

#pragma mark - Load

OrionGPT2Weights* orion_gpt2_weights_load(const char* blob_dir) {
    // GPT-2 124M config
    int n_layer = 12;
    int d_model = 768;
    int d_ff = 3072;  // 4 * d_model
    int vocab = 50257;
    int max_seq = 1024;
    char path[1024];

    snprintf(path, sizeof(path), "%s/wte.bin", blob_dir);
    int wte_count = orion_blob_element_count(path);
    if (wte_count <= 0 || (wte_count % d_model) != 0) {
        fprintf(stderr, "weight_loader: failed to infer vocab from %s\n", path);
        return NULL;
    }
    vocab = wte_count / d_model;

    snprintf(path, sizeof(path), "%s/wpe.bin", blob_dir);
    int wpe_count = orion_blob_element_count(path);
    if (wpe_count <= 0 || (wpe_count % d_model) != 0) {
        fprintf(stderr, "weight_loader: failed to infer max_seq from %s\n", path);
        return NULL;
    }
    max_seq = wpe_count / d_model;

    OrionGPT2Weights *w = (OrionGPT2Weights *)calloc(1, sizeof(OrionGPT2Weights));
    w->n_layer = n_layer;
    w->d_model = d_model;
    w->d_ff = d_ff;
    w->vocab = vocab;
    w->max_seq = max_seq;
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

void orion_qwen35_manifest_free(OrionQwen35Manifest* manifest) {
    if (!manifest) return;
    free(manifest->blob_dir);
    free(manifest->manifest_path);
    free(manifest);
}
