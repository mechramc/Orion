#ifndef ORION_WEIGHT_LOADER_H
#define ORION_WEIGHT_LOADER_H

#import <Foundation/Foundation.h>
#include <stdio.h>

/// GPT-2 124M layer weights (fp32, loaded from BLOBFILE blobs).
typedef struct {
    float *ln1_g;    // [d_model] LayerNorm1 gamma
    float *ln1_b;    // [d_model] LayerNorm1 beta
    float *wq;       // [d_model, d_model] Q projection (transposed for matmul)
    float *wk;       // [d_model, d_model] K projection
    float *wv;       // [d_model, d_model] V projection
    float *bq;       // [d_model] Q bias
    float *bk;       // [d_model] K bias
    float *bv;       // [d_model] V bias
    float *wo;       // [d_model, d_model] output projection
    float *bo;       // [d_model] output bias
    float *ln2_g;    // [d_model] LayerNorm2 gamma
    float *ln2_b;    // [d_model] LayerNorm2 beta
    float *wfc;      // [d_ff, d_model] FFN up (transposed)
    float *bfc;      // [d_ff] FFN up bias
    float *wproj;    // [d_model, d_ff] FFN down (transposed)
    float *bproj;    // [d_model] FFN down bias
} OrionGPT2LayerWeights;

/// Full GPT-2 model weights.
typedef struct {
    int n_layer;
    int d_model;
    int d_ff;
    int vocab;
    int max_seq;
    float *wte;      // [vocab, d_model] token embedding
    float *wpe;      // [max_seq, d_model] position embedding
    OrionGPT2LayerWeights *layers; // [n_layer]
    float *ln_f_g;   // [d_model] final LayerNorm gamma
    float *ln_f_b;   // [d_model] final LayerNorm beta
} OrionGPT2Weights;

/// Lightweight Qwen3.5 text-only runtime metadata loaded from manifest.json.
/// This is a stage-1 porting structure used to validate that converted blobs
/// and runtime dimensions are coherent before full CPU-only inference exists.
typedef struct {
    int n_layer;
    int d_model;
    int d_ff;
    int n_head;
    int n_kv_head;
    int head_dim;
    int vocab;
    int max_seq;
    int tie_word_embeddings;
    float rope_theta;
    float partial_rotary_factor;
    int rotary_dim;
    int n_entries;
    int n_alias_entries;
    int n_linear_layers;
    int n_full_layers;
    char *blob_dir;
    char *manifest_path;
} OrionQwen35Manifest;

typedef struct OrionBlobRowReader OrionBlobRowReader;

/// Load GPT-2 weights from BLOBFILE directory.
/// @param blob_dir Path to directory containing blob files (e.g., model/blobs/gpt2_124m/)
/// @return Loaded weights, or NULL on failure. Caller must free with orion_gpt2_weights_free.
OrionGPT2Weights* orion_gpt2_weights_load(const char* blob_dir);

/// Free loaded weights.
void orion_gpt2_weights_free(OrionGPT2Weights* w);

/// Load stage-1 Qwen3.5 runtime metadata from manifest.json.
/// @param blob_dir Path to converted blob directory that contains manifest.json.
/// @return Loaded manifest metadata, or NULL on failure. Caller must free with
///         orion_qwen35_manifest_free.
OrionQwen35Manifest* orion_qwen35_manifest_load(const char* blob_dir);

/// Free loaded Qwen3.5 manifest metadata.
void orion_qwen35_manifest_free(OrionQwen35Manifest* manifest);

/// Return the number of fp16 elements stored in a BLOBFILE.
/// Returns -1 on error.
int orion_blob_element_count(const char* path);

/// Read a contiguous fp16 row from a BLOBFILE and convert it to fp32.
/// The tensor is treated as a row-major 2D matrix with width `row_width`.
/// Vectors can be read by passing row_index=0 and row_width=<vector length>.
/// Returns 1 on success, 0 on failure.
int orion_read_blob_row_f32(const char* path, int row_index, int row_width, float* out_row);

/// Open a buffered row reader for repeated fp16 -> fp32 row access.
/// Returns NULL on failure.
OrionBlobRowReader* orion_blob_row_reader_open(const char* path, int row_width);

/// Read a row via an open row reader.
/// Returns 1 on success, 0 on failure.
int orion_blob_row_reader_read_f32(OrionBlobRowReader* reader, int row_index, float* out_row);

/// Close a row reader and release its buffers.
void orion_blob_row_reader_close(OrionBlobRowReader* reader);

/// Read a full BLOBFILE tensor into fp32 with an exact expected element count.
/// Returns heap-allocated fp32 data on success, or NULL on failure.
float* orion_read_blob_f32_exact(const char* path, int count);

#endif // ORION_WEIGHT_LOADER_H
