#ifndef ORION_WEIGHT_LOADER_H
#define ORION_WEIGHT_LOADER_H

#import <Foundation/Foundation.h>

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

/// Load GPT-2 weights from BLOBFILE directory.
/// @param blob_dir Path to directory containing blob files (e.g., model/blobs/gpt2_124m/)
/// @return Loaded weights, or NULL on failure. Caller must free with orion_gpt2_weights_free.
OrionGPT2Weights* orion_gpt2_weights_load(const char* blob_dir);

/// Free loaded weights.
void orion_gpt2_weights_free(OrionGPT2Weights* w);

#endif // ORION_WEIGHT_LOADER_H
