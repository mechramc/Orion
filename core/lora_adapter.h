// lora_adapter.h — LoRA adapter loading and IOSurface management (T159)
//
// Loads LoRA adapter matrices (A, B per projection) from BLOBFILE format
// into IOSurface tensors for ANE evaluation.
//
// ANE constraints:
//   - All input IOSurfaces must have the same allocation size (uniform alloc)
//   - Input ordering is alphabetical by MIL parameter name
//   - Flat buffer is read as packed [1,C,1,S] data (no stride)
//
// Adapter directory layout:
//   adapter_dir/
//     config.txt          — "rank=16 alpha=16.0 targets=q,v"
//     layer0/
//       lora_q_A.bin      — [d_model, rank] BLOBFILE (128-byte header + fp16)
//       lora_q_B.bin      — [d_model, rank] BLOBFILE
//       lora_v_A.bin      — [d_model, rank] BLOBFILE
//       lora_v_B.bin      — [d_model, rank] BLOBFILE
//     layer1/
//       ...

#ifndef ORION_LORA_ADAPTER_H
#define ORION_LORA_ADAPTER_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

/// Per-layer LoRA adapter surfaces.
/// Each pair (A, B) is an IOSurface tensor. NULL if not applied to that projection.
/// A shape: [1, d_model, 1, rank] packed in surface allocated for [1, d_model, 1, seq]
/// B shape: [1, d_model, 1, rank] packed in surface allocated for [1, d_model, 1, seq]
typedef struct {
    IOSurfaceRef q_A;
    IOSurfaceRef q_B;
    IOSurfaceRef k_A;
    IOSurfaceRef k_B;
    IOSurfaceRef v_A;
    IOSurfaceRef v_B;
    IOSurfaceRef o_A;
    IOSurfaceRef o_B;
} OrionLoRALayerSurfaces;

/// Full LoRA adapter: config + per-layer IOSurface tensors.
typedef struct {
    int rank;
    float alpha;
    bool apply_q, apply_k, apply_v, apply_o;
    int n_layer;
    int d_model;
    int seq;           // Sequence length for uniform IOSurface allocation
    OrionLoRALayerSurfaces *layers;  // [n_layer]
} OrionLoRAAdapter;

/// Load a LoRA adapter from directory.
/// @param path     Path to adapter directory containing config.txt and layerN/ subdirs
/// @param d_model  Model dimension (for validation)
/// @param n_layer  Number of layers
/// @param seq      Sequence length (for uniform IOSurface allocation — must match input x)
/// @return Loaded adapter, or NULL on error. Caller must free with orion_lora_free.
OrionLoRAAdapter* orion_lora_load(const char* path, int d_model, int n_layer, int seq);

/// Free a loaded LoRA adapter (releases all IOSurfaces).
void orion_lora_free(OrionLoRAAdapter* adapter);

/// Create a zero LoRA adapter (for testing — all surfaces zero-filled).
/// Useful as baseline: output should match base model exactly.
OrionLoRAAdapter* orion_lora_create_zero(int rank, float alpha,
                                          bool apply_q, bool apply_k,
                                          bool apply_v, bool apply_o,
                                          int d_model, int n_layer, int seq);

#endif // ORION_LORA_ADAPTER_H
