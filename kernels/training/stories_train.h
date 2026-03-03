#ifndef ORION_STORIES_TRAIN_H
#define ORION_STORIES_TRAIN_H

#import "../../core/ane_runtime.h"
#import "../../core/iosurface_tensor.h"
#import "../../model/configs/stories110m.h"
#import "stories_cpu_ops.h"

// T072: Single training step for Stories110M on ANE.
// Forward (ANE) → Loss (CPU) → Backward (ANE) → dW (CPU) → Adam (CPU)
//
// Per-layer data flow:
//   fwdAttn:  x → [wo_out, q_out, k_out, v_out, attn_out, rms1_out]
//   residual: x2 = x + wo_out  (CPU)
//   fwdFFN:   x2 → [w2_out, w1_out, w3_out, gate, rms2_out]
//   residual: x_next = x2 + w2_out  (CPU)
//
// Backward (reverse order per layer):
//   ffnBwd:   [dy, h1, h3] → [dx, dh1, dh3]
//   rmsnorm2_bwd: dx_ffn, x2 → dx2  (CPU)
//   sdpaBwd1: [qf, kf, vf, dx2] → [dvf, pf, dpf]
//   sdpaBwd2: [pf, dpf, qf, kf] → [dqf, dkf]
//   qkvBwd:   [dq, dk, dv] → dx_attn
//   rmsnorm1_bwd: dx_attn, x → dx_rms1  (CPU)
//   dy = dx_rms1 + dx2  (residual merge)

/// Per-layer ANE kernel programs.
typedef struct {
    OrionProgram *fwd_attn;   // 6 outputs
    OrionProgram *fwd_ffn;    // 5 outputs
    OrionProgram *ffn_bwd;    // 3 outputs
    OrionProgram *sdpa_bwd1;  // 3 outputs
    OrionProgram *sdpa_bwd2;  // 2 outputs
    OrionProgram *qkv_bwd;    // 1 output
} OrionLayerKernels;

/// Per-layer weight arrays (fp32, CPU-side).
typedef struct {
    float *rms_att;    // [dim]
    float *wq;         // [dim * dim]
    float *wk;         // [dim * dim]
    float *wv;         // [dim * dim]
    float *wo;         // [dim * dim]
    float *rms_ffn;    // [dim]
    float *w1;         // [hidden * dim]
    float *w3;         // [hidden * dim]
    float *w2;         // [dim * hidden]
} OrionLayerWeights;

/// Per-layer gradient arrays (fp32, CPU-side).
typedef struct {
    float *drms_att;   // [dim]
    float *dwq;        // [dim * dim]
    float *dwk;        // [dim * dim]
    float *dwv;        // [dim * dim]
    float *dwo;        // [dim * dim]
    float *drms_ffn;   // [dim]
    float *dw1;        // [hidden * dim]
    float *dw3;        // [hidden * dim]
    float *dw2;        // [dim * hidden]
} OrionLayerGrads;

/// Per-layer Adam optimizer state.
typedef struct {
    float *m_rms_att, *v_rms_att;
    float *m_wq, *v_wq;
    float *m_wk, *v_wk;
    float *m_wv, *v_wv;
    float *m_wo, *v_wo;
    float *m_rms_ffn, *v_rms_ffn;
    float *m_w1, *v_w1;
    float *m_w3, *v_w3;
    float *m_w2, *v_w2;
} OrionLayerAdam;

/// Per-layer IOSurface buffers for ANE kernel I/O.
typedef struct {
    // Forward inputs (written from CPU)
    IOSurfaceRef fwd_attn_in;     // [1, d, 1, s]
    IOSurfaceRef fwd_ffn_in;      // [1, d, 1, s]

    // Forward outputs (6 + 5 separate IOSurfaces)
    IOSurfaceRef fwd_attn_out[6]; // wo_out, q_out, k_out, v_out, attn_out, rms1_out
    IOSurfaceRef fwd_ffn_out[5];  // w2_out, w1_out, w3_out, gate, rms2_out

    // Backward inputs (concatenated, written from forward taps)
    IOSurfaceRef ffn_bwd_in;      // [1, d+2h, 1, s]
    IOSurfaceRef sdpa_bwd1_in;    // [1, 4d, 1, s]
    IOSurfaceRef sdpa_bwd2_in;    // [1, 2*sc+2d, 1, s]
    IOSurfaceRef qkv_bwd_in;     // [1, 3d, 1, s]

    // Backward outputs
    IOSurfaceRef ffn_bwd_out[3];  // dx, dh1, dh3
    IOSurfaceRef sdpa_bwd1_out[3]; // dvf, pf, dpf
    IOSurfaceRef sdpa_bwd2_out[2]; // dqf, dkf
    IOSurfaceRef qkv_bwd_out;    // dx (single)
} OrionLayerIO;

/// Full trainer state.
typedef struct {
    const OrionModelConfig *cfg;
    int n_layers;

    // Per-layer state
    OrionLayerKernels *kernels;
    OrionLayerWeights *weights;
    OrionLayerGrads   *grads;
    OrionLayerAdam    *adam;
    OrionLayerIO      *io;

    // Embedding + final RMSNorm
    float *embed;          // [vocab * dim]
    float *rms_final;      // [dim]
    float *dembed;         // [vocab * dim]  gradient
    float *drms_final;     // [dim]  gradient

    // Adam for embed + rms_final
    float *m_embed, *v_embed;
    float *m_rms_final, *v_rms_final;

    // Training state
    int adam_t;            // Adam timestep
    float lr;
    float beta1, beta2, eps;

    // CPU activation buffers (fp32, [seq * dim])
    float *act_x;         // per-layer input activations [n_layers+1][seq*dim]
    float *act_x2;        // per-layer x2 = x + attn_out [n_layers][seq*dim]
    float *logits;        // [seq * vocab]
    float *dlogits;       // [seq * vocab]
} OrionTrainer;

/// Create trainer: compile all ANE programs, allocate buffers.
/// weight_path: directory containing layer0/, layer1/, ..., embed.bin, rms_final.bin
/// Returns NULL on failure.
OrionTrainer* orion_trainer_create(const OrionModelConfig* cfg, const char* weight_path);

/// Run one training step. Returns loss value.
/// input_tokens: [seq_len] token IDs
/// target_tokens: [seq_len] target token IDs (input shifted by 1)
float orion_train_step(OrionTrainer* trainer,
                        const int* input_tokens, const int* target_tokens);

/// Apply Adam update to all weights using accumulated gradients.
/// Call after gradient accumulation is complete.
void orion_trainer_adam_update(OrionTrainer* trainer);

/// Zero all gradient buffers (call before each accumulation batch).
void orion_trainer_zero_grads(OrionTrainer* trainer);

/// Scale all gradient buffers by a constant (e.g. 1.0/accum_steps).
/// Call after accumulating N micro-batches, before Adam update.
void orion_trainer_scale_grads(OrionTrainer* trainer, float scale);

/// Recompile all ANE programs with current CPU-side weights (Mode A).
/// Must be called after orion_trainer_adam_update to bake updated weights
/// into ANE programs. Returns false if any compile fails.
bool orion_trainer_recompile(OrionTrainer* trainer, const char* weight_path);

/// Check if we're approaching the ANE compile limit (~119 per process).
/// Returns true if a restart is recommended before the next recompile.
/// budget_remaining: if non-NULL, set to remaining compiles before limit.
bool orion_trainer_needs_restart(const OrionTrainer* trainer, int* budget_remaining);

/// Free all trainer resources.
void orion_trainer_free(OrionTrainer* trainer);

#endif // ORION_STORIES_TRAIN_H
