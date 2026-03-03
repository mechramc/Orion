#ifndef ORION_CHECKPOINT_H
#define ORION_CHECKPOINT_H

#import "../kernels/training/stories_train.h"

// Checkpoint header — matches ANEgpt CkptHdr for compatibility.
typedef struct {
    int magic;              // ORION_CKPT_MAGIC (0x424C5A54 "BLZT")
    int version;            // ORION_CKPT_VERSION (2)
    int step;               // Current training step
    int total_steps;        // Total steps planned
    int n_layers;
    int vocab_size;
    int dim;
    int hidden_dim;
    int n_heads;
    int seq_len;
    float lr;
    float loss;             // Loss at checkpoint
    double cum_compile;     // Cumulative compile time (seconds)
    double cum_train;       // Cumulative training time (seconds)
    double cum_wall;        // Cumulative wall time (seconds)
    int cum_steps;          // Cumulative steps across restarts
    int cum_batches;        // Cumulative micro-batches
    int adam_t;             // Adam timestep
} OrionCkptHdr;

/// Save trainer state to checkpoint file.
/// Writes: header + per-layer weights + per-layer Adam state + embed + rms_final + Adam embed/rms.
/// @param trainer  The trainer to save
/// @param path     Output file path
/// @param step     Current training step number
/// @param loss     Loss at this checkpoint
/// @return true on success
bool orion_checkpoint_save(const OrionTrainer* trainer, const char* path,
                           int step, float loss);

/// Load checkpoint into an existing trainer.
/// The trainer must already be created (programs compiled, buffers allocated).
/// This overwrites weights and Adam state from the checkpoint.
/// @param trainer  Target trainer (must match config)
/// @param path     Checkpoint file path
/// @param out_step Outputs the step number from the checkpoint (may be NULL)
/// @param out_loss Outputs the loss from the checkpoint (may be NULL)
/// @return true on success, false if file missing or config mismatch
bool orion_checkpoint_load(OrionTrainer* trainer, const char* path,
                           int* out_step, float* out_loss);

#endif // ORION_CHECKPOINT_H
