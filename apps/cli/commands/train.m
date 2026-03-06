#import <Foundation/Foundation.h>
#import <stdio.h>
#import <string.h>
#import <unistd.h>
#import <mach/mach_time.h>

#import "kernels/training/stories_train.h"
#import "kernels/training/data_loader.h"
#import "core/checkpoint.h"
#import "model/configs/stories110m.h"

// T079: CLI arg parsing for train
// T080: Wire orion train E2E
// T081: Training profiler

typedef struct {
    const char *weight_path;    // --weights
    const char *data_path;      // --dataset
    const char *checkpoint_dir; // --checkpoint_dir
    const char *resume;         // --resume (checkpoint to resume from)
    const char *csv_path;       // --csv (per-step CSV log)
    int steps;                  // --steps (total steps)
    int grad_accum;             // --grad_accum
    int checkpoint_every;       // --checkpoint_every
    float lr;                   // --lr
    int seq_len;                // --seq (override config)
    bool help;
} TrainArgs;

static void print_train_usage(void) {
    fprintf(stderr,
        "Usage: orion train [options]\n"
        "\n"
        "Options:\n"
        "  --weights DIR       Weight directory (BLOBFILE format)\n"
        "  --dataset FILE      Pretokenized data file (uint16 binary)\n"
        "  --checkpoint_dir D  Directory for checkpoint files (default: ./checkpoints)\n"
        "  --resume FILE       Resume from checkpoint file\n"
        "  --steps N           Total training steps (default: 100)\n"
        "  --grad_accum N      Gradient accumulation steps (default: %d)\n"
        "  --checkpoint_every N Save checkpoint every N steps (default: 25)\n"
        "  --lr F              Learning rate (default: 3e-4)\n"
        "  --seq N             Sequence length (default: %d)\n"
        "  --help              Show this help\n",
        STORIES_ACCUM_STEPS, kStories110M.max_seq
    );
}

static TrainArgs parse_train_args(int argc, const char* argv[]) {
    TrainArgs args = {
        .weight_path = NULL,
        .data_path = NULL,
        .checkpoint_dir = "./checkpoints",
        .resume = NULL,
        .csv_path = NULL,
        .steps = 100,
        .grad_accum = STORIES_ACCUM_STEPS,
        .checkpoint_every = 25,
        .lr = 3e-4f,
        .seq_len = kStories110M.max_seq,
        .help = false,
    };

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            args.help = true;
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            args.weight_path = argv[++i];
        } else if (strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
            args.data_path = argv[++i];
        } else if (strcmp(argv[i], "--checkpoint_dir") == 0 && i + 1 < argc) {
            args.checkpoint_dir = argv[++i];
        } else if (strcmp(argv[i], "--resume") == 0 && i + 1 < argc) {
            args.resume = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            args.steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--grad_accum") == 0 && i + 1 < argc) {
            args.grad_accum = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--checkpoint_every") == 0 && i + 1 < argc) {
            args.checkpoint_every = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            args.lr = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            args.seq_len = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
            args.csv_path = argv[++i];
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            args.help = true;
        }
    }
    return args;
}

static double time_seconds(void) {
    static mach_timebase_info_data_t tb;
    if (tb.denom == 0) mach_timebase_info(&tb);
    return (double)mach_absolute_time() * tb.numer / tb.denom / 1e9;
}

/// Estimate FLOPS per training step for a Llama2-style model.
/// Forward: ~2 * params * seq_len (matmuls dominate)
/// Backward: ~4 * params * seq_len (2x forward for dx + dW)
/// Total: ~6 * params * seq_len per micro-batch
static double estimate_step_flops(const OrionModelConfig *cfg, int grad_accum) {
    // Parameter count: embedding + n_layer*(attn_qkv + attn_out + ffn_up + ffn_down) + final_rms
    int64_t d = cfg->d_model;
    int64_t h = cfg->hidden_dim;
    int64_t v = cfg->vocab;
    int64_t s = cfg->max_seq;
    int64_t L = cfg->n_layer;

    int64_t params_per_layer = 4*d*d + 2*d*h;  // qkv_proj + out_proj + ffn_up + ffn_down
    int64_t total_params = v*d + L*params_per_layer;  // embed + layers

    // 6 * params * seq_len per micro-batch (forward + backward)
    return 6.0 * (double)total_params * (double)s * (double)grad_accum;
}

int orion_cmd_train(int argc, const char* argv[]) {
    @autoreleasepool {
        TrainArgs args = parse_train_args(argc, argv);
        if (args.help) {
            print_train_usage();
            return args.help ? 0 : 1;
        }

        if (!args.weight_path) {
            fprintf(stderr, "Error: --weights is required\n");
            print_train_usage();
            return 1;
        }
        if (!args.data_path) {
            fprintf(stderr, "Error: --dataset is required\n");
            print_train_usage();
            return 1;
        }

        // Create config (use Stories110M as base)
        const OrionModelConfig *cfg = &kStories110M;

        // Create trainer
        double t0 = time_seconds();
        OrionTrainer *trainer;
        int start_step = 0;
        float last_loss = 0.0f;

        if (args.resume) {
            // Deferred path: allocate + load weights, skip ANE compile.
            // Checkpoint will overwrite CPU weights, then recompile bakes
            // the correct (post-Adam) weights into ANE programs.
            // This uses 72 compiles instead of 144 (which exceeds ~119 limit).
            fprintf(stderr, "Creating trainer (deferred compile for resume)...\n");
            trainer = orion_trainer_create_deferred(cfg, args.weight_path);
            if (!trainer) {
                fprintf(stderr, "Error: failed to create trainer\n");
                return 1;
            }

            fprintf(stderr, "Resuming from %s...\n", args.resume);
            if (!orion_checkpoint_load(trainer, args.resume, &start_step, &last_loss)) {
                fprintf(stderr, "Error: failed to load checkpoint %s\n", args.resume);
                orion_trainer_free(trainer);
                return 1;
            }
            fprintf(stderr, "Resumed at step %d, loss=%.4f, adam_t=%d\n",
                    start_step, last_loss, trainer->adam_t);

            // Write checkpoint weights to BLOBFILEs, then compile with correct weights
            fprintf(stderr, "Compiling %d ANE programs with resumed weights...\n",
                    cfg->n_layer * 6);
            if (!orion_trainer_recompile(trainer, args.weight_path)) {
                fprintf(stderr, "Error: failed to compile ANE programs after resume\n");
                orion_trainer_free(trainer);
                return 1;
            }

        } else {
            // Normal path: compile immediately with on-disk weights
            fprintf(stderr, "Creating trainer (compiling %d ANE programs)...\n",
                    cfg->n_layer * 6);
            trainer = orion_trainer_create(cfg, args.weight_path);
            if (!trainer) {
                fprintf(stderr, "Error: failed to create trainer\n");
                return 1;
            }
        }

        double compile_time = time_seconds() - t0;
        fprintf(stderr, "Compiled in %.1f s\n", compile_time);

        trainer->lr = args.lr;

        // Open data loader
        OrionDataLoader *dl = orion_data_loader_open(args.data_path, cfg->max_seq);
        if (!dl) {
            fprintf(stderr, "Error: failed to open dataset %s\n", args.data_path);
            orion_trainer_free(trainer);
            return 1;
        }
        fprintf(stderr, "Dataset: %lld tokens, %lld samples\n",
                dl->n_tokens, orion_data_loader_num_samples(dl));

        // Create checkpoint directory
        [[NSFileManager defaultManager] createDirectoryAtPath:@(args.checkpoint_dir)
                                  withIntermediateDirectories:YES attributes:nil error:nil];

        // Allocate token buffers
        int *input = (int *)malloc(cfg->max_seq * sizeof(int));
        int *target = (int *)malloc(cfg->max_seq * sizeof(int));

        // Profiling accumulators
        double total_train_ms = 0;   // fwd+bwd+dW+Adam time
        double total_recompile_ms = 0;
        double step_flops = estimate_step_flops(cfg, args.grad_accum);
        int steps_completed = 0;

        // CSV log
        FILE *csv = NULL;
        if (args.csv_path) {
            csv = fopen(args.csv_path, "w");
            if (csv) fprintf(csv, "step,loss,train_ms,reload_ms,total_ms,tflops\n");
        }

        // Training loop
        double wall_start = time_seconds();
        fprintf(stderr, "\nTraining: steps=%d, grad_accum=%d, lr=%.1e, checkpoint_every=%d\n\n",
                args.steps, args.grad_accum, args.lr, args.checkpoint_every);

        for (int step = start_step; step < args.steps; step++) {
            double step_start = time_seconds();

            // Gradient accumulation
            orion_trainer_zero_grads(trainer);
            float step_loss = 0.0f;
            for (int mb = 0; mb < args.grad_accum; mb++) {
                if (!orion_data_loader_next(dl, input, target)) {
                    orion_data_loader_reset(dl);
                    orion_data_loader_next(dl, input, target);
                }

                float loss = orion_train_step(trainer, input, target);
                step_loss += loss;
            }
            step_loss /= (float)args.grad_accum;
            orion_trainer_scale_grads(trainer, 1.0f / (float)args.grad_accum);
            orion_trainer_adam_update(trainer);

            double train_time = time_seconds() - step_start;
            total_train_ms += train_time * 1000.0;

            last_loss = step_loss;
            steps_completed++;

            // Print progress with TFLOPS
            double step_tflops = step_flops / (train_time * 1e12);
            fprintf(stderr, "step %4d | loss %.4f | %.1f ms | %.3f TFLOPS\n",
                    step + 1, step_loss, train_time * 1000.0, step_tflops);

            // Checkpoint
            if (args.checkpoint_every > 0 && (step + 1) % args.checkpoint_every == 0) {
                char ckpt_path[512];
                snprintf(ckpt_path, sizeof(ckpt_path), "%s/ckpt_%05d.bin",
                         args.checkpoint_dir, step + 1);
                if (orion_checkpoint_save(trainer, ckpt_path, step + 1, step_loss)) {
                    fprintf(stderr, "  Saved checkpoint: %s\n", ckpt_path);
                }
            }

            // T154: Compile budget check removed — delta recompile (T152) skips
            // the ANE compiler entirely, so the ~119 compile limit no longer applies.

            // Delta-recompile with updated weights (T152: no ANE compilation)
            double rc_start = time_seconds();
            if (!orion_trainer_recompile_delta(trainer, args.weight_path)) {
                fprintf(stderr, "Error: delta recompile failed at step %d\n", step + 1);
                break;
            }
            double rc_ms = (time_seconds() - rc_start) * 1000.0;
            total_recompile_ms += rc_ms;

            // CSV log
            if (csv) {
                double step_total_ms = (time_seconds() - step_start) * 1000.0;
                fprintf(csv, "%d,%.4f,%.1f,%.1f,%.1f,%.3f\n",
                        step + 1, step_loss, train_time * 1000.0, rc_ms, step_total_ms, step_tflops);
                if ((step + 1) % 50 == 0) fflush(csv);
            }
        }

        double total_time = time_seconds() - wall_start;
        if (csv) { fclose(csv); csv = NULL; }

        // --- Training Summary ---
        fprintf(stderr, "\n--- Training Summary ---\n");
        fprintf(stderr, "  Steps:          %d\n", steps_completed);
        fprintf(stderr, "  Final loss:     %.4f\n", last_loss);
        fprintf(stderr, "  Total time:     %.1f s\n", total_time);
        if (steps_completed > 0) {
            double avg_train = total_train_ms / steps_completed;
            double avg_recompile = total_recompile_ms / steps_completed;
            double avg_total = (total_time * 1000.0) / steps_completed;
            double recompile_pct = (total_recompile_ms / (total_time * 1000.0)) * 100.0;
            double compile_pct = (compile_time / total_time) * 100.0;
            double avg_tflops = step_flops / (avg_train * 1e-3 * 1e12);

            fprintf(stderr, "  Avg train:      %.1f ms/step\n", avg_train);
            fprintf(stderr, "  Avg recompile:  %.1f ms/step\n", avg_recompile);
            fprintf(stderr, "  Avg total:      %.1f ms/step\n", avg_total);
            fprintf(stderr, "  Avg TFLOPS:     %.3f\n", avg_tflops);
            fprintf(stderr, "  Init compile:   %.1f s (%.1f%%)\n", compile_time, compile_pct);
            fprintf(stderr, "  Recompile:      %.1f s (%.1f%%)\n",
                    total_recompile_ms / 1000.0, recompile_pct);
        }
        fprintf(stderr, "------------------------\n");

        // Final checkpoint
        char final_path[512];
        snprintf(final_path, sizeof(final_path), "%s/ckpt_final.bin", args.checkpoint_dir);
        orion_checkpoint_save(trainer, final_path, args.steps, last_loss);
        fprintf(stderr, "Final checkpoint: %s\n", final_path);

        free(input);
        free(target);
        orion_data_loader_close(dl);
        orion_trainer_free(trainer);
        return 0;
    }
}
