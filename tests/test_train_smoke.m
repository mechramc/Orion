#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <math.h>
#import "kernels/training/stories_train.h"
#import "core/checkpoint.h"

// T072 Training Smoke Test
// Creates synthetic weight blobs in a temp directory, then runs
// the full training step end-to-end to verify:
// - All ANE programs compile
// - Forward pass produces finite loss
// - Backward pass completes without errors
// - Adam update runs
// - Loss decreases over a few steps

static int g_pass = 0, g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        NSLog(@"FAIL: %s — %@", __func__, msg); \
        g_fail++; return; \
    } \
} while(0)

#define PASS() do { NSLog(@"PASS: %s", __func__); g_pass++; } while(0)

#pragma mark - Helpers

/// Write a synthetic BLOBFILE (128-byte header + fp16 data) to disk.
static void write_blob(NSString* path, int n_elements) {
    int data_bytes = n_elements * sizeof(_Float16);
    int total = 128 + data_bytes;

    uint8_t *buf = (uint8_t *)calloc(total, 1);
    // BLOBFILE header
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = data_bytes;
    *(uint32_t *)(buf + 80) = 128;

    // Small random weights (Xavier-ish scale)
    float scale = 1.0f / sqrtf((float)n_elements);
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (int i = 0; i < n_elements; i++) {
        // Deterministic pseudo-random
        float val = scale * sinf((float)i * 0.137f + 0.5f);
        fp16[i] = (_Float16)val;
    }

    NSData *data = [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
    [[NSFileManager defaultManager] createDirectoryAtPath:[path stringByDeletingLastPathComponent]
                              withIntermediateDirectories:YES attributes:nil error:nil];
    [data writeToFile:path atomically:YES];
}

/// Create synthetic weight blobs for Stories110M in a temp directory.
static NSString* create_synthetic_weights(const OrionModelConfig* cfg) {
    NSString *tmp = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"orion_test_%d", getpid()]];
    [[NSFileManager defaultManager] createDirectoryAtPath:tmp
                              withIntermediateDirectories:YES attributes:nil error:nil];

    int d = cfg->d_model;
    int h = cfg->hidden_dim;
    int s = cfg->max_seq;

    for (int L = 0; L < cfg->n_layer; L++) {
        NSString *layer = [NSString stringWithFormat:@"%@/layer%d", tmp, L];
        // Forward weights
        write_blob([layer stringByAppendingPathComponent:@"rms1.bin"], d);
        write_blob([layer stringByAppendingPathComponent:@"wq.bin"], d * d);
        write_blob([layer stringByAppendingPathComponent:@"wk.bin"], d * d);
        write_blob([layer stringByAppendingPathComponent:@"wv.bin"], d * d);
        write_blob([layer stringByAppendingPathComponent:@"wo.bin"], d * d);
        write_blob([layer stringByAppendingPathComponent:@"rms2.bin"], d);
        write_blob([layer stringByAppendingPathComponent:@"w1.bin"], h * d);
        write_blob([layer stringByAppendingPathComponent:@"w3.bin"], h * d);
        write_blob([layer stringByAppendingPathComponent:@"w2.bin"], d * h);
        // Transposed weights for backward
        write_blob([layer stringByAppendingPathComponent:@"w2t.bin"], h * d);
        write_blob([layer stringByAppendingPathComponent:@"w1t.bin"], d * h);
        write_blob([layer stringByAppendingPathComponent:@"w3t.bin"], d * h);
        write_blob([layer stringByAppendingPathComponent:@"wot.bin"], d * d);
        write_blob([layer stringByAppendingPathComponent:@"wqt.bin"], d * d);
        write_blob([layer stringByAppendingPathComponent:@"wkt.bin"], d * d);
        write_blob([layer stringByAppendingPathComponent:@"wvt.bin"], d * d);
    }

    // Causal mask
    NSString *masks = [tmp stringByAppendingPathComponent:@"masks"];
    write_blob([masks stringByAppendingPathComponent:
        [NSString stringWithFormat:@"causal_%d.bin", s]], s * s);

    // Embedding + final RMSNorm
    write_blob([tmp stringByAppendingPathComponent:@"embed.bin"], cfg->vocab * d);
    write_blob([tmp stringByAppendingPathComponent:@"rms_final.bin"], d);

    return tmp;
}

static void cleanup_weights(NSString* path) {
    [[NSFileManager defaultManager] removeItemAtPath:path error:nil];
}

#pragma mark - Tests

// Use 1-layer Stories110M config for testing (6 compiles)
// Full 12-layer would be 72 compiles, too slow for a smoke test.
static const OrionModelConfig kTinyConfig = {
    .n_layer    = 1,
    .n_head     = 12,
    .d_model    = 768,
    .head_dim   = 64,
    .hidden_dim = 2048,
    .vocab      = 256,   // Small vocab for fast test (real is 32000)
    .max_seq    = 256,
};

static void test_trainer_create(void) {
    NSString *wdir = create_synthetic_weights(&kTinyConfig);
    OrionTrainer *t = orion_trainer_create(&kTinyConfig, wdir.UTF8String);
    ASSERT(t != NULL, @"trainer should create with synthetic weights");
    ASSERT(t->n_layers == 1, @"should have 1 layer");
    ASSERT(t->embed != NULL, @"embedding should be allocated");
    ASSERT(t->act_x != NULL, @"activations should be allocated");

    orion_trainer_free(t);
    cleanup_weights(wdir);
    PASS();
}

static void test_single_train_step(void) {
    NSString *wdir = create_synthetic_weights(&kTinyConfig);
    OrionTrainer *t = orion_trainer_create(&kTinyConfig, wdir.UTF8String);
    ASSERT(t != NULL, @"trainer should create");

    // Create synthetic input/target tokens
    int s = kTinyConfig.max_seq;
    int *input = (int *)malloc(s * sizeof(int));
    int *target = (int *)malloc(s * sizeof(int));
    for (int i = 0; i < s; i++) {
        input[i] = i % kTinyConfig.vocab;
        target[i] = (i + 1) % kTinyConfig.vocab;
    }

    orion_trainer_zero_grads(t);
    float loss = orion_train_step(t, input, target);
    NSLog(@"  Step 1 loss: %f", loss);

    ASSERT(isfinite(loss), @"loss should be finite");
    ASSERT(loss > 0.0f, @"loss should be positive");

    free(input);
    free(target);
    orion_trainer_free(t);
    cleanup_weights(wdir);
    PASS();
}

static void test_adam_update(void) {
    NSString *wdir = create_synthetic_weights(&kTinyConfig);
    OrionTrainer *t = orion_trainer_create(&kTinyConfig, wdir.UTF8String);
    ASSERT(t != NULL, @"trainer should create");

    int s = kTinyConfig.max_seq;
    int *input = (int *)malloc(s * sizeof(int));
    int *target = (int *)malloc(s * sizeof(int));
    for (int i = 0; i < s; i++) {
        input[i] = i % kTinyConfig.vocab;
        target[i] = (i + 1) % kTinyConfig.vocab;
    }

    // Run one step to generate gradients
    orion_trainer_zero_grads(t);
    float loss1 = orion_train_step(t, input, target);
    ASSERT(isfinite(loss1), @"step 1 loss should be finite");

    // Apply Adam update
    orion_trainer_adam_update(t);
    ASSERT(t->adam_t == 1, @"adam_t should be 1 after first update");

    free(input);
    free(target);
    orion_trainer_free(t);
    cleanup_weights(wdir);
    PASS();
}

static void test_loss_decreases(void) {
    NSString *wdir = create_synthetic_weights(&kTinyConfig);
    OrionTrainer *t = orion_trainer_create(&kTinyConfig, wdir.UTF8String);
    ASSERT(t != NULL, @"trainer should create");

    int s = kTinyConfig.max_seq;
    int *input = (int *)malloc(s * sizeof(int));
    int *target = (int *)malloc(s * sizeof(int));
    for (int i = 0; i < s; i++) {
        input[i] = i % kTinyConfig.vocab;
        target[i] = (i + 1) % kTinyConfig.vocab;
    }

    // Run 5 training steps with Adam updates
    float losses[5];
    for (int step = 0; step < 5; step++) {
        orion_trainer_zero_grads(t);
        losses[step] = orion_train_step(t, input, target);
        orion_trainer_adam_update(t);
        NSLog(@"  Step %d loss: %f", step + 1, losses[step]);
        NSString *fmsg = [NSString stringWithFormat:@"step %d loss should be finite", step + 1];
        ASSERT(isfinite(losses[step]), fmsg);
    }

    // Loss should decrease from step 1 to step 5
    // (Not strictly monotonic due to random init, but should trend down)
    NSString *dmsg = [NSString stringWithFormat:@"loss should decrease: %.4f → %.4f", losses[0], losses[4]];
    ASSERT(losses[4] < losses[0], dmsg);

    free(input);
    free(target);
    orion_trainer_free(t);
    cleanup_weights(wdir);
    PASS();
}

static void test_checkpoint_save_load(void) {
    NSString *wdir = create_synthetic_weights(&kTinyConfig);
    OrionTrainer *t = orion_trainer_create(&kTinyConfig, wdir.UTF8String);
    ASSERT(t != NULL, @"trainer should create");

    int s = kTinyConfig.max_seq;
    int *input = (int *)malloc(s * sizeof(int));
    int *target = (int *)malloc(s * sizeof(int));
    for (int i = 0; i < s; i++) {
        input[i] = i % kTinyConfig.vocab;
        target[i] = (i + 1) % kTinyConfig.vocab;
    }

    // Train 3 steps
    orion_trainer_zero_grads(t);
    float loss = 0.0f;
    for (int step = 0; step < 3; step++) {
        loss = orion_train_step(t, input, target);
        orion_trainer_adam_update(t);
    }
    NSLog(@"  Before checkpoint: step 3, loss=%f, adam_t=%d", loss, t->adam_t);

    // Save checkpoint
    NSString *ckpt_path = [NSTemporaryDirectory() stringByAppendingPathComponent:
        [NSString stringWithFormat:@"orion_ckpt_%d.bin", getpid()]];
    bool saved = orion_checkpoint_save(t, ckpt_path.UTF8String, 3, loss);
    ASSERT(saved, @"checkpoint save should succeed");

    // Verify file exists and has reasonable size
    NSDictionary *attrs = [[NSFileManager defaultManager] attributesOfItemAtPath:ckpt_path error:nil];
    unsigned long long size = [attrs fileSize];
    NSLog(@"  Checkpoint size: %llu bytes", size);
    ASSERT(size > sizeof(OrionCkptHdr), @"checkpoint should be larger than header");

    // Create a fresh trainer and load the checkpoint
    OrionTrainer *t2 = orion_trainer_create(&kTinyConfig, wdir.UTF8String);
    ASSERT(t2 != NULL, @"second trainer should create");

    int loaded_step = 0;
    float loaded_loss = 0.0f;
    bool loaded = orion_checkpoint_load(t2, ckpt_path.UTF8String, &loaded_step, &loaded_loss);
    ASSERT(loaded, @"checkpoint load should succeed");
    ASSERT(loaded_step == 3, @"loaded step should be 3");
    ASSERT(fabsf(loaded_loss - loss) < 1e-6f, @"loaded loss should match saved loss");
    ASSERT(t2->adam_t == t->adam_t, @"adam_t should match");
    NSLog(@"  Loaded checkpoint: step=%d, loss=%f, adam_t=%d", loaded_step, loaded_loss, t2->adam_t);

    // Verify weights match — spot check first layer wq
    int d = kTinyConfig.d_model;
    float diff = 0.0f;
    for (int i = 0; i < d*d; i++) {
        diff += fabsf(t->weights[0].wq[i] - t2->weights[0].wq[i]);
    }
    ASSERT(diff < 1e-6f, @"restored weights should match original");

    // Train one more step on loaded trainer — should produce same result as original
    orion_trainer_zero_grads(t);
    float loss_orig = orion_train_step(t, input, target);
    orion_trainer_zero_grads(t2);
    float loss_loaded = orion_train_step(t2, input, target);
    NSLog(@"  Continue orig loss=%f, loaded loss=%f", loss_orig, loss_loaded);
    ASSERT(fabsf(loss_orig - loss_loaded) < 1e-4f,
           @"loss from continued training should match");

    // Cleanup
    [[NSFileManager defaultManager] removeItemAtPath:ckpt_path error:nil];
    free(input); free(target);
    orion_trainer_free(t);
    orion_trainer_free(t2);
    cleanup_weights(wdir);
    PASS();
}

static void test_gradient_accumulation(void) {
    NSString *wdir = create_synthetic_weights(&kTinyConfig);
    OrionTrainer *t = orion_trainer_create(&kTinyConfig, wdir.UTF8String);
    ASSERT(t != NULL, @"trainer should create");

    int s = kTinyConfig.max_seq;
    int *input = (int *)malloc(s * sizeof(int));
    int *target = (int *)malloc(s * sizeof(int));
    for (int i = 0; i < s; i++) {
        input[i] = i % kTinyConfig.vocab;
        target[i] = (i + 1) % kTinyConfig.vocab;
    }

    // Accumulate 4 micro-batches then apply Adam
    int accum_steps = 4;
    orion_trainer_zero_grads(t);
    float total_loss = 0.0f;
    for (int mb = 0; mb < accum_steps; mb++) {
        float loss = orion_train_step(t, input, target);
        ASSERT(isfinite(loss), @"micro-batch loss should be finite");
        total_loss += loss;
    }
    orion_trainer_scale_grads(t, 1.0f / (float)accum_steps);
    orion_trainer_adam_update(t);
    float avg_loss = total_loss / (float)accum_steps;
    NSLog(@"  Accumulated %d micro-batches, avg loss: %f", accum_steps, avg_loss);

    // After accumulated update, run another step — loss should be finite
    orion_trainer_zero_grads(t);
    float loss2 = orion_train_step(t, input, target);
    NSLog(@"  Post-accumulation loss: %f", loss2);
    ASSERT(isfinite(loss2), @"post-accumulation loss should be finite");

    free(input);
    free(target);
    orion_trainer_free(t);
    cleanup_weights(wdir);
    PASS();
}

#pragma mark - Main

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        NSLog(@"=== T072: Training Smoke Tests ===");
        NSLog(@"Using tiny config: %d layer, d=%d, h=%d, vocab=%d, seq=%d",
              kTinyConfig.n_layer, kTinyConfig.d_model, kTinyConfig.hidden_dim,
              kTinyConfig.vocab, kTinyConfig.max_seq);

        test_trainer_create();
        test_single_train_step();
        test_adam_update();
        test_loss_decreases();
        test_gradient_accumulation();
        test_checkpoint_save_load();

        NSLog(@"\n=== Results: %d passed, %d failed ===", g_pass, g_fail);
        return g_fail > 0 ? 1 : 0;
    }
}
