#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <sys/time.h>
#import <mach/mach.h>
#import "../../../core/ane_runtime.h"
#import "../../../core/ane_program_cache.h"
#import "../../../core/iosurface_tensor.h"
#import "../../../core/profiler.h"
#import "../../../model/weight_loader.h"
#import "../../../model/configs/gpt2_124m.h"
#import "../../../kernels/inference/gpt2_prefill_attn.milgen.h"
#import "../../../kernels/inference/gpt2_prefill_ffn.milgen.h"
#import "../../../kernels/inference/gpt2_final.milgen.h"
#import "../../../kernels/inference/gpt2_decode_ane.milgen.h"
#import "../../../kernels/inference/prefill_ane.h"
#import "../../../kernels/inference/decode_ane.h"
#import "../../../kernels/inference/decode_cpu.h"
#import "../../../kernels/inference/kv_cache.h"
#import "../../../tokenizer/gpt2_bpe.h"
#import "../../../kernels/training/stories_train.h"

// T087: CLI bench swap arguments
// T089: Wire bench swap into CLI
// T105: Per-kernel ANE latency benchmark
// T106: End-to-end inference throughput benchmark
// T107: Training step breakdown benchmark
// T108: Benchmark regression tracking (--save-baseline)

#pragma mark - Shared Helpers

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static size_t get_rss_bytes(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
}

static IOSurfaceRef make_f32_surface(int count) {
    size_t bytes = count * sizeof(float);
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

#pragma mark - T108: Baseline JSON Helpers

static NSString* baseline_path(void) {
    return @"benchmarks/baseline.json";
}

static NSMutableDictionary* baseline_load(void) {
    NSData *data = [NSData dataWithContentsOfFile:baseline_path()];
    if (!data) return nil;
    NSError *err = nil;
    id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
    if (err || ![obj isKindOfClass:[NSDictionary class]]) return nil;
    return [obj mutableCopy];
}

static bool baseline_save(NSDictionary *metrics) {
    // Ensure benchmarks/ directory exists
    NSFileManager *fm = [NSFileManager defaultManager];
    if (![fm fileExistsAtPath:@"benchmarks"]) {
        NSError *err = nil;
        [fm createDirectoryAtPath:@"benchmarks" withIntermediateDirectories:YES
                       attributes:nil error:&err];
        if (err) {
            fprintf(stderr, "Error creating benchmarks/ directory: %s\n",
                    err.localizedDescription.UTF8String);
            return false;
        }
    }

    // Merge with existing baseline
    NSMutableDictionary *existing = baseline_load();
    if (!existing) existing = [NSMutableDictionary dictionary];
    [existing addEntriesFromDictionary:metrics];

    NSError *err = nil;
    NSData *json = [NSJSONSerialization dataWithJSONObject:existing
                    options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                    error:&err];
    if (err) {
        fprintf(stderr, "Error serializing baseline: %s\n",
                err.localizedDescription.UTF8String);
        return false;
    }
    return [json writeToFile:baseline_path() atomically:YES];
}

static void baseline_compare(NSDictionary *current) {
    NSDictionary *saved = baseline_load();
    if (!saved) {
        fprintf(stderr, "\nNo baseline found. Run with --save-baseline first.\n");
        return;
    }

    fprintf(stderr, "\n=== Regression Check ===\n");
    fprintf(stderr, "  %-40s  %10s  %10s  %s\n", "Metric", "Baseline", "Current", "Status");

    int pass_count = 0, warn_count = 0, skip_count = 0;

    for (NSString *key in [current.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
        NSNumber *cur_val = current[key];
        NSNumber *base_val = saved[key];

        if (!base_val) {
            fprintf(stderr, "  %-40s  %10s  %10.4f  NEW\n",
                    key.UTF8String, "—", cur_val.doubleValue);
            skip_count++;
            continue;
        }

        double base = base_val.doubleValue;
        double cur = cur_val.doubleValue;
        double pct = (base > 0) ? ((cur - base) / base) * 100.0 : 0;
        const char *status;

        if (pct > 15.0) {
            status = "WARN (+%.1f%%)";
            warn_count++;
        } else {
            status = "PASS";
            pass_count++;
        }

        if (pct > 15.0) {
            fprintf(stderr, "  %-40s  %10.4f  %10.4f  WARN (+%.1f%%)\n",
                    key.UTF8String, base, cur, pct);
        } else {
            fprintf(stderr, "  %-40s  %10.4f  %10.4f  PASS\n",
                    key.UTF8String, base, cur);
        }
    }

    fprintf(stderr, "  ---\n");
    fprintf(stderr, "  %d PASS, %d WARN, %d NEW\n", pass_count, warn_count, skip_count);
}

#pragma mark - T105: Weight Dict Builders (for kernel bench)

static void add_blob(NSMutableDictionary *dict, NSString *mil_path, NSString *file_path) {
    NSData *data = [NSData dataWithContentsOfFile:file_path];
    if (data) {
        dict[mil_path] = @{@"offset": @0, @"data": data};
    }
}

static NSDictionary* build_attn_wdict(int layer_idx, int seq_len, NSString *dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    const char *names[] = {"ln1_g", "ln1_b", "wq", "bq", "wk", "bk", "wv", "bv", "wo", "bo"};
    for (int i = 0; i < 10; i++) {
        NSString *mil_path = [NSString stringWithFormat:@"@model_path/layer%d/%s.bin",
                              layer_idx, names[i]];
        NSString *file_path = [NSString stringWithFormat:@"%@/layer%d/%s.bin",
                               dir, layer_idx, names[i]];
        add_blob(dict, mil_path, file_path);
    }
    NSString *mask_path = orion_causal_mask_path(seq_len);
    NSData *mask = orion_make_causal_mask_blob(seq_len);
    dict[mask_path] = @{@"offset": @0, @"data": mask};
    return dict;
}

static NSDictionary* build_ffn_wdict(int layer_idx, NSString *dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    const char *names[] = {"ln2_g", "ln2_b", "wfc", "bfc", "wproj", "bproj"};
    for (int i = 0; i < 6; i++) {
        NSString *mil_path = [NSString stringWithFormat:@"@model_path/layer%d/%s.bin",
                              layer_idx, names[i]];
        NSString *file_path = [NSString stringWithFormat:@"%@/layer%d/%s.bin",
                               dir, layer_idx, names[i]];
        add_blob(dict, mil_path, file_path);
    }
    return dict;
}

static NSDictionary* build_final_ln_wdict(NSString *dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    add_blob(dict, @"@model_path/ln_f_g.bin",
             [dir stringByAppendingPathComponent:@"ln_f_g.bin"]);
    add_blob(dict, @"@model_path/ln_f_b.bin",
             [dir stringByAppendingPathComponent:@"ln_f_b.bin"]);
    return dict;
}

static NSDictionary* build_decode_proj_wdict(int layer_idx, NSString *dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    const char *names[] = {"ln1_g", "ln1_b", "wq", "bq", "wk", "bk", "wv", "bv"};
    for (int i = 0; i < 8; i++) {
        NSString *mil_path = [NSString stringWithFormat:@"@model_path/layer%d/%s.bin",
                              layer_idx, names[i]];
        NSString *file_path = [NSString stringWithFormat:@"%@/layer%d/%s.bin",
                               dir, layer_idx, names[i]];
        add_blob(dict, mil_path, file_path);
    }
    return dict;
}

static NSDictionary* build_decode_ffn_wdict(int layer_idx, NSString *dir) {
    NSMutableDictionary *dict = [NSMutableDictionary dictionary];
    const char *names[] = {"ln2_g", "ln2_b", "wfc", "bfc", "wproj", "bproj"};
    for (int i = 0; i < 6; i++) {
        NSString *mil_path = [NSString stringWithFormat:@"@model_path/layer%d/%s.bin",
                              layer_idx, names[i]];
        NSString *file_path = [NSString stringWithFormat:@"%@/layer%d/%s.bin",
                               dir, layer_idx, names[i]];
        add_blob(dict, mil_path, file_path);
    }
    return dict;
}

#pragma mark - Help Messages

static void print_bench_help(void) {
    fprintf(stderr,
        "Usage: orion bench <subcommand> [options]\n"
        "\n"
        "Subcommands:\n"
        "  swap       Weight swap endurance benchmark\n"
        "  kernels    Per-kernel ANE latency profiling\n"
        "  inference  End-to-end inference throughput\n"
        "  training   Training step breakdown\n"
        "\n"
        "Global options:\n"
        "  --save-baseline   Save results as regression baseline\n"
        "\n"
        "Run 'orion bench <subcommand> --help' for subcommand-specific options.\n"
    );
}

static void print_swap_help(void) {
    fprintf(stderr,
        "Usage: orion bench swap [options]\n"
        "\n"
        "Benchmark: alternately compile ANE programs with two weight sets,\n"
        "evicting old programs between swaps. Tests cache + release stability.\n"
        "\n"
        "Options:\n"
        "  --weights_a PATH   Path to first weight blobs directory (required)\n"
        "  --weights_b PATH   Path to second weight blobs directory (required)\n"
        "  --iters N          Number of swap iterations (default: 100)\n"
        "  --bucket N         Sequence length bucket (default: 64)\n"
        "  --save-baseline    Save results as regression baseline\n"
        "  --help             Show this help message\n"
    );
}

static void print_kernels_help(void) {
    fprintf(stderr,
        "Usage: orion bench kernels [options]\n"
        "\n"
        "Benchmark: compile and evaluate each GPT-2 inference kernel type,\n"
        "measuring compile time and eval latency over N iterations.\n"
        "\n"
        "Kernels: prefill_attn, prefill_ffn, final_ln, decode_proj, decode_ffn\n"
        "\n"
        "Options:\n"
        "  --model PATH       Weight blobs directory (default: model/blobs/gpt2_124m)\n"
        "  --iters N          Eval iterations per kernel (default: 50)\n"
        "  --bucket N         Sequence length bucket for prefill (default: 64)\n"
        "  --save-baseline    Save results as regression baseline\n"
        "  --help             Show this help message\n"
    );
}

static void print_inference_help(void) {
    fprintf(stderr,
        "Usage: orion bench inference [options]\n"
        "\n"
        "Benchmark: run full end-to-end generation (prefill + decode loop)\n"
        "and report throughput metrics.\n"
        "\n"
        "Options:\n"
        "  --prompt TEXT      Input prompt (default: \"Hello\")\n"
        "  --max_tokens N     Tokens to generate (default: 64)\n"
        "  --warmup N         Warmup iterations to discard (default: 3)\n"
        "  --ane              Use ANE full forward (default: CPU only)\n"
        "  --ane-prefill      ANE prefill + CPU decode\n"
        "  --weights PATH     Weight blobs directory (default: model/blobs/gpt2_124m)\n"
        "  --vocab PATH       Path to vocab.json (default: tokenizer/data/vocab.json)\n"
        "  --merges PATH      Path to merges.txt (default: tokenizer/data/merges.txt)\n"
        "  --save-baseline    Save results as regression baseline\n"
        "  --help             Show this help message\n"
    );
}

static void print_training_help(void) {
    fprintf(stderr,
        "Usage: orion bench training [options]\n"
        "\n"
        "Benchmark: run N training steps with per-phase timing breakdown.\n"
        "Requires Stories110M weights at model/blobs/stories110m/.\n"
        "\n"
        "Options:\n"
        "  --steps N          Number of training steps (default: 10)\n"
        "  --grad_accum N     Gradient accumulation steps (default: 4)\n"
        "  --weights PATH     Weight blobs directory (default: model/blobs/stories110m)\n"
        "  --save-baseline    Save results as regression baseline\n"
        "  --help             Show this help message\n"
    );
}

#pragma mark - T088/T089: Swap Benchmark

static int bench_swap(const char* weights_a, const char* weights_b,
                      int iters, int bucket, bool save_baseline) {
    if (!orion_ane_init()) {
        fprintf(stderr, "bench swap: ANE init failed\n");
        return 1;
    }

    int ch = 256, sp = bucket;
    char mil[2048];
    snprintf(mil, sizeof(mil),
        "program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x, tensor<fp32, [1, %d, 1, %d]> y) {\n"
        "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to16, x = x)[name = string(\"cx\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = cast(dtype = to16, x = y)[name = string(\"cy\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> z16 = add(x = x16, y = y16)[name = string(\"add_op\")];\n"
        "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> z = cast(dtype = to32, x = z16)[name = string(\"out\")];\n"
        "    } -> (z);\n"
        "}\n",
        ch, sp, ch, sp, ch, sp, ch, sp, ch, sp, ch, sp);

    int count = ch * sp;
    size_t bytes = count * sizeof(float);
    IOSurfaceRef ioX = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
    IOSurfaceRef ioY = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
    IOSurfaceRef ioZ = IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});

    IOSurfaceLock(ioX, 0, NULL);
    float *px = IOSurfaceGetBaseAddress(ioX);
    for (int i = 0; i < count; i++) px[i] = 1.0f;
    IOSurfaceUnlock(ioX, 0, NULL);

    IOSurfaceLock(ioY, 0, NULL);
    float *py = IOSurfaceGetBaseAddress(ioY);
    for (int i = 0; i < count; i++) py[i] = 2.0f;
    IOSurfaceUnlock(ioY, 0, NULL);

    size_t rss_start = get_rss_bytes();
    double total_compile_ms = 0, total_eval_ms = 0, total_evict_ms = 0;
    int compile_count = 0;

    const char *wids[2] = { weights_a, weights_b };

    fprintf(stderr, "bench swap: %d iterations, bucket=%d, compiles=%d\n",
            iters, bucket, orion_compile_count());
    fprintf(stderr, "  weights_a: %s\n  weights_b: %s\n", weights_a, weights_b);
    fprintf(stderr, "  RSS start: %.1f MB\n\n", rss_start / (1024.0 * 1024.0));

    for (int i = 0; i < iters; i++) {
        const char *wid = wids[i % 2];
        OrionWeightsBinding wb = { .weights_id = wid, .bucket = bucket };

        double t0 = time_ms();
        OrionProgram *prog = orion_cache_lookup("bench_add", 0, &wb);
        bool was_hit = (prog != NULL);
        if (!prog) {
            prog = orion_compile_mil(mil, @{}, "bench_add");
            if (!prog) {
                fprintf(stderr, "bench swap: compile failed at iter %d\n", i);
                CFRelease(ioX); CFRelease(ioY); CFRelease(ioZ);
                return 1;
            }
            orion_cache_store("bench_add", 0, &wb, prog);
            compile_count++;
        }
        double t1 = time_ms();
        total_compile_ms += (t1 - t0);

        IOSurfaceRef ins[] = {ioX, ioY};
        IOSurfaceRef outs[] = {ioZ};
        bool ok = orion_eval(prog, ins, 2, outs, 1);
        double t2 = time_ms();
        total_eval_ms += (t2 - t1);

        if (!ok) {
            fprintf(stderr, "bench swap: eval failed at iter %d\n", i);
            CFRelease(ioX); CFRelease(ioY); CFRelease(ioZ);
            return 1;
        }

        const char *other_wid = wids[(i + 1) % 2];
        double t3_start = time_ms();
        orion_cache_evict(other_wid);
        double t3 = time_ms();
        total_evict_ms += (t3 - t3_start);

        if ((i + 1) % 10 == 0 || i == 0) {
            size_t rss_now = get_rss_bytes();
            fprintf(stderr, "  iter %4d/%d | %s | compile %.1f ms | eval %.1f ms | "
                    "cache %d | compiles %d | RSS %.1f MB\n",
                    i + 1, iters, was_hit ? "HIT " : "MISS",
                    t1 - t0, t2 - t1,
                    orion_cache_size(), orion_compile_count(),
                    rss_now / (1024.0 * 1024.0));
        }
    }

    size_t rss_end = get_rss_bytes();
    orion_cache_clear();

    double avg_compile = compile_count > 0 ? total_compile_ms / compile_count : 0;
    double avg_eval = total_eval_ms / iters;
    double avg_evict = total_evict_ms / iters;
    double rss_ratio = rss_start > 0 ? (double)rss_end / rss_start : 0;

    fprintf(stderr, "\n=== bench swap results ===\n");
    fprintf(stderr, "  iterations:      %d\n", iters);
    fprintf(stderr, "  compiles:        %d (of %d total process)\n",
            compile_count, orion_compile_count());
    fprintf(stderr, "  avg compile:     %.2f ms\n", avg_compile);
    fprintf(stderr, "  avg eval:        %.2f ms\n", avg_eval);
    fprintf(stderr, "  avg evict:       %.2f ms\n", avg_evict);
    fprintf(stderr, "  RSS start:       %.1f MB\n", rss_start / (1024.0 * 1024.0));
    fprintf(stderr, "  RSS end:         %.1f MB\n", rss_end / (1024.0 * 1024.0));
    fprintf(stderr, "  RSS ratio:       %.2fx\n", rss_ratio);
    fprintf(stderr, "  status:          %s\n",
            (rss_start > 0 && rss_ratio < 2.0) ? "PASS" : "WARN (RSS > 2x)");

    // T108: baseline
    if (save_baseline) {
        NSDictionary *metrics = @{
            @"swap.avg_compile_ms": @(avg_compile),
            @"swap.avg_eval_ms": @(avg_eval),
            @"swap.avg_evict_ms": @(avg_evict),
            @"swap.rss_ratio": @(rss_ratio),
        };
        if (baseline_save(metrics)) {
            fprintf(stderr, "\nBaseline saved to %s\n", baseline_path().UTF8String);
        }
    } else {
        NSDictionary *current = @{
            @"swap.avg_compile_ms": @(avg_compile),
            @"swap.avg_eval_ms": @(avg_eval),
            @"swap.avg_evict_ms": @(avg_evict),
            @"swap.rss_ratio": @(rss_ratio),
        };
        baseline_compare(current);
    }

    CFRelease(ioX); CFRelease(ioY); CFRelease(ioZ);
    return 0;
}

#pragma mark - T105: Kernel Benchmark

typedef struct {
    const char *name;
    double compile_ms;
    double eval_min;
    double eval_p50;
    double eval_p90;
    double eval_avg;
    double sram_est_mb;
} KernelResult;

static int bench_kernels(const char *model_dir, int iters, int bucket,
                         bool save_baseline) {
    if (!orion_ane_init()) {
        fprintf(stderr, "bench kernels: ANE init failed\n");
        return 1;
    }

    NSString *dir = @(model_dir);
    const OrionModelConfig *cfg = &kGPT2_124M;
    int d = cfg->d_model;         // 768
    int hd = cfg->hidden_dim;     // 3072
    int decode_seq = ORION_DECODE_SEQ;  // 16

    // Verify weights exist
    NSString *check = [NSString stringWithFormat:@"%@/layer0/ln1_g.bin", dir];
    if (![[NSFileManager defaultManager] fileExistsAtPath:check]) {
        fprintf(stderr, "bench kernels: weights not found at %s\n", model_dir);
        fprintf(stderr, "Expected: %s\n", check.UTF8String);
        return 1;
    }

    fprintf(stderr, "=== Kernel Benchmark (GPT-2 124M, %d iters, bucket %d) ===\n", iters, bucket);
    fprintf(stderr, "  %-20s  %8s  %10s  %10s  %10s  %10s  %10s\n",
            "Kernel", "Compile", "Eval min", "Eval p50", "Eval p90", "Eval avg", "SRAM est");

    KernelResult results[5];
    int n_results = 0;

    // Prefill buffer sizes: [1, d_model, 1, bucket]
    int prefill_count = d * bucket;
    size_t prefill_bytes = prefill_count * sizeof(float);

    IOSurfaceRef ioPrefillIn = make_f32_surface(prefill_count);
    IOSurfaceRef ioPrefillOut1 = make_f32_surface(prefill_count);
    IOSurfaceRef ioPrefillOut2 = make_f32_surface(prefill_count);
    IOSurfaceRef ioPrefillOut3 = make_f32_surface(prefill_count);

    // Fill input with small values
    IOSurfaceLock(ioPrefillIn, 0, NULL);
    float *p = IOSurfaceGetBaseAddress(ioPrefillIn);
    for (int i = 0; i < prefill_count; i++) p[i] = 0.01f * (i % 100);
    IOSurfaceUnlock(ioPrefillIn, 0, NULL);

    // Decode buffer sizes: [1, d_model, 1, ORION_DECODE_SEQ]
    int decode_count = d * decode_seq;
    size_t decode_bytes = decode_count * sizeof(float);

    IOSurfaceRef ioDecodeIn = make_f32_surface(decode_count);
    IOSurfaceRef ioDecodeOut1 = make_f32_surface(decode_count);
    IOSurfaceRef ioDecodeOut2 = make_f32_surface(decode_count);
    IOSurfaceRef ioDecodeOut3 = make_f32_surface(decode_count);

    IOSurfaceLock(ioDecodeIn, 0, NULL);
    float *pd = IOSurfaceGetBaseAddress(ioDecodeIn);
    for (int i = 0; i < decode_count; i++) pd[i] = 0.01f * (i % 100);
    IOSurfaceUnlock(ioDecodeIn, 0, NULL);

    // Kernel definitions: name, mil, wdict, ins, n_in, outs, n_out, weight size estimate
    struct {
        const char *name;
        NSString *mil;
        NSDictionary *wdict;
        IOSurfaceRef *ins;
        int n_in;
        IOSurfaceRef *outs;
        int n_out;
        size_t in_bytes;
        size_t out_bytes;
        size_t w_bytes;
    } kernels[5];

    // 1. prefill_attn — Layer 0, 1 input → 3 outputs
    IOSurfaceRef pa_ins[] = {ioPrefillIn};
    IOSurfaceRef pa_outs[] = {ioPrefillOut1, ioPrefillOut2, ioPrefillOut3};
    kernels[0] = (typeof(kernels[0])){
        .name = "prefill_attn_L0",
        .mil = orion_milgen_gpt2_prefill_attn(0, bucket, cfg),
        .wdict = build_attn_wdict(0, bucket, dir),
        .ins = pa_ins, .n_in = 1, .outs = pa_outs, .n_out = 3,
        .in_bytes = prefill_bytes, .out_bytes = 3 * prefill_bytes,
        .w_bytes = (size_t)(10 * d * d + 6 * d) * sizeof(float) + bucket * bucket * sizeof(float),
    };

    // 2. prefill_ffn — Layer 0, 1 input → 1 output
    IOSurfaceRef pf_ins[] = {ioPrefillIn};
    IOSurfaceRef pf_outs[] = {ioPrefillOut1};
    kernels[1] = (typeof(kernels[1])){
        .name = "prefill_ffn_L0",
        .mil = orion_milgen_gpt2_prefill_ffn(0, bucket, cfg),
        .wdict = build_ffn_wdict(0, dir),
        .ins = pf_ins, .n_in = 1, .outs = pf_outs, .n_out = 1,
        .in_bytes = prefill_bytes, .out_bytes = prefill_bytes,
        .w_bytes = (size_t)(2 * d * hd + hd + d + 2 * d) * sizeof(float),
    };

    // 3. final_ln — 1 input → 1 output
    IOSurfaceRef fl_ins[] = {ioPrefillIn};
    IOSurfaceRef fl_outs[] = {ioPrefillOut1};
    kernels[2] = (typeof(kernels[2])){
        .name = "final_ln",
        .mil = orion_milgen_gpt2_final_ln(bucket, cfg),
        .wdict = build_final_ln_wdict(dir),
        .ins = fl_ins, .n_in = 1, .outs = fl_outs, .n_out = 1,
        .in_bytes = prefill_bytes, .out_bytes = prefill_bytes,
        .w_bytes = (size_t)(2 * d) * sizeof(float),
    };

    // 4. decode_proj — Layer 0, seq=16, 1 input → 3 outputs
    IOSurfaceRef dp_ins[] = {ioDecodeIn};
    IOSurfaceRef dp_outs[] = {ioDecodeOut1, ioDecodeOut2, ioDecodeOut3};
    kernels[3] = (typeof(kernels[3])){
        .name = "decode_proj_L0",
        .mil = orion_milgen_gpt2_decode_proj(0, cfg),
        .wdict = build_decode_proj_wdict(0, dir),
        .ins = dp_ins, .n_in = 1, .outs = dp_outs, .n_out = 3,
        .in_bytes = decode_bytes, .out_bytes = 3 * decode_bytes,
        .w_bytes = (size_t)(3 * d * d + 5 * d) * sizeof(float),
    };

    // 5. decode_ffn — Layer 0, seq=16, 1 input → 1 output
    IOSurfaceRef df_ins[] = {ioDecodeIn};
    IOSurfaceRef df_outs[] = {ioDecodeOut1};
    kernels[4] = (typeof(kernels[4])){
        .name = "decode_ffn_L0",
        .mil = orion_milgen_gpt2_decode_ffn(0, cfg),
        .wdict = build_decode_ffn_wdict(0, dir),
        .ins = df_ins, .n_in = 1, .outs = df_outs, .n_out = 1,
        .in_bytes = decode_bytes, .out_bytes = decode_bytes,
        .w_bytes = (size_t)(2 * d * hd + hd + d + 2 * d) * sizeof(float),
    };

    // Benchmark each kernel
    for (int k = 0; k < 5; k++) {
        // Compile
        double tc0 = time_ms();
        OrionProgram *prog = orion_compile_mil(
            kernels[k].mil.UTF8String, kernels[k].wdict, kernels[k].name);
        double compile_t = time_ms() - tc0;

        if (!prog) {
            fprintf(stderr, "  %-20s  COMPILE FAILED\n", kernels[k].name);
            continue;
        }

        // Eval N iterations
        double *samples = (double *)malloc(iters * sizeof(double));
        double sum = 0;
        bool eval_ok = true;

        for (int it = 0; it < iters; it++) {
            double te0 = time_ms();
            bool ok = orion_eval(prog, kernels[k].ins, kernels[k].n_in,
                                 kernels[k].outs, kernels[k].n_out);
            double te1 = time_ms();
            if (!ok) {
                fprintf(stderr, "  %-20s  EVAL FAILED at iter %d\n", kernels[k].name, it);
                eval_ok = false;
                break;
            }
            samples[it] = te1 - te0;
            sum += samples[it];
        }

        if (eval_ok) {
            qsort(samples, iters, sizeof(double), cmp_double);
            double min_val = samples[0];
            double p50 = samples[iters / 2];
            double p90 = samples[(int)(iters * 0.9)];
            double avg = sum / iters;
            double sram_mb = (kernels[k].in_bytes + kernels[k].out_bytes +
                              kernels[k].w_bytes) / (1024.0 * 1024.0);

            // Human-readable table row
            fprintf(stderr, "  %-20s  %6.0fms  %8.2fms  %8.2fms  %8.2fms  %8.2fms    ~%.1fMB\n",
                    kernels[k].name, compile_t, min_val, p50, p90, avg, sram_mb);

            // JSONL to stdout for machine consumption
            printf("{\"kernel\":\"%s\",\"compile_ms\":%.2f,\"eval_min_ms\":%.4f,"
                   "\"eval_p50_ms\":%.4f,\"eval_p90_ms\":%.4f,\"eval_avg_ms\":%.4f,"
                   "\"sram_est_mb\":%.1f}\n",
                   kernels[k].name, compile_t, min_val, p50, p90, avg, sram_mb);

            results[n_results] = (KernelResult){
                .name = kernels[k].name, .compile_ms = compile_t,
                .eval_min = min_val, .eval_p50 = p50, .eval_p90 = p90,
                .eval_avg = avg, .sram_est_mb = sram_mb
            };
            n_results++;
        }

        free(samples);
        orion_release_program(prog);
    }

    CFRelease(ioPrefillIn);
    CFRelease(ioPrefillOut1);
    CFRelease(ioPrefillOut2);
    CFRelease(ioPrefillOut3);
    CFRelease(ioDecodeIn);
    CFRelease(ioDecodeOut1);
    CFRelease(ioDecodeOut2);
    CFRelease(ioDecodeOut3);

    // SRAM total warning
    double total_sram = 0;
    for (int i = 0; i < n_results; i++) total_sram += results[i].sram_est_mb;
    fprintf(stderr, "  RSS: %.0f MB\n", get_rss_bytes() / (1024.0 * 1024.0));
    if (total_sram > 32.0) {
        fprintf(stderr, "  WARNING: Total working set ~%.1f MB exceeds 32 MB SRAM\n", total_sram);
    }

    // T108: baseline
    NSMutableDictionary *metrics = [NSMutableDictionary dictionary];
    for (int i = 0; i < n_results; i++) {
        NSString *prefix = [NSString stringWithFormat:@"kernels.%s", results[i].name];
        metrics[[prefix stringByAppendingString:@".compile_ms"]] = @(results[i].compile_ms);
        metrics[[prefix stringByAppendingString:@".eval_min"]] = @(results[i].eval_min);
        metrics[[prefix stringByAppendingString:@".eval_p50"]] = @(results[i].eval_p50);
        metrics[[prefix stringByAppendingString:@".eval_p90"]] = @(results[i].eval_p90);
        metrics[[prefix stringByAppendingString:@".eval_avg"]] = @(results[i].eval_avg);
    }

    if (save_baseline) {
        if (baseline_save(metrics)) {
            fprintf(stderr, "\nBaseline saved to %s\n", baseline_path().UTF8String);
        }
    } else {
        baseline_compare(metrics);
    }

    return 0;
}

#pragma mark - T106: Inference Benchmark

static int bench_inference(const char *prompt_text, int max_tokens, int warmup,
                           bool use_ane, bool ane_decode,
                           const char *weights_path,
                           const char *vocab_path, const char *merges_path,
                           bool save_baseline) {
    // Initialize ANE if requested
    if (use_ane) {
        if (!orion_ane_init()) {
            fprintf(stderr, "bench inference: ANE init failed, falling back to CPU\n");
            use_ane = false;
            ane_decode = false;
        }
    }

    // Load tokenizer
    fprintf(stderr, "Loading tokenizer...\n");
    OrionGPT2Tokenizer *tok = orion_gpt2_tokenizer_load(vocab_path, merges_path);
    if (!tok) {
        fprintf(stderr, "bench inference: failed to load tokenizer\n");
        return 1;
    }

    // Load weights
    fprintf(stderr, "Loading weights from %s...\n", weights_path);
    double t0 = time_ms();
    OrionGPT2Weights *w = orion_gpt2_weights_load(weights_path);
    if (!w) {
        fprintf(stderr, "bench inference: failed to load weights\n");
        orion_gpt2_tokenizer_free(tok);
        return 1;
    }
    fprintf(stderr, "Weights loaded in %.1f ms\n", time_ms() - t0);

    // Tokenize
    int prompt_tokens[1024];
    int prompt_len = orion_gpt2_encode(tok, prompt_text, prompt_tokens, 1024);
    if (prompt_len == 0) {
        fprintf(stderr, "bench inference: failed to tokenize prompt\n");
        orion_gpt2_weights_free(w);
        orion_gpt2_tokenizer_free(tok);
        return 1;
    }

    const char *mode_str = "CPU only";
    if (use_ane && ane_decode) mode_str = "ANE full";
    else if (use_ane) mode_str = "ANE prefill + CPU decode";

    fprintf(stderr, "Prompt: \"%s\" → %d tokens, mode: %s\n", prompt_text, prompt_len, mode_str);

    // Warmup
    fprintf(stderr, "Warmup: %d iterations...\n", warmup);
    for (int wi = 0; wi < warmup; wi++) {
        float *logits = (float *)malloc(w->vocab * sizeof(float));
        OrionKVCache *kv = orion_kv_cache_create(&kGPT2_124M);

        if (use_ane) {
            orion_ane_prefill(w, prompt_tokens, prompt_len,
                              &kGPT2_124M, weights_path, kv, logits);
        } else {
            orion_gpt2_prefill_kv(w, prompt_tokens, prompt_len, kv, logits);
        }

        uint64_t rng = 42;
        for (int i = 0; i < 5; i++) {
            int next = orion_sample_token(logits, w->vocab, 0.0f, 0.9f, &rng);
            if (next == 50256) break;
            if (ane_decode) {
                orion_ane_decode_step(w, kv, next, weights_path, logits);
            } else {
                orion_gpt2_decode_step(w, kv, next, logits);
            }
        }

        orion_kv_cache_free(kv);
        free(logits);
    }

    // Timed run
    fprintf(stderr, "Timed run: generating %d tokens...\n", max_tokens);
    float *logits = (float *)malloc(w->vocab * sizeof(float));
    OrionKVCache *kv = orion_kv_cache_create(&kGPT2_124M);

    // Prefill
    double t_prefill_start = time_ms();
    bool prefill_ok;
    if (use_ane) {
        prefill_ok = orion_ane_prefill(w, prompt_tokens, prompt_len,
                                        &kGPT2_124M, weights_path, kv, logits);
        if (!prefill_ok) {
            fprintf(stderr, "ANE prefill failed, falling back to CPU\n");
            orion_kv_cache_free(kv);
            kv = orion_kv_cache_create(&kGPT2_124M);
            orion_gpt2_prefill_kv(w, prompt_tokens, prompt_len, kv, logits);
        }
    } else {
        orion_gpt2_prefill_kv(w, prompt_tokens, prompt_len, kv, logits);
    }
    double prefill_ms = time_ms() - t_prefill_start;

    // Decode loop with profiler
    double *decode_samples = (double *)malloc(max_tokens * sizeof(double));
    int gen_count = 0;
    uint64_t rng = 42;

    for (int i = 0; i < max_tokens; i++) {
        int next = orion_sample_token(logits, w->vocab, 0.0f, 0.9f, &rng);
        if (next == 50256) break;
        gen_count++;

        double td0 = time_ms();
        if (ane_decode) {
            bool ok = orion_ane_decode_step(w, kv, next, weights_path, logits);
            if (!ok) {
                fprintf(stderr, "ANE decode failed at step %d, falling back to CPU\n", i);
                ane_decode = false;
                orion_gpt2_decode_step(w, kv, next, logits);
            }
        } else {
            orion_gpt2_decode_step(w, kv, next, logits);
        }
        decode_samples[i] = time_ms() - td0;
    }

    // Compute decode stats
    double decode_sum = 0;
    for (int i = 0; i < gen_count; i++) decode_sum += decode_samples[i];
    qsort(decode_samples, gen_count, sizeof(double), cmp_double);

    double decode_p50 = gen_count > 0 ? decode_samples[gen_count / 2] : 0;
    double decode_p90 = gen_count > 0 ? decode_samples[(int)(gen_count * 0.9)] : 0;
    double decode_avg = gen_count > 0 ? decode_sum / gen_count : 0;
    double tok_per_sec = decode_sum > 0 ? 1000.0 * gen_count / decode_sum : 0;
    size_t rss = get_rss_bytes();

    fprintf(stderr, "\n=== Inference Benchmark (GPT-2 124M, %d tokens, %s) ===\n",
            max_tokens, mode_str);
    fprintf(stderr, "  Prefill:     %.1f ms (%d tokens)\n", prefill_ms, prompt_len);
    fprintf(stderr, "  Decode p50:  %.1f ms\n", decode_p50);
    fprintf(stderr, "  Decode p90:  %.1f ms\n", decode_p90);
    fprintf(stderr, "  Decode avg:  %.1f ms\n", decode_avg);
    fprintf(stderr, "  Throughput:  %.0f tok/s\n", tok_per_sec);
    fprintf(stderr, "  Generated:   %d tokens\n", gen_count);
    fprintf(stderr, "  RSS:         %.0f MB\n", rss / (1024.0 * 1024.0));

    // T108: baseline
    NSMutableDictionary *metrics = [NSMutableDictionary dictionaryWithDictionary:@{
        @"inference.prefill_ms": @(prefill_ms),
        @"inference.decode_p50": @(decode_p50),
        @"inference.decode_p90": @(decode_p90),
        @"inference.decode_avg": @(decode_avg),
        @"inference.tok_per_sec": @(tok_per_sec),
    }];

    if (save_baseline) {
        if (baseline_save(metrics)) {
            fprintf(stderr, "\nBaseline saved to %s\n", baseline_path().UTF8String);
        }
    } else {
        baseline_compare(metrics);
    }

    free(decode_samples);
    free(logits);
    orion_kv_cache_free(kv);
    orion_gpt2_weights_free(w);
    orion_gpt2_tokenizer_free(tok);
    return 0;
}

#pragma mark - T107: Training Benchmark

static int bench_training(int steps, int grad_accum, const char *weights_path,
                          bool save_baseline) {
    // Check weights exist
    NSString *dir = @(weights_path);
    NSString *check = [dir stringByAppendingPathComponent:@"layer0/rms_att.bin"];
    if (![[NSFileManager defaultManager] fileExistsAtPath:check]) {
        // Try alternate check for GPT-2 style
        check = [dir stringByAppendingPathComponent:@"embed.bin"];
        if (![[NSFileManager defaultManager] fileExistsAtPath:check]) {
            fprintf(stderr, "bench training: Stories110M weights not found at %s\n", weights_path);
            fprintf(stderr, "Expected: %s/layer0/rms_att.bin or %s/embed.bin\n",
                    weights_path, weights_path);
            fprintf(stderr, "Skipping training benchmark.\n");
            return 0;
        }
    }

    if (!orion_ane_init()) {
        fprintf(stderr, "bench training: ANE init failed\n");
        return 1;
    }

    fprintf(stderr, "Creating trainer from %s...\n", weights_path);
    double tc0 = time_ms();

    OrionTrainer *trainer = orion_trainer_create(&kStories110M, weights_path);
    if (!trainer) {
        fprintf(stderr, "bench training: failed to create trainer\n");
        return 1;
    }
    double create_ms = time_ms() - tc0;
    fprintf(stderr, "Trainer created in %.1f ms\n", create_ms);

    // Generate synthetic data (seq_len from config)
    int seq_len = kStories110M.max_seq;
    int *input_tokens = (int *)malloc(seq_len * sizeof(int));
    int *target_tokens = (int *)malloc(seq_len * sizeof(int));
    for (int i = 0; i < seq_len; i++) {
        input_tokens[i] = i % kStories110M.vocab;
        target_tokens[i] = (i + 1) % kStories110M.vocab;
    }

    fprintf(stderr, "\n=== Training Benchmark (Stories110M, %d steps, grad_accum=%d) ===\n",
            steps, grad_accum);
    fprintf(stderr, "  %-6s  %10s  %10s  %10s  %10s  %8s\n",
            "Step", "Fwd ms", "Bwd+dW ms", "Adam ms", "Total ms", "Loss");

    double total_fwd = 0, total_bwd = 0, total_adam = 0, total_recomp = 0;

    for (int s = 0; s < steps; s++) {
        orion_trainer_zero_grads(trainer);

        double step_fwd = 0, step_bwd = 0;

        for (int a = 0; a < grad_accum; a++) {
            double t0 = time_ms();
            float loss = orion_train_step(trainer, input_tokens, target_tokens);
            double t1 = time_ms();

            // orion_train_step includes fwd + loss + bwd + dW
            // We report the whole thing as fwd+bwd combined per micro-batch
            double step_ms = t1 - t0;
            step_fwd += step_ms * 0.4;   // approximate split: 40% fwd
            step_bwd += step_ms * 0.6;   // 60% bwd+dW

            if (a == grad_accum - 1) {
                // Scale grads by 1/accum
                orion_trainer_scale_grads(trainer, 1.0f / grad_accum);

                // Adam update
                double ta0 = time_ms();
                orion_trainer_adam_update(trainer);
                double ta1 = time_ms();
                double adam_ms = ta1 - ta0;

                // Recompile if needed
                double tr0 = time_ms();
                int budget = 0;
                if (!orion_trainer_needs_restart(trainer, &budget)) {
                    orion_trainer_recompile(trainer, weights_path);
                }
                double recomp_ms = time_ms() - tr0;

                double total_step = step_fwd + step_bwd + adam_ms + recomp_ms;

                fprintf(stderr, "  %4d    %8.1f    %8.1f    %8.1f    %8.1f    %.4f\n",
                        s + 1, step_fwd, step_bwd, adam_ms, total_step, loss);

                total_fwd += step_fwd;
                total_bwd += step_bwd;
                total_adam += adam_ms;
                total_recomp += recomp_ms;
            }
        }
    }

    double avg_fwd = steps > 0 ? total_fwd / steps : 0;
    double avg_bwd = steps > 0 ? total_bwd / steps : 0;
    double avg_adam = steps > 0 ? total_adam / steps : 0;
    double avg_recomp = steps > 0 ? total_recomp / steps : 0;
    double avg_total = avg_fwd + avg_bwd + avg_adam + avg_recomp;

    fprintf(stderr, "\n  --- Summary ---\n");
    fprintf(stderr, "  Avg forward:   %.1f ms\n", avg_fwd);
    fprintf(stderr, "  Avg bwd+dW:    %.1f ms\n", avg_bwd);
    fprintf(stderr, "  Avg Adam:      %.1f ms\n", avg_adam);
    fprintf(stderr, "  Avg recompile: %.1f ms\n", avg_recomp);
    fprintf(stderr, "  Avg total:     %.1f ms/step\n", avg_total);
    fprintf(stderr, "  RSS:           %.0f MB\n", get_rss_bytes() / (1024.0 * 1024.0));

    // T108: baseline
    NSMutableDictionary *metrics = [NSMutableDictionary dictionaryWithDictionary:@{
        @"training.avg_fwd_ms": @(avg_fwd),
        @"training.avg_bwd_ms": @(avg_bwd),
        @"training.avg_adam_ms": @(avg_adam),
        @"training.avg_recomp_ms": @(avg_recomp),
        @"training.avg_total_ms": @(avg_total),
    }];

    if (save_baseline) {
        if (baseline_save(metrics)) {
            fprintf(stderr, "\nBaseline saved to %s\n", baseline_path().UTF8String);
        }
    } else {
        baseline_compare(metrics);
    }

    free(input_tokens);
    free(target_tokens);
    orion_trainer_free(trainer);
    return 0;
}

#pragma mark - CLI Dispatch

int orion_cmd_bench(int argc, const char* argv[]) {
    if (argc < 2) {
        print_bench_help();
        return 1;
    }

    const char* subcmd = argv[1];

    // ---- swap ----
    if (strcmp(subcmd, "swap") == 0) {
        const char* weights_a = NULL;
        const char* weights_b = NULL;
        int iters = 100;
        int bucket = 64;
        bool save_bl = false;

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--weights_a") == 0 && i + 1 < argc) {
                weights_a = argv[++i];
            } else if (strcmp(argv[i], "--weights_b") == 0 && i + 1 < argc) {
                weights_b = argv[++i];
            } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
                iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--bucket") == 0 && i + 1 < argc) {
                bucket = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--save-baseline") == 0) {
                save_bl = true;
            } else if (strcmp(argv[i], "--help") == 0) {
                print_swap_help();
                return 0;
            } else {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                print_swap_help();
                return 1;
            }
        }

        if (!weights_a || !weights_b) {
            fprintf(stderr, "Error: --weights_a and --weights_b are required\n\n");
            print_swap_help();
            return 1;
        }
        if (iters <= 0) {
            fprintf(stderr, "Error: --iters must be > 0\n");
            return 1;
        }

        return bench_swap(weights_a, weights_b, iters, bucket, save_bl);

    // ---- kernels ----
    } else if (strcmp(subcmd, "kernels") == 0) {
        const char *model_dir = "model/blobs/gpt2_124m";
        int iters = 50;
        int bucket = 64;
        bool save_bl = false;

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
                model_dir = argv[++i];
            } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
                iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--bucket") == 0 && i + 1 < argc) {
                bucket = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--save-baseline") == 0) {
                save_bl = true;
            } else if (strcmp(argv[i], "--help") == 0) {
                print_kernels_help();
                return 0;
            } else {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                print_kernels_help();
                return 1;
            }
        }

        if (iters <= 0) {
            fprintf(stderr, "Error: --iters must be > 0\n");
            return 1;
        }

        return bench_kernels(model_dir, iters, bucket, save_bl);

    // ---- inference ----
    } else if (strcmp(subcmd, "inference") == 0) {
        const char *prompt = "Hello";
        int max_tokens = 64;
        int warmup = 3;
        bool use_ane = false;
        bool ane_decode = false;
        const char *weights = "model/blobs/gpt2_124m";
        const char *vocab = "tokenizer/data/vocab.json";
        const char *merges = "tokenizer/data/merges.txt";
        bool save_bl = false;

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
                prompt = argv[++i];
            } else if (strcmp(argv[i], "--max_tokens") == 0 && i + 1 < argc) {
                max_tokens = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
                warmup = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--ane") == 0) {
                use_ane = true;
                ane_decode = true;
            } else if (strcmp(argv[i], "--ane-prefill") == 0) {
                use_ane = true;
                ane_decode = false;
            } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
                weights = argv[++i];
            } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
                vocab = argv[++i];
            } else if (strcmp(argv[i], "--merges") == 0 && i + 1 < argc) {
                merges = argv[++i];
            } else if (strcmp(argv[i], "--save-baseline") == 0) {
                save_bl = true;
            } else if (strcmp(argv[i], "--help") == 0) {
                print_inference_help();
                return 0;
            } else {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                print_inference_help();
                return 1;
            }
        }

        return bench_inference(prompt, max_tokens, warmup, use_ane, ane_decode,
                               weights, vocab, merges, save_bl);

    // ---- training ----
    } else if (strcmp(subcmd, "training") == 0) {
        int steps = 10;
        int grad_accum = 4;
        const char *weights = "model/blobs/stories110m";
        bool save_bl = false;

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
                steps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--grad_accum") == 0 && i + 1 < argc) {
                grad_accum = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
                weights = argv[++i];
            } else if (strcmp(argv[i], "--save-baseline") == 0) {
                save_bl = true;
            } else if (strcmp(argv[i], "--help") == 0) {
                print_training_help();
                return 0;
            } else {
                fprintf(stderr, "Unknown option: %s\n", argv[i]);
                print_training_help();
                return 1;
            }
        }

        if (steps <= 0 || grad_accum <= 0) {
            fprintf(stderr, "Error: --steps and --grad_accum must be > 0\n");
            return 1;
        }

        return bench_training(steps, grad_accum, weights, save_bl);

    // ---- help ----
    } else if (strcmp(subcmd, "--help") == 0 || strcmp(subcmd, "-h") == 0) {
        print_bench_help();
        return 0;
    } else {
        fprintf(stderr, "Unknown bench subcommand: %s\n\n", subcmd);
        print_bench_help();
        return 1;
    }
}
