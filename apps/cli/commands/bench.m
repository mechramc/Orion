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

// T087: CLI bench swap arguments
// T089: Wire bench swap into CLI
//
// Usage: orion bench swap --weights_a PATH --weights_b PATH --iters N [--bucket N]

static void print_bench_help(void) {
    fprintf(stderr,
        "Usage: orion bench <subcommand> [options]\n"
        "\n"
        "Subcommands:\n"
        "  swap    Weight swap endurance benchmark\n"
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
        "  --help             Show this help message\n"
    );
}

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

// T088/T089: swap endurance benchmark core
// Compiles a simple add program with weight binding, evals, evicts, repeats.
// Tests: program cache store/evict cycle, ANE resource cleanup, RSS stability.
static int bench_swap(const char* weights_a, const char* weights_b,
                      int iters, int bucket) {
    if (!orion_ane_init()) {
        fprintf(stderr, "bench swap: ANE init failed\n");
        return 1;
    }

    // We use a simple MIL program (add) as a proxy for weight-baked programs.
    // The point is to stress-test compile → cache → evict → recompile cycles.
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

    // Allocate I/O surfaces
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

    // Fill inputs
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

        // Compile (or cache hit)
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

        // Eval
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

        // Evict the OTHER weights_id to force recompile on next swap
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

    fprintf(stderr, "\n=== bench swap results ===\n");
    fprintf(stderr, "  iterations:      %d\n", iters);
    fprintf(stderr, "  compiles:        %d (of %d total process)\n",
            compile_count, orion_compile_count());
    fprintf(stderr, "  avg compile:     %.2f ms\n",
            compile_count > 0 ? total_compile_ms / compile_count : 0);
    fprintf(stderr, "  avg eval:        %.2f ms\n", total_eval_ms / iters);
    fprintf(stderr, "  avg evict:       %.2f ms\n", total_evict_ms / iters);
    fprintf(stderr, "  RSS start:       %.1f MB\n", rss_start / (1024.0 * 1024.0));
    fprintf(stderr, "  RSS end:         %.1f MB\n", rss_end / (1024.0 * 1024.0));
    fprintf(stderr, "  RSS ratio:       %.2fx\n",
            rss_start > 0 ? (double)rss_end / rss_start : 0);
    fprintf(stderr, "  status:          %s\n",
            (rss_start > 0 && (double)rss_end / rss_start < 2.0) ? "PASS" : "WARN (RSS > 2x)");

    CFRelease(ioX); CFRelease(ioY); CFRelease(ioZ);
    return 0;
}

int orion_cmd_bench(int argc, const char* argv[]) {
    if (argc < 2) {
        print_bench_help();
        return 1;
    }

    const char* subcmd = argv[1];

    if (strcmp(subcmd, "swap") == 0) {
        // Parse swap arguments
        const char* weights_a = NULL;
        const char* weights_b = NULL;
        int iters = 100;
        int bucket = 64;

        for (int i = 2; i < argc; i++) {
            if (strcmp(argv[i], "--weights_a") == 0 && i + 1 < argc) {
                weights_a = argv[++i];
            } else if (strcmp(argv[i], "--weights_b") == 0 && i + 1 < argc) {
                weights_b = argv[++i];
            } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
                iters = atoi(argv[++i]);
            } else if (strcmp(argv[i], "--bucket") == 0 && i + 1 < argc) {
                bucket = atoi(argv[++i]);
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

        return bench_swap(weights_a, weights_b, iters, bucket);

    } else if (strcmp(subcmd, "--help") == 0 || strcmp(subcmd, "-h") == 0) {
        print_bench_help();
        return 0;
    } else {
        fprintf(stderr, "Unknown bench subcommand: %s\n\n", subcmd);
        print_bench_help();
        return 1;
    }
}
