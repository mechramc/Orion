// spike_single_token.m — T099: ANE single-token dispatch feasibility spike
//
// Tests: Can ANE handle seq_len=1 dispatch? What is the minimum practical tensor size?
// Tests three programs at various seq_lens:
//   1. Simple add (baseline)
//   2. 1×1 conv (GPT-2 matmul proxy)
//   3. Multi-layer conv chain (FFN proxy: conv→gelu→conv)
//
// Build:
//   xcrun clang -O2 -fobjc-arc \
//     -framework Foundation -framework IOSurface -ldl -I . \
//     core/ane_runtime.m core/iosurface_tensor.m \
//     experiments/spike_single_token.m -o experiments/spike_single_token
//
// Run:
//   ./experiments/spike_single_token

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <stdio.h>
#import <math.h>
#import <sys/time.h>
#import "core/ane_runtime.h"

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes), (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1, (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes), (id)kIOSurfacePixelFormat: @0});
}

static IOSurfaceRef make_surface_fill(int count, float val) {
    IOSurfaceRef s = make_surface(count * sizeof(float));
    IOSurfaceLock(s, 0, NULL);
    float *p = IOSurfaceGetBaseAddress(s);
    for (int i = 0; i < count; i++) p[i] = val;
    IOSurfaceUnlock(s, 0, NULL);
    return s;
}

#pragma mark - MIL Generators

// Program 1: z = x + y (simplest possible)
static NSString* mil_add(int ch, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
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
        ch, seq, ch, seq, ch, seq, ch, seq, ch, seq, ch, seq];
}

// Program 2: y = conv2d(x, W, b) — 1×1 conv (GPT-2 linear layer proxy)
// Uses proper MIL conv syntax with all required named parameters
static NSString* mil_conv(int ch_in, int ch_out, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to16, x=x)[name=string(\"cx\")];\n"
        // Conv constants (required by MIL)
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        // Weight + bias
        "        tensor<fp16, [%d, %d, 1, 1]> W = const()[name=string(\"W\"), val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/W.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, 1]> b = const()[name=string(\"b\"), val=tensor<fp16, [1, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/b.bin\"), offset=uint64(64)))];\n"
        // Conv + bias add
        "        tensor<fp16, [1, %d, 1, %d]> c = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W, x=x16)[name=string(\"conv_op\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = add(x=c, y=b)[name=string(\"bias_add\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to32, x=y16)[name=string(\"out\")];\n"
        "    } -> (y);\n"
        "}\n",
        ch_in, seq,        // input
        ch_in, seq,        // cast input
        ch_out, ch_in, ch_out, ch_in,  // W shape
        ch_out, ch_out,    // b shape
        ch_out, seq,       // conv output
        ch_out, seq,       // bias add output
        ch_out, seq];      // final cast output
}

// Program 3: FFN proxy: x → conv1 → gelu_approx → conv2
// Uses proper MIL conv syntax. GELU decomposed to tanh approximation.
static NSString* mil_ffn(int d_model, int hidden, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n"
        "        string to16 = const()[name=string(\"to16\"), val=string(\"fp16\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype=to16, x=x)[name=string(\"cx\")];\n"
        // Conv constants (shared by both convs)
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        // Up projection: d_model → hidden
        "        tensor<fp16, [%d, %d, 1, 1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/W1.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, 1]> b1 = const()[name=string(\"b1\"), val=tensor<fp16, [1, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/b1.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> up = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W1, x=x16)[name=string(\"up\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> h = add(x=up, y=b1)[name=string(\"h\")];\n"
        // GELU approximation: 0.5 * h * (1 + tanh(sqrt(2/pi) * (h + 0.044715 * h^3)))
        "        tensor<fp16, [1, %d, 1, %d]> h2 = mul(x=h, y=h)[name=string(\"h2\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> h3 = mul(x=h2, y=h)[name=string(\"h3\")];\n"
        "        tensor<fp16, []> c1 = const()[name=string(\"c1\"), val=fp16(0.044715)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> t1 = mul(x=h3, y=c1)[name=string(\"t1\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> t2 = add(x=h, y=t1)[name=string(\"t2\")];\n"
        "        tensor<fp16, []> c2 = const()[name=string(\"c2\"), val=fp16(0.7978845608)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> t3 = mul(x=t2, y=c2)[name=string(\"t3\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> t4 = tanh(x=t3)[name=string(\"t4\")];\n"
        "        tensor<fp16, []> one = const()[name=string(\"one\"), val=fp16(1.0)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> t5 = add(x=t4, y=one)[name=string(\"t5\")];\n"
        "        tensor<fp16, []> half = const()[name=string(\"half\"), val=fp16(0.5)];\n"
        "        tensor<fp16, [1, %d, 1, %d]> t6 = mul(x=h, y=half)[name=string(\"t6\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> act = mul(x=t6, y=t5)[name=string(\"gelu\")];\n"
        // Down projection: hidden → d_model
        "        tensor<fp16, [%d, %d, 1, 1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/W2.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, 1]> b2 = const()[name=string(\"b2\"), val=tensor<fp16, [1, %d, 1, 1]>(BLOBFILE(path=string(\"@model_path/b2.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1, %d, 1, %d]> dn = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=W2, x=act)[name=string(\"dn\")];\n"
        "        tensor<fp16, [1, %d, 1, %d]> y16 = add(x=dn, y=b2)[name=string(\"y16\")];\n"
        "        string to32 = const()[name=string(\"to32\"), val=string(\"fp32\")];\n"
        "        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype=to32, x=y16)[name=string(\"out\")];\n"
        "    } -> (y);\n"
        "}\n",
        d_model, seq,                // input
        d_model, seq,                // cast
        hidden, d_model, hidden, d_model,  // W1
        hidden, hidden,              // b1
        hidden, seq,                 // up conv output
        hidden, seq,                 // h (up + bias)
        hidden, seq,                 // h2
        hidden, seq,                 // h3
        hidden, seq,                 // t1
        hidden, seq,                 // t2
        hidden, seq,                 // t3
        hidden, seq,                 // t4
        hidden, seq,                 // t5
        hidden, seq,                 // t6
        hidden, seq,                 // gelu
        d_model, hidden, d_model, hidden,  // W2
        d_model, d_model,            // b2
        d_model, seq,                // down conv output
        d_model, seq,                // y16 (down + bias)
        d_model, seq];               // final cast
}

#pragma mark - BLOBFILE Writer

// Write a BLOBFILE with 128-byte header + fp16 data (matching Orion format).
// Header layout: [0]: version=1, [4]: dtype=2(fp16),
// [64-67]: 0xDEADBEEF (chunk magic), [68]: chunk_count=1,
// [72-75]: data_size, [80-83]: data_offset=128
static NSData* make_blobfile(int num_elements, float fill_value) {
    size_t data_bytes = num_elements * sizeof(uint16_t); // fp16
    size_t total = 128 + data_bytes;

    uint8_t *buf = (uint8_t *)calloc(total, 1);
    // BLOBFILE header (matching test_train_smoke.m / stories_train.m format)
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 1;
    *(uint32_t *)(buf + 72) = (uint32_t)data_bytes;
    *(uint32_t *)(buf + 80) = 128;

    // Fill fp16 data at offset 128
    _Float16 *fp16_data = (_Float16 *)(buf + 128);
    _Float16 fv = (_Float16)fill_value;
    for (int i = 0; i < num_elements; i++) fp16_data[i] = fv;

    NSData *blob = [NSData dataWithBytes:buf length:total];
    free(buf);
    return blob;
}

#pragma mark - Test Runner

typedef struct {
    const char *name;
    int ch_in;
    int ch_out;
    int seq;
    int n_inputs;
    double compile_ms;
    double eval_ms;
    bool compiled;
    bool evaled;
} TestResult;

static void test_add(int ch, int seq, TestResult *result) {
    result->name = "add";
    result->ch_in = ch;
    result->ch_out = ch;
    result->seq = seq;
    result->n_inputs = 2;

    NSString *mil = mil_add(ch, seq);

    double t0 = time_ms();
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, @{}, "spike_add");
    double t1 = time_ms();
    result->compile_ms = t1 - t0;
    result->compiled = (prog != NULL);

    if (!prog) { result->evaled = false; return; }

    int count = ch * seq;
    IOSurfaceRef ioX = make_surface_fill(count, 1.0f);
    IOSurfaceRef ioY = make_surface_fill(count, 2.0f);
    IOSurfaceRef ioZ = make_surface(count * sizeof(float));

    IOSurfaceRef ins[] = {ioX, ioY};
    IOSurfaceRef outs[] = {ioZ};

    // Warmup
    orion_eval(prog, ins, 2, outs, 1);

    // Measure (average of 10 runs)
    double t2 = time_ms();
    int n_runs = 10;
    for (int i = 0; i < n_runs; i++) {
        orion_eval(prog, ins, 2, outs, 1);
    }
    double t3 = time_ms();
    result->eval_ms = (t3 - t2) / n_runs;
    result->evaled = true;

    orion_release_program(prog);
    CFRelease(ioX); CFRelease(ioY); CFRelease(ioZ);
}

static void test_conv(int ch_in, int ch_out, int seq, TestResult *result) {
    result->name = "conv1x1";
    result->ch_in = ch_in;
    result->ch_out = ch_out;
    result->seq = seq;
    result->n_inputs = 1;

    NSString *mil = mil_conv(ch_in, ch_out, seq);

    // Create weight blobs
    NSData *W_blob = make_blobfile(ch_out * ch_in, 0.001f);
    NSData *b_blob = make_blobfile(ch_out, 0.0f);
    NSDictionary *wdict = @{
        @"@model_path/W.bin": @{@"offset": @0, @"data": W_blob},
        @"@model_path/b.bin": @{@"offset": @0, @"data": b_blob}
    };

    double t0 = time_ms();
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "spike_conv");
    double t1 = time_ms();
    result->compile_ms = t1 - t0;
    result->compiled = (prog != NULL);

    if (!prog) { result->evaled = false; return; }

    int in_count = ch_in * seq;
    int out_count = ch_out * seq;
    IOSurfaceRef ioX = make_surface_fill(in_count, 1.0f);
    IOSurfaceRef ioY = make_surface(out_count * sizeof(float));

    IOSurfaceRef ins[] = {ioX};
    IOSurfaceRef outs[] = {ioY};

    // Warmup
    orion_eval(prog, ins, 1, outs, 1);

    // Measure
    double t2 = time_ms();
    int n_runs = 10;
    for (int i = 0; i < n_runs; i++) {
        orion_eval(prog, ins, 1, outs, 1);
    }
    double t3 = time_ms();
    result->eval_ms = (t3 - t2) / n_runs;
    result->evaled = true;

    orion_release_program(prog);
    CFRelease(ioX); CFRelease(ioY);
}

static void test_ffn(int d_model, int hidden, int seq, TestResult *result) {
    result->name = "ffn";
    result->ch_in = d_model;
    result->ch_out = d_model;
    result->seq = seq;
    result->n_inputs = 1;

    NSString *mil = mil_ffn(d_model, hidden, seq);

    // Create weight blobs: W1 [hidden, d_model], b1 [hidden], W2 [d_model, hidden], b2 [d_model]
    NSData *W1_blob = make_blobfile(hidden * d_model, 0.001f);
    NSData *b1_blob = make_blobfile(hidden, 0.0f);
    NSData *W2_blob = make_blobfile(d_model * hidden, 0.001f);
    NSData *b2_blob = make_blobfile(d_model, 0.0f);
    NSDictionary *wdict = @{
        @"@model_path/W1.bin": @{@"offset": @0, @"data": W1_blob},
        @"@model_path/b1.bin": @{@"offset": @0, @"data": b1_blob},
        @"@model_path/W2.bin": @{@"offset": @0, @"data": W2_blob},
        @"@model_path/b2.bin": @{@"offset": @0, @"data": b2_blob}
    };

    double t0 = time_ms();
    OrionProgram *prog = orion_compile_mil(mil.UTF8String, wdict, "spike_ffn");
    double t1 = time_ms();
    result->compile_ms = t1 - t0;
    result->compiled = (prog != NULL);

    if (!prog) { result->evaled = false; return; }

    int count = d_model * seq;
    IOSurfaceRef ioX = make_surface_fill(count, 1.0f);
    IOSurfaceRef ioY = make_surface(count * sizeof(float));

    IOSurfaceRef ins[] = {ioX};
    IOSurfaceRef outs[] = {ioY};

    // Warmup
    orion_eval(prog, ins, 1, outs, 1);

    // Measure
    double t2 = time_ms();
    int n_runs = 10;
    for (int i = 0; i < n_runs; i++) {
        orion_eval(prog, ins, 1, outs, 1);
    }
    double t3 = time_ms();
    result->eval_ms = (t3 - t2) / n_runs;
    result->evaled = true;

    orion_release_program(prog);
    CFRelease(ioX); CFRelease(ioY);
}

int main(void) {
    @autoreleasepool {
        printf("=== T099: ANE Single-Token Dispatch Spike ===\n\n");

        if (!orion_ane_init()) {
            fprintf(stderr, "FATAL: ANE not available\n");
            return 1;
        }

        // GPT-2 124M dimensions
        int d_model = 768;
        int hidden = 3072; // 4 * d_model
        int ch_small = 256; // for add tests

        // Test sequence lengths: from single token up to prefill bucket
        int seq_lens[] = {1, 2, 4, 8, 16, 32, 64};
        int n_seq = sizeof(seq_lens) / sizeof(seq_lens[0]);

        printf("Test 1: Simple add [%d, seq]\n", ch_small);
        printf("%-6s | %-8s | %-8s | %-6s\n", "seq", "compile", "eval", "status");
        printf("-------|----------|----------|-------\n");
        for (int i = 0; i < n_seq; i++) {
            TestResult r = {0};
            test_add(ch_small, seq_lens[i], &r);
            printf("%-6d | %6.1f ms | %6.3f ms | %s%s\n",
                   r.seq, r.compile_ms, r.eval_ms,
                   r.compiled ? "OK" : "FAIL-COMPILE",
                   r.evaled ? "" : " FAIL-EVAL");
        }

        printf("\nTest 2: 1x1 conv [%d → %d, seq] (GPT-2 linear proxy)\n", d_model, d_model);
        printf("%-6s | %-8s | %-8s | %-6s\n", "seq", "compile", "eval", "status");
        printf("-------|----------|----------|-------\n");
        for (int i = 0; i < n_seq; i++) {
            TestResult r = {0};
            test_conv(d_model, d_model, seq_lens[i], &r);
            printf("%-6d | %6.1f ms | %6.3f ms | %s%s\n",
                   r.seq, r.compile_ms, r.eval_ms,
                   r.compiled ? "OK" : "FAIL-COMPILE",
                   r.evaled ? "" : " FAIL-EVAL");
        }

        printf("\nTest 3: FFN [%d → %d → %d, seq] (GPT-2 FFN proxy with GELU)\n",
               d_model, hidden, d_model);
        printf("%-6s | %-8s | %-8s | %-6s\n", "seq", "compile", "eval", "status");
        printf("-------|----------|----------|-------\n");
        for (int i = 0; i < n_seq; i++) {
            TestResult r = {0};
            test_ffn(d_model, hidden, seq_lens[i], &r);
            printf("%-6d | %6.1f ms | %6.3f ms | %s%s\n",
                   r.seq, r.compile_ms, r.eval_ms,
                   r.compiled ? "OK" : "FAIL-COMPILE",
                   r.evaled ? "" : " FAIL-EVAL");
        }

        printf("\n=== Summary ===\n");
        printf("Total compiles used: %d (limit ~119)\n", orion_compile_count());
        printf("\nConclusion: Check if seq=1 compiles and evals. If not, find minimum viable seq_len.\n");
        printf("If eval latency at min viable seq_len is < 5ms, v3 ANE decode is feasible.\n");

        return 0;
    }
}
