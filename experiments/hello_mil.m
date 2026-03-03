// hello_mil.m — T008: Hello MIL proof-of-concept
// Compiles a trivial MIL program (z = x + y), evaluates on ANE, reads result.
// This validates the entire ANE pipeline: dlopen → descriptor → compile → load → eval → read.
//
// Build:
//   xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl hello_mil.m -o hello_mil
// Run:
//   ./hello_mil

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>

// ---------------------------------------------------------------------------
// MIL text: z = add(x, y) on fp16 [1, 4, 1, 4] tensors
// Deliberately tiny (4 channels, 4 spatial) so compile is fast.
// ---------------------------------------------------------------------------
#define CH 256
#define SP 64

static NSString *hello_mil_text(void) {
    // Use fp32 I/O (matching upstream pattern) with internal fp16 cast.
    // ANE IOSurface buffers are sized for fp32 (4 bytes/elem).
    // Dimensions: [1, 256, 1, 64] — known to work on ANE from upstream.
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
        CH, SP, CH, SP, CH, SP, CH, SP, CH, SP, CH, SP];
}

// ---------------------------------------------------------------------------
// IOSurface helpers
// ---------------------------------------------------------------------------
static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

static void surface_write_f32(IOSurfaceRef s, const float *data, size_t count) {
    IOSurfaceLock(s, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(s), data, count * sizeof(float));
    IOSurfaceUnlock(s, 0, NULL);
}

static void surface_read_f32(IOSurfaceRef s, float *data, size_t count) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(s), count * sizeof(float));
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    @autoreleasepool {
        printf("=== Hello MIL — ANE Proof of Concept ===\n\n");

        // Step 1: Load ANE framework
        printf("[1/7] Loading AppleNeuralEngine framework...\n");
        void *h = dlopen("/System/Library/PrivateFrameworks/"
                         "AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        if (!h) {
            printf("FAIL: dlopen: %s\n", dlerror());
            return 1;
        }

        Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM  = NSClassFromString(@"_ANEInMemoryModel");
        Class AR   = NSClassFromString(@"_ANERequest");
        Class AIO  = NSClassFromString(@"_ANEIOSurfaceObject");

        if (!Desc || !IMM || !AR || !AIO) {
            printf("FAIL: Could not resolve ANE classes.\n");
            printf("  Desc=%p IMM=%p AR=%p AIO=%p\n",
                   (__bridge void*)Desc, (__bridge void*)IMM,
                   (__bridge void*)AR, (__bridge void*)AIO);
            return 1;
        }
        printf("  OK — all 4 ANE classes resolved.\n");

        // Step 2: Build MIL descriptor (no weights needed for add)
        printf("[2/7] Creating MIL descriptor...\n");
        NSString *milText = hello_mil_text();
        NSData *milData = [milText dataUsingEncoding:NSUTF8StringEncoding];

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, @{}, nil);
        if (!desc) { printf("FAIL: modelWithMILText returned nil.\n"); return 1; }
        printf("  OK — descriptor created.\n");

        // Step 3: Create in-memory model + pre-populate temp dir
        printf("[3/7] Creating in-memory model...\n");
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) { printf("FAIL: inMemoryModelWithDescriptor returned nil.\n"); return 1; }

        // Pre-populate temp directory (ANE compiler reads from here)
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:tmpDir withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        printf("  OK — model created, temp dir: %s\n", [tmpDir UTF8String]);

        // Step 4: Compile
        printf("[4/7] Compiling MIL to ANE bytecode...\n");
        mach_timebase_info_data_t tb;
        mach_timebase_info(&tb);
        uint64_t t0 = mach_absolute_time();

        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        double compileMs = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;

        if (!ok) {
            printf("FAIL: compile returned NO.\n");
            if (e) printf("  Error: %s\n", [[e description] UTF8String]);
            return 1;
        }
        printf("  OK — compiled in %.1f ms.\n", compileMs);

        // Step 5: Load into ANE
        printf("[5/7] Loading program into ANE...\n");
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            printf("FAIL: load returned NO.\n");
            if (e) printf("  Error: %s\n", [[e description] UTF8String]);
            return 1;
        }
        printf("  OK — loaded.\n");

        // Step 6: Create IOSurface tensors and evaluate
        printf("[6/7] Evaluating z = x + y on ANE...\n");

        int channels = CH, spatial = SP;
        int nelems = channels * spatial;
        size_t bytes = nelems * sizeof(float); // fp32 IOSurface

        IOSurfaceRef ioX = make_surface(bytes);
        IOSurfaceRef ioY = make_surface(bytes);
        IOSurfaceRef ioZ = make_surface(bytes);

        // Fill x with 1.0, y with 2.0
        float *x_data = (float *)calloc(nelems, sizeof(float));
        float *y_data = (float *)calloc(nelems, sizeof(float));
        for (int i = 0; i < nelems; i++) {
            x_data[i] = 1.0f;
            y_data[i] = 2.0f;
        }
        surface_write_f32(ioX, x_data, nelems);
        surface_write_f32(ioY, y_data, nelems);
        free(x_data);
        free(y_data);

        // Build request
        id wX = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            AIO, @selector(objectWithIOSurface:), ioX);
        id wY = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            AIO, @selector(objectWithIOSurface:), ioY);
        id wZ = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
            AIO, @selector(objectWithIOSurface:), ioZ);

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                      weightsBuffer:perfStats:procedureIndex:),
            @[wX, wY], @[@0, @1], @[wZ], @[@0], nil, nil, @0);

        // Evaluate
        t0 = mach_absolute_time();
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &e);
        double evalMs = (double)(mach_absolute_time() - t0) * tb.numer / tb.denom / 1e6;

        if (!ok) {
            printf("FAIL: evaluate returned NO.\n");
            if (e) printf("  Error: %s\n", [[e description] UTF8String]);
            // Cleanup
            CFRelease(ioX); CFRelease(ioY); CFRelease(ioZ);
            [fm removeItemAtPath:tmpDir error:nil];
            return 1;
        }
        printf("  OK — eval completed in %.3f ms.\n", evalMs);

        // Step 7: Verify results
        printf("[7/7] Reading and verifying results...\n");
        float *z_data = (float *)calloc(nelems, sizeof(float));
        surface_read_f32(ioZ, z_data, nelems);

        int pass = 1, mismatches = 0;
        for (int i = 0; i < nelems; i++) {
            if (z_data[i] < 2.9f || z_data[i] > 3.1f) {
                if (mismatches < 5) {
                    printf("  MISMATCH at [%d]: got %.4f, expected 3.0\n", i, z_data[i]);
                }
                mismatches++;
                pass = 0;
            }
        }
        if (mismatches > 5) printf("  ... and %d more mismatches\n", mismatches - 5);

        printf("\n========================================\n");
        if (pass) {
            printf("PASS: z = x + y computed correctly on ANE!\n");
            printf("  All %d elements = 3.0 (1.0 + 2.0)\n", nelems);
        } else {
            printf("FAIL: %d/%d elements mismatched.\n", mismatches, nelems);
        }
        printf("  Compile: %.1f ms | Eval: %.3f ms\n", compileMs, evalMs);
        printf("  Tensor shape: [1, %d, 1, %d] fp32→fp16→fp32\n", channels, spatial);
        printf("========================================\n");

        // Print first few values
        printf("\nFirst 8 output values: ");
        for (int i = 0; i < 8 && i < nelems; i++) {
            printf("%.4f ", z_data[i]);
        }
        printf("\n");
        free(z_data);

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioX); CFRelease(ioY); CFRelease(ioZ);
        [fm removeItemAtPath:tmpDir error:nil];

        return pass ? 0 : 1;
    }
}
