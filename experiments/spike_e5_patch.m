// spike_e5_patch.m — T150: Weight region identification + patch verification
//
// For a single kernel, locates the exact byte range(s) where weight data
// is embedded in the compiled ANE binary (`data` file).
// Tests: patch bytes in compiled `data` → reload → eval → compare to fresh compile.
//
// Build:
//   cd /Users/murai-labs/Github/Orion && \
//   xcrun clang -O2 -fobjc-arc \
//     -framework Foundation -framework IOSurface -ldl -I . \
//     core/ane_runtime.m core/iosurface_tensor.m \
//     experiments/spike_e5_patch.m -o experiments/spike_e5_patch
//
// Run:
//   ./experiments/spike_e5_patch

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <stdio.h>
#import <math.h>
#import <string.h>
#import <sys/stat.h>
#import "core/ane_runtime.h"
#import "core/iosurface_tensor.h"

#pragma mark - BLOBFILE helper (matches test_mil_builder.m format)

static NSData* make_blobfile(const float *data, int count) {
    size_t fp16_size = count * sizeof(uint16_t);
    size_t total = 128 + fp16_size;
    uint8_t *b = (uint8_t *)calloc(total, 1);
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = (uint32_t)fp16_size;
    *(uint32_t *)(b + 80) = 128;
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < count; i++) {
        fp16[i] = (_Float16)data[i];
    }
    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

#pragma mark - MIL text (matching orion_mil_linear pattern)

static NSString* mil_linear(int in_dim, int out_dim, int seq) {
    return [NSString stringWithFormat:
        @"program(1.3)\n"
        "[buildInfo = dict<string, string>({"
        "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
        "{\"coremlc-version\", \"3505.4.1\"}, "
        "{\"coremltools-component-milinternal\", \"\"}, "
        "{\"coremltools-version\", \"9.0\"}"
        "})]\n"
        "{\n"
        "    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n"
        "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "        tensor<fp16, [%d,%d,1,1]> W = const()[name=string(\"W\"), "
            "val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w.bin\"), offset=uint64(64)))];\n"
        "        tensor<fp16, [1,%d,1,%d]> y = conv("
            "dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, "
            "weight=W, x=x)[name=string(\"y\")];\n"
        "    } -> (y);\n"
        "}\n",
        in_dim, seq, out_dim, in_dim, out_dim, in_dim, out_dim, seq];
}

#pragma mark - Direct ANE compile with temp dir access

// Compile MIL with weights and return the temp directory path.
// Does NOT clean up the temp dir so we can inspect/modify files.
static NSString* compile_get_tmpdir(const char *mil_text, NSDictionary *weight_dict) {
    @autoreleasepool {
        NSData *milData = [NSData dataWithBytes:mil_text length:strlen(mil_text)];
        NSDictionary *wdict = weight_dict ?: @{};

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            NSClassFromString(@"_ANEInMemoryModelDescriptor"),
            @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) return nil;

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            NSClassFromString(@"_ANEInMemoryModel"),
            @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) return nil;

        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];

        for (NSString *path in wdict) {
            NSDictionary *entry = wdict[path];
            NSData *data = entry[@"data"];
            if (data) {
                NSString *relPath = [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                                                    withString:@""];
                NSString *fullPath = [tmpDir stringByAppendingPathComponent:relPath];
                NSString *dir = [fullPath stringByDeletingLastPathComponent];
                [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
                [data writeToFile:fullPath atomically:YES];
            }
        }

        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            printf("compile failed: %s\n", e.description.UTF8String);
            return nil;
        }

        return tmpDir;
    }
}

#pragma mark - Direct load from temp dir

// Load a compiled model from a temp directory (without compiling).
// Returns an OrionProgram-compatible handle via eval.
typedef struct {
    void *model;   // _ANEInMemoryModel (retained)
    bool loaded;
} PatchedProgram;

static PatchedProgram* load_from_tmpdir(const char *mil_text, NSDictionary *weight_dict, NSString *tmpDir) {
    @autoreleasepool {
        NSData *milData = [NSData dataWithBytes:mil_text length:strlen(mil_text)];
        NSDictionary *wdict = weight_dict ?: @{};

        // Create descriptor + model (same MIL text + weights as original)
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            NSClassFromString(@"_ANEInMemoryModelDescriptor"),
            @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) { printf("desc failed\n"); return NULL; }

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            NSClassFromString(@"_ANEInMemoryModel"),
            @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) { printf("model failed\n"); return NULL; }

        // Get THIS model's expected tmp dir
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *modelDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];

        // Copy compiled artifacts from the source tmpDir to this model's dir
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:modelDir withIntermediateDirectories:YES attributes:nil error:nil];

        // Copy the data file and net.plist
        NSString *srcData = [tmpDir stringByAppendingPathComponent:@"data"];
        NSString *dstData = [modelDir stringByAppendingPathComponent:@"data"];
        NSString *srcPlist = [tmpDir stringByAppendingPathComponent:@"net.plist"];
        NSString *dstPlist = [modelDir stringByAppendingPathComponent:@"net.plist"];

        [fm copyItemAtPath:srcData toPath:dstData error:nil];
        [fm copyItemAtPath:srcPlist toPath:dstPlist error:nil];

        // Also copy weights dir and model.mil (might be needed for load)
        NSString *srcMil = [tmpDir stringByAppendingPathComponent:@"model.mil"];
        NSString *dstMil = [modelDir stringByAppendingPathComponent:@"model.mil"];
        [fm copyItemAtPath:srcMil toPath:dstMil error:nil];

        NSString *srcWeights = [tmpDir stringByAppendingPathComponent:@"weights"];
        NSString *dstWeights = [modelDir stringByAppendingPathComponent:@"weights"];
        if ([fm fileExistsAtPath:srcWeights]) {
            [fm copyItemAtPath:srcWeights toPath:dstWeights error:nil];
        }

        // Try to load WITHOUT compiling
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            printf("LOAD FAILED (no compile): %s\n",
                   e ? e.description.UTF8String : "unknown");
            return NULL;
        }

        PatchedProgram *pp = (PatchedProgram *)calloc(1, sizeof(PatchedProgram));
        pp->model = (void *)CFBridgingRetain(model);
        pp->loaded = true;
        return pp;
    }
}

static bool patched_eval(PatchedProgram *pp, IOSurfaceRef *inputs, int nin,
                          IOSurfaceRef *outputs, int nout) {
    if (!pp || !pp->loaded) return false;
    @autoreleasepool {
        Class g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
        Class g_AR = NSClassFromString(@"_ANERequest");

        NSMutableArray *inArr = [NSMutableArray arrayWithCapacity:nin];
        NSMutableArray *inIdx = [NSMutableArray arrayWithCapacity:nin];
        for (int i = 0; i < nin; i++) {
            id w = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_AIO, @selector(objectWithIOSurface:), inputs[i]);
            [inArr addObject:w];
            [inIdx addObject:@(i)];
        }

        NSMutableArray *outArr = [NSMutableArray arrayWithCapacity:nout];
        NSMutableArray *outIdx = [NSMutableArray arrayWithCapacity:nout];
        for (int i = 0; i < nout; i++) {
            id w = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_AIO, @selector(objectWithIOSurface:), outputs[i]);
            [outArr addObject:w];
            [outIdx addObject:@(i)];
        }

        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                      weightsBuffer:perfStats:procedureIndex:),
            inArr, inIdx, outArr, outIdx, nil, nil, @0);

        id model = (__bridge id)pp->model;
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &e);
        if (!ok && e) printf("patched_eval ERROR: %s\n", e.description.UTF8String);
        return ok;
    }
}

static void patched_release(PatchedProgram *pp) {
    if (!pp) return;
    @autoreleasepool {
        if (pp->loaded && pp->model) {
            id model = (__bridge id)pp->model;
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &e);
        }
        if (pp->model) {
            CFRelease(pp->model);
        }
        free(pp);
    }
}

#pragma mark - Main

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("=== T150: Weight Region Identification + Patch Test ===\n\n");

        if (!orion_ane_init()) {
            printf("FATAL: ANE init failed\n");
            return 1;
        }

        // Use 768x768 (Stories-scale) for realistic test
        int dim = 768;
        int seq = 32;
        int weight_count = dim * dim; // 589,824

        printf("Config: dim=%d, seq=%d, weights=%d (%.1f MB fp16)\n",
               dim, seq, weight_count, weight_count * 2.0 / 1024 / 1024);

        // Create two weight sets
        srand(42);
        float *weights_a = (float *)malloc(weight_count * sizeof(float));
        float *weights_b = (float *)malloc(weight_count * sizeof(float));
        for (int i = 0; i < weight_count; i++) {
            weights_a[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
            weights_b[i] = weights_a[i]; // Start identical
        }
        // Perturb ALL weights in B (simulating Adam step)
        for (int i = 0; i < weight_count; i++) {
            weights_b[i] += ((float)rand() / RAND_MAX - 0.5f) * 0.002f; // Small perturbation
        }

        NSData *blob_a = make_blobfile(weights_a, weight_count);
        NSData *blob_b = make_blobfile(weights_b, weight_count);

        NSString *mil = mil_linear(dim, dim, seq);
        NSDictionary *wdict_a = @{
            @"@model_path/weights/w.bin": @{@"offset": @0, @"data": blob_a}
        };
        NSDictionary *wdict_b = @{
            @"@model_path/weights/w.bin": @{@"offset": @0, @"data": blob_b}
        };

        // Create a fixed test input
        float *input_data = (float *)malloc(dim * seq * sizeof(float));
        srand(123);
        for (int i = 0; i < dim * seq; i++) {
            input_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }

        // ============================================================
        // Step 1: Compile with weights A (get reference output + tmpdir)
        // ============================================================
        printf("\n=== Step 1: Compile with weights A ===\n");
        NSString *tmpdir_a = compile_get_tmpdir(mil.UTF8String, wdict_a);
        if (!tmpdir_a) {
            printf("FATAL: Failed to compile A\n");
            return 1;
        }
        printf("Tmpdir A: %s\n", tmpdir_a.UTF8String);

        // Also compile normally for reference eval
        OrionProgram *prog_a = orion_compile_mil(mil.UTF8String, wdict_a, "ref_A");
        if (!prog_a) {
            printf("FATAL: Failed to compile ref_A\n");
            return 1;
        }

        IOSurfaceRef in_surf = orion_tensor_create(dim, seq);
        orion_tensor_write_f32(in_surf, input_data, dim * seq);
        IOSurfaceRef out_ref_a = orion_tensor_create(dim, seq);
        bool ok = orion_eval(prog_a, &in_surf, 1, &out_ref_a, 1);
        printf("Eval ref_A: %s\n", ok ? "OK" : "FAIL");

        float *result_a = (float *)malloc(dim * seq * sizeof(float));
        orion_tensor_read_f32(out_ref_a, result_a, dim * seq);
        printf("Output A[0..3]: %.6f %.6f %.6f %.6f\n",
               result_a[0], result_a[1], result_a[2], result_a[3]);

        // ============================================================
        // Step 2: Compile with weights B (reference output)
        // ============================================================
        printf("\n=== Step 2: Compile with weights B (reference) ===\n");
        NSString *tmpdir_b = compile_get_tmpdir(mil.UTF8String, wdict_b);
        OrionProgram *prog_b = orion_compile_mil(mil.UTF8String, wdict_b, "ref_B");
        if (!prog_b) {
            printf("FATAL: Failed to compile ref_B\n");
            return 1;
        }

        IOSurfaceRef out_ref_b = orion_tensor_create(dim, seq);
        ok = orion_eval(prog_b, &in_surf, 1, &out_ref_b, 1);
        printf("Eval ref_B: %s\n", ok ? "OK" : "FAIL");

        float *result_b = (float *)malloc(dim * seq * sizeof(float));
        orion_tensor_read_f32(out_ref_b, result_b, dim * seq);
        printf("Output B[0..3]: %.6f %.6f %.6f %.6f\n",
               result_b[0], result_b[1], result_b[2], result_b[3]);

        // ============================================================
        // Step 3: Compare data files (find exact weight mapping)
        // ============================================================
        printf("\n=== Step 3: Compare compiled data files ===\n");
        NSData *data_a = [NSData dataWithContentsOfFile:
            [tmpdir_a stringByAppendingPathComponent:@"data"]];
        NSData *data_b = [NSData dataWithContentsOfFile:
            [tmpdir_b stringByAppendingPathComponent:@"data"]];

        if (!data_a || !data_b) {
            printf("FATAL: Could not read data files\n");
            return 1;
        }

        printf("Data A: %zu bytes\n", (size_t)data_a.length);
        printf("Data B: %zu bytes\n", (size_t)data_b.length);

        // Also compare data_a with BLOBFILE A
        printf("\nComparing data_A vs BLOBFILE_A...\n");
        if (data_a.length == blob_a.length) {
            if ([data_a isEqualToData:blob_a]) {
                printf("  data == BLOBFILE (IDENTICAL!) -> data file IS the BLOBFILE\n");
            } else {
                const uint8_t *da = data_a.bytes;
                const uint8_t *ba = blob_a.bytes;
                int ndiff = 0;
                for (size_t i = 0; i < data_a.length; i++) {
                    if (da[i] != ba[i]) ndiff++;
                }
                printf("  data != BLOBFILE: %d bytes differ (%.1f%%)\n",
                       ndiff, 100.0 * ndiff / data_a.length);
            }
        } else {
            printf("  Different sizes: data=%zu, blob=%zu\n",
                   (size_t)data_a.length, (size_t)blob_a.length);
        }

        // ============================================================
        // Step 4: Patch approach 1 — replace data file with B's data
        // ============================================================
        printf("\n=== Step 4: Patch test — replace A's data with B's data ===\n");

        // Copy tmpdir_a, replace data file with data_b
        NSFileManager *fm = [NSFileManager defaultManager];
        NSString *patchDir = [NSTemporaryDirectory()
            stringByAppendingPathComponent:@"orion_patch_test"];
        [fm removeItemAtPath:patchDir error:nil];
        [fm copyItemAtPath:tmpdir_a toPath:patchDir error:nil];

        // Overwrite data file with B's compiled data
        NSString *patchDataPath = [patchDir stringByAppendingPathComponent:@"data"];
        [data_b writeToFile:patchDataPath atomically:YES];
        printf("Patched data file at: %s\n", patchDataPath.UTF8String);

        // Also overwrite the weight blob (the compiler may check this on load)
        NSString *patchWeightPath = [patchDir
            stringByAppendingPathComponent:@"weights/w.bin"];
        [blob_b writeToFile:patchWeightPath atomically:YES];

        // Try to load from patched directory
        printf("Attempting load from patched directory...\n");
        PatchedProgram *pp = load_from_tmpdir(mil.UTF8String, wdict_b, patchDir);
        if (pp) {
            printf("LOAD FROM PATCH: SUCCESS!\n");

            // Eval patched program
            IOSurfaceRef out_patch = orion_tensor_create(dim, seq);
            ok = patched_eval(pp, &in_surf, 1, &out_patch, 1);
            printf("Eval patched: %s\n", ok ? "OK" : "FAIL");

            if (ok) {
                float *result_patch = (float *)malloc(dim * seq * sizeof(float));
                orion_tensor_read_f32(out_patch, result_patch, dim * seq);
                printf("Output patch[0..3]: %.6f %.6f %.6f %.6f\n",
                       result_patch[0], result_patch[1], result_patch[2], result_patch[3]);

                // Compare patched output to reference B output
                float max_diff = 0;
                for (int i = 0; i < dim * seq; i++) {
                    float d = fabsf(result_patch[i] - result_b[i]);
                    if (d > max_diff) max_diff = d;
                }
                printf("Max diff (patch vs ref_B): %.8f\n", max_diff);
                if (max_diff < 1e-3f) {
                    printf("PASS: Patched output matches fresh compile!\n");
                    printf("DELTA PATCHING IS FEASIBLE.\n");
                } else {
                    printf("FAIL: Outputs differ significantly.\n");
                }
                free(result_patch);
            }

            CFRelease(out_patch);
            patched_release(pp);
        } else {
            printf("LOAD FROM PATCH: FAILED\n");
            printf("ANE rejected the patched artifacts.\n");

            // ============================================================
            // Step 5: Alternative — try compiling with B's weights but
            // using A's model identity (different approach)
            // ============================================================
            printf("\n=== Step 5: Alternative — compile B then load A's data ===\n");
            printf("(Trying if compile+load is needed but we can swap data between loads)\n");

            // Compile with A, but swap the data file before load
            // This requires a modified compile flow — skip for now
            printf("Alternative approaches to try:\n");
            printf("  1. Patch data file between compile and load steps\n");
            printf("  2. Unload model A, patch data file, reload same model\n");
            printf("  3. Use IOSurface weight buffer in eval request\n");
        }

        // ============================================================
        // Step 5b: Try patching via unload → patch → reload
        // ============================================================
        printf("\n=== Step 5b: Unload A → patch data → reload ===\n");
        {
            // Compile a fresh program with weights A
            // Then unload, patch the data file, and reload
            NSData *milData = [NSData dataWithBytes:mil.UTF8String length:mil.length];
            NSDictionary *wdict = wdict_a;

            id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
                NSClassFromString(@"_ANEInMemoryModelDescriptor"),
                @selector(modelWithMILText:weights:optionsPlist:),
                milData, wdict, nil);
            id model = ((id(*)(Class,SEL,id))objc_msgSend)(
                NSClassFromString(@"_ANEInMemoryModel"),
                @selector(inMemoryModelWithDescriptor:), desc);

            id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
            NSString *modelDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];

            // Set up temp dir
            [fm createDirectoryAtPath:[modelDir stringByAppendingPathComponent:@"weights"]
                withIntermediateDirectories:YES attributes:nil error:nil];
            [milData writeToFile:[modelDir stringByAppendingPathComponent:@"model.mil"]
                      atomically:YES];
            for (NSString *path in wdict) {
                NSDictionary *entry = wdict[path];
                NSData *d = entry[@"data"];
                if (d) {
                    NSString *relPath = [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                                                        withString:@""];
                    NSString *fullPath = [modelDir stringByAppendingPathComponent:relPath];
                    NSString *dir = [fullPath stringByDeletingLastPathComponent];
                    [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
                    [d writeToFile:fullPath atomically:YES];
                }
            }

            // Compile with A
            NSError *e = nil;
            BOOL cok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
            if (!cok) { printf("compile failed\n"); goto cleanup; }

            // Load with A
            cok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            if (!cok) { printf("initial load failed\n"); goto cleanup; }
            printf("Initial compile+load with A: OK\n");

            // Verify output matches reference A
            {
                Class g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
                Class g_AR = NSClassFromString(@"_ANERequest");

                IOSurfaceRef out = orion_tensor_create(dim, seq);

                id inw = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_AIO, @selector(objectWithIOSurface:), in_surf);
                id outw = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_AIO, @selector(objectWithIOSurface:), out);

                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                    g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                              weightsBuffer:perfStats:procedureIndex:),
                    @[inw], @[@0], @[outw], @[@0], nil, nil, @0);

                cok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, req, &e);
                printf("Eval with A weights: %s\n", cok ? "OK" : "FAIL");

                float *res = (float *)malloc(dim * seq * sizeof(float));
                orion_tensor_read_f32(out, res, dim * seq);
                printf("Output A[0..3]: %.6f %.6f %.6f %.6f\n",
                       res[0], res[1], res[2], res[3]);
                free(res);
                CFRelease(out);
            }

            // Unload
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &e);
            printf("Unloaded model.\n");

            // Patch: replace data file with B's compiled data
            NSString *dataPath = [modelDir stringByAppendingPathComponent:@"data"];
            [data_b writeToFile:dataPath atomically:YES];
            // Also replace weights
            NSString *wPath = [modelDir stringByAppendingPathComponent:@"weights/w.bin"];
            [blob_b writeToFile:wPath atomically:YES];
            printf("Patched data + weights in temp dir.\n");

            // Reload (no recompile)
            e = nil;
            cok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
            if (!cok) {
                printf("RELOAD FAILED: %s\n", e ? e.description.UTF8String : "unknown");
                printf("ANE caches the compiled artifact in memory after compile.\n");
                printf("Patching filesystem files does not affect loaded program.\n");
            } else {
                printf("RELOAD: SUCCESS!\n");

                // Eval and compare
                Class g_AIO = NSClassFromString(@"_ANEIOSurfaceObject");
                Class g_AR = NSClassFromString(@"_ANERequest");

                IOSurfaceRef out = orion_tensor_create(dim, seq);

                id inw = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_AIO, @selector(objectWithIOSurface:), in_surf);
                id outw = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                    g_AIO, @selector(objectWithIOSurface:), out);

                id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
                    g_AR,
                    @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                              weightsBuffer:perfStats:procedureIndex:),
                    @[inw], @[@0], @[outw], @[@0], nil, nil, @0);

                cok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, req, &e);
                printf("Eval after patch+reload: %s\n", cok ? "OK" : "FAIL");

                if (cok) {
                    float *res = (float *)malloc(dim * seq * sizeof(float));
                    orion_tensor_read_f32(out, res, dim * seq);
                    printf("Output patch[0..3]: %.6f %.6f %.6f %.6f\n",
                           res[0], res[1], res[2], res[3]);
                    printf("Expected B[0..3]:   %.6f %.6f %.6f %.6f\n",
                           result_b[0], result_b[1], result_b[2], result_b[3]);

                    float max_diff = 0;
                    for (int i = 0; i < dim * seq; i++) {
                        float d = fabsf(res[i] - result_b[i]);
                        if (d > max_diff) max_diff = d;
                    }
                    float max_diff_a = 0;
                    for (int i = 0; i < dim * seq; i++) {
                        float d = fabsf(res[i] - result_a[i]);
                        if (d > max_diff_a) max_diff_a = d;
                    }
                    printf("Max diff vs ref_B: %.8f\n", max_diff);
                    printf("Max diff vs ref_A: %.8f\n", max_diff_a);

                    if (max_diff < 1e-3f) {
                        printf("\nPASS: Unload→patch→reload produces correct output!\n");
                        printf("DELTA PATCHING VIA UNLOAD/RELOAD IS FEASIBLE.\n");
                    } else if (max_diff_a < 1e-3f) {
                        printf("\nReload used cached data (still outputs A, not B).\n");
                        printf("Filesystem patch did not affect loaded program.\n");
                    } else {
                        printf("\nOutput doesn't match either A or B — corruption.\n");
                    }
                    free(res);
                }
                CFRelease(out);
            }

            // Unload final
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &e);
        }

cleanup:
        // Cleanup
        orion_release_program(prog_a);
        orion_release_program(prog_b);
        CFRelease(in_surf);
        CFRelease(out_ref_a);
        CFRelease(out_ref_b);
        free(weights_a); free(weights_b);
        free(input_data); free(result_a); free(result_b);

        printf("\n=== T150 Spike Complete ===\n");
        printf("Compile count: %d (of ~119 budget)\n", orion_compile_count());

        return 0;
    }
}
