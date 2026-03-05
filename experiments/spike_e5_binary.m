// spike_e5_binary.m — T149: E5 binary format analysis
//
// Compiles the SAME MIL program with TWO different weight sets.
// Binary diffs the compiled artifacts to determine:
//   1. Whether weight data appears contiguously in the E5 binary
//   2. Whether weight offsets are predictable
//   3. Whether delta patching is feasible
//
// Build:
//   cd /Users/murai-labs/Github/Orion && \
//   xcrun clang -O2 -fobjc-arc \
//     -framework Foundation -framework IOSurface -ldl -I . \
//     core/ane_runtime.m core/iosurface_tensor.m \
//     experiments/spike_e5_binary.m -o experiments/spike_e5_binary
//
// Run:
//   ./experiments/spike_e5_binary

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <stdio.h>
#import <math.h>
#import <string.h>
#import <sys/stat.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import "core/ane_runtime.h"
#import "core/iosurface_tensor.h"

#pragma mark - BLOBFILE helpers

// Create a BLOBFILE (128-byte header + fp16 data) from fp32 array.
// Matches the exact format from tests/test_mil_builder.m make_blob_const().
static NSData* make_blobfile(const float *data, int count) {
    size_t fp16_size = count * sizeof(uint16_t);
    size_t total = 128 + fp16_size;
    uint8_t *b = (uint8_t *)calloc(total, 1);

    // Header (matches proven format from test_mil_builder.m)
    b[0] = 1; b[4] = 2;
    b[64] = 0xEF; b[65] = 0xBE; b[66] = 0xAD; b[67] = 0xDE; b[68] = 1;
    *(uint32_t *)(b + 72) = (uint32_t)fp16_size;
    *(uint32_t *)(b + 80) = 128;

    // Fill fp16 data
    _Float16 *fp16 = (_Float16 *)(b + 128);
    for (int i = 0; i < count; i++) {
        fp16[i] = (_Float16)data[i];
    }

    return [NSData dataWithBytesNoCopy:b length:total freeWhenDone:YES];
}

#pragma mark - MIL text generation

// Generate a simple linear (conv) MIL program with one weight matrix.
// Y = conv(X, W) — W is BLOBFILE-referenced weight.
// Uses fp16 I/O matching Orion's actual inference pattern.
// Matches the exact orion_mil_linear() pattern from core/mil_builder.m.
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

#pragma mark - File analysis helpers

// List all files recursively in a directory
static void list_files(NSString *dir, int depth) {
    NSFileManager *fm = [NSFileManager defaultManager];
    NSArray *items = [fm contentsOfDirectoryAtPath:dir error:nil];
    for (NSString *item in items) {
        NSString *full = [dir stringByAppendingPathComponent:item];
        BOOL isDir = NO;
        [fm fileExistsAtPath:full isDirectory:&isDir];
        struct stat st;
        stat(full.UTF8String, &st);
        for (int i = 0; i < depth; i++) printf("  ");
        printf("%s %s (%lld bytes)\n",
               isDir ? "[DIR]" : "     ",
               item.UTF8String,
               (long long)st.st_size);
        if (isDir) list_files(full, depth + 1);
    }
}

// Find all files matching extension in directory (recursive)
static NSArray<NSString*>* find_files(NSString *dir, NSString *ext) {
    NSMutableArray *results = [NSMutableArray array];
    NSFileManager *fm = [NSFileManager defaultManager];
    NSDirectoryEnumerator *e = [fm enumeratorAtPath:dir];
    NSString *path;
    while ((path = [e nextObject])) {
        if ([path.pathExtension isEqualToString:ext]) {
            [results addObject:[dir stringByAppendingPathComponent:path]];
        }
    }
    return results;
}

// Binary diff two files, report regions of difference
typedef struct {
    size_t offset;
    size_t length;
} DiffRegion;

static void binary_diff(const uint8_t *a, const uint8_t *b, size_t len) {
    printf("\n=== Binary diff (%zu bytes total) ===\n", len);

    size_t diff_count = 0;
    size_t diff_bytes = 0;
    int in_diff = 0;
    size_t diff_start = 0;

    // Track diff regions
    DiffRegion regions[1024];
    int nregions = 0;

    for (size_t i = 0; i <= len; i++) {
        int differs = (i < len) && (a[i] != b[i]);
        if (differs && !in_diff) {
            diff_start = i;
            in_diff = 1;
        } else if (!differs && in_diff) {
            size_t rlen = i - diff_start;
            diff_bytes += rlen;
            diff_count++;
            if (nregions < 1024) {
                regions[nregions].offset = diff_start;
                regions[nregions].length = rlen;
                nregions++;
            }
            printf("  DIFF region #%zu: offset 0x%06zx (%zu), length %zu bytes\n",
                   diff_count, diff_start, diff_start, rlen);
            // Show first few bytes of each version
            printf("    A: ");
            for (size_t j = diff_start; j < diff_start + (rlen < 32 ? rlen : 32); j++)
                printf("%02x ", a[j]);
            printf("%s\n", rlen > 32 ? "..." : "");
            printf("    B: ");
            for (size_t j = diff_start; j < diff_start + (rlen < 32 ? rlen : 32); j++)
                printf("%02x ", b[j]);
            printf("%s\n", rlen > 32 ? "..." : "");
            in_diff = 0;
        }
    }

    printf("\nSummary: %zu diff regions, %zu bytes differ out of %zu total (%.1f%%)\n",
           diff_count, diff_bytes, len, 100.0 * diff_bytes / len);

    // Check contiguity
    if (nregions == 1) {
        printf("CONTIGUOUS: All weight differences in a single region!\n");
        printf("  Offset: 0x%06zx (%zu)\n", regions[0].offset, regions[0].offset);
        printf("  Length: %zu bytes\n", regions[0].length);
    } else if (nregions > 1) {
        printf("FRAGMENTED: Weight data split across %d regions\n", nregions);
        // Check if regions are close together (might be padding between)
        for (int i = 1; i < nregions; i++) {
            size_t gap = regions[i].offset - (regions[i-1].offset + regions[i-1].length);
            printf("  Gap between region %d and %d: %zu bytes\n", i-1, i, gap);
        }
    }
}

#pragma mark - Modified compile that preserves temp dir

// Compile MIL and return program + temp directory path (not cleaned up).
// We need access to the temp dir to inspect compiled artifacts.
typedef struct {
    OrionProgram *prog;
    NSString *tmpDir;
} CompileResult;

// We'll use a modified version that lets us access the temp dir.
// Since OrionProgram stores tmpDir internally, we can use a workaround:
// compile normally, then find the temp dir from the hexStringIdentifier.

// Actually, we need direct access to the compiled output. Let's compile
// manually using the private API to keep the temp directory.

static Class g_Desc_local;
static Class g_IMM_local;

static CompileResult compile_and_keep(const char *mil_text, NSDictionary *weight_dict, const char *tag) {
    CompileResult result = {NULL, nil};

    @autoreleasepool {
        NSData *milData = [NSData dataWithBytes:mil_text length:strlen(mil_text)];
        NSDictionary *wdict = weight_dict ?: @{};

        // Use orion's runtime (already initialized)
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            NSClassFromString(@"_ANEInMemoryModelDescriptor"),
            @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) { printf("ERROR: descriptor creation failed\n"); return result; }

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            NSClassFromString(@"_ANEInMemoryModel"),
            @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) { printf("ERROR: model creation failed\n"); return result; }

        // Get temp directory
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];

        // Write weight blobs
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

        // Compile
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            printf("ERROR: compile failed: %s\n", e.description.UTF8String);
            return result;
        }

        // Load
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            printf("ERROR: load failed: %s\n", e.description.UTF8String);
            return result;
        }

        // Wrap as OrionProgram for eval
        result.prog = orion_compile_mil(mil_text, weight_dict, tag);
        result.tmpDir = tmpDir;

        // Also list what's in the temp dir
        printf("\n--- Temp dir for '%s': %s ---\n", tag, tmpDir.UTF8String);
        list_files(tmpDir, 0);

        return result;
    }
}

#pragma mark - Main

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        printf("=== T149: E5 Binary Format Analysis ===\n\n");

        // Initialize ANE
        if (!orion_ane_init()) {
            printf("FATAL: ANE init failed\n");
            return 1;
        }
        printf("ANE initialized.\n");

        // Parameters — use realistic sizes
        // Small enough to be fast, large enough to see weight patterns
        int in_dim = 64;
        int out_dim = 64;
        int seq = 32;
        int weight_count = out_dim * in_dim; // 4096 elements

        printf("Config: in_dim=%d, out_dim=%d, seq=%d, weight_count=%d\n",
               in_dim, out_dim, seq, weight_count);

        // Generate two different weight sets
        float *weights_a = (float *)malloc(weight_count * sizeof(float));
        float *weights_b = (float *)malloc(weight_count * sizeof(float));
        srand(42);
        for (int i = 0; i < weight_count; i++) {
            weights_a[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
            weights_b[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        }
        // Make them clearly different
        printf("Weight A[0..3]: %.6f %.6f %.6f %.6f\n",
               weights_a[0], weights_a[1], weights_a[2], weights_a[3]);
        printf("Weight B[0..3]: %.6f %.6f %.6f %.6f\n",
               weights_b[0], weights_b[1], weights_b[2], weights_b[3]);

        // Create BLOBFILEs
        NSData *blob_a = make_blobfile(weights_a, weight_count);
        NSData *blob_b = make_blobfile(weights_b, weight_count);
        printf("BLOBFILE A: %zu bytes (header 128 + fp16 data %d bytes)\n",
               blob_a.length, weight_count * 2);
        printf("BLOBFILE B: %zu bytes\n", blob_b.length);

        // Create weight dicts
        NSDictionary *wdict_a = @{
            @"@model_path/weights/w.bin": @{
                @"offset": @0,
                @"data": blob_a
            }
        };
        NSDictionary *wdict_b = @{
            @"@model_path/weights/w.bin": @{
                @"offset": @0,
                @"data": blob_b
            }
        };

        // Generate MIL
        NSString *mil = mil_linear(in_dim, out_dim, seq);
        printf("\nMIL program:\n%s\n", mil.UTF8String);

        // Compile with weight set A
        printf("\n========== Compiling with Weight Set A ==========\n");
        OrionProgram *prog_a = orion_compile_mil(mil.UTF8String, wdict_a, "weight_A");
        if (!prog_a) {
            printf("FATAL: Failed to compile with weight set A\n");
            return 1;
        }

        // Compile with weight set B
        printf("\n========== Compiling with Weight Set B ==========\n");
        OrionProgram *prog_b = orion_compile_mil(mil.UTF8String, wdict_b, "weight_B");
        if (!prog_b) {
            printf("FATAL: Failed to compile with weight set B\n");
            return 1;
        }

        // Verify both produce different outputs
        printf("\n========== Verifying outputs differ ==========\n");
        {
            IOSurfaceRef in_surf = orion_tensor_create(in_dim, seq);
            float *input_data = (float *)malloc(in_dim * seq * sizeof(float));
            for (int i = 0; i < in_dim * seq; i++) input_data[i] = 0.01f * (i % 100);
            orion_tensor_write_f32(in_surf, input_data, in_dim * seq);

            IOSurfaceRef out_a = orion_tensor_create(out_dim, seq);
            IOSurfaceRef out_b = orion_tensor_create(out_dim, seq);

            bool ok_a = orion_eval(prog_a, &in_surf, 1, &out_a, 1);
            bool ok_b = orion_eval(prog_b, &in_surf, 1, &out_b, 1);
            printf("Eval A: %s, Eval B: %s\n", ok_a ? "OK" : "FAIL", ok_b ? "OK" : "FAIL");

            float *res_a = (float *)malloc(out_dim * seq * sizeof(float));
            float *res_b = (float *)malloc(out_dim * seq * sizeof(float));
            orion_tensor_read_f32(out_a, res_a, out_dim * seq);
            orion_tensor_read_f32(out_b, res_b, out_dim * seq);

            float max_diff = 0;
            for (int i = 0; i < out_dim * seq; i++) {
                float d = fabsf(res_a[i] - res_b[i]);
                if (d > max_diff) max_diff = d;
            }
            printf("Max output difference: %.6f (should be > 0 if weights differ)\n", max_diff);
            printf("Output A[0..3]: %.6f %.6f %.6f %.6f\n", res_a[0], res_a[1], res_a[2], res_a[3]);
            printf("Output B[0..3]: %.6f %.6f %.6f %.6f\n", res_b[0], res_b[1], res_b[2], res_b[3]);

            free(input_data); free(res_a); free(res_b);
            CFRelease(in_surf); CFRelease(out_a); CFRelease(out_b);
        }

        // Now find and compare the compiled artifacts.
        // The ANE compiler writes to NSTemporaryDirectory()/<hexId>/
        // Look for .e5, .hwx, or other binary artifacts
        printf("\n========== Scanning temp directories for compiled artifacts ==========\n");

        NSString *tmpBase = NSTemporaryDirectory();
        NSFileManager *fm = [NSFileManager defaultManager];
        NSArray *tmpContents = [fm contentsOfDirectoryAtPath:tmpBase error:nil];

        // Find recent directories that look like ANE compile output
        NSMutableArray *aneDirs = [NSMutableArray array];
        for (NSString *item in tmpContents) {
            NSString *full = [tmpBase stringByAppendingPathComponent:item];
            BOOL isDir;
            if ([fm fileExistsAtPath:full isDirectory:&isDir] && isDir) {
                // Check if it contains model.mil (our compile output)
                NSString *milFile = [full stringByAppendingPathComponent:@"model.mil"];
                if ([fm fileExistsAtPath:milFile]) {
                    [aneDirs addObject:full];
                    printf("\nANE compile dir: %s\n", full.UTF8String);
                    list_files(full, 1);
                }
            }
        }

        // Also check for compiled artifacts in a subdirectory
        // ANE typically puts compiled output in <hexId>/model/ or <hexId>/
        printf("\n========== Looking for binary artifacts ==========\n");

        // Search for all binary files across all ANE temp dirs
        NSMutableArray<NSString*> *artifacts_a = [NSMutableArray array];
        NSMutableArray<NSString*> *artifacts_b = [NSMutableArray array];

        for (NSString *dir in aneDirs) {
            NSDirectoryEnumerator *enumerator = [fm enumeratorAtPath:dir];
            NSString *path;
            while ((path = [enumerator nextObject])) {
                NSString *fullPath = [dir stringByAppendingPathComponent:path];
                BOOL isDir;
                [fm fileExistsAtPath:fullPath isDirectory:&isDir];
                if (!isDir) {
                    NSDictionary *attrs = [fm attributesOfItemAtPath:fullPath error:nil];
                    unsigned long long size = [attrs fileSize];
                    NSString *ext = path.pathExtension;

                    // Skip known non-binary files
                    if ([ext isEqualToString:@"mil"] || [ext isEqualToString:@"bin"]) continue;

                    printf("  %s: %s (%llu bytes)\n", dir.lastPathComponent.UTF8String,
                           path.UTF8String, size);
                }
            }
        }

        // Find matching artifact pairs for diff
        // Try to identify E5 or similar compiled binary files
        printf("\n========== Attempting binary diff of compiled artifacts ==========\n");

        if (aneDirs.count >= 2) {
            // For each directory pair, find files with same name
            NSString *dir_first = aneDirs[aneDirs.count - 2]; // Second-to-last = A
            NSString *dir_last = aneDirs[aneDirs.count - 1];  // Last = B

            printf("Comparing:\n  A: %s\n  B: %s\n", dir_first.UTF8String, dir_last.UTF8String);

            // Find all non-mil/non-bin files in first dir
            NSDirectoryEnumerator *e1 = [fm enumeratorAtPath:dir_first];
            NSString *relPath;
            while ((relPath = [e1 nextObject])) {
                NSString *fullA = [dir_first stringByAppendingPathComponent:relPath];
                NSString *fullB = [dir_last stringByAppendingPathComponent:relPath];
                BOOL isDir;
                [fm fileExistsAtPath:fullA isDirectory:&isDir];
                if (isDir) continue;

                // Skip source files
                if ([relPath.pathExtension isEqualToString:@"mil"]) continue;
                if ([relPath.pathExtension isEqualToString:@"bin"]) continue;

                if ([fm fileExistsAtPath:fullB]) {
                    NSData *dataA = [NSData dataWithContentsOfFile:fullA];
                    NSData *dataB = [NSData dataWithContentsOfFile:fullB];

                    printf("\n--- Diffing: %s ---\n", relPath.UTF8String);
                    printf("  A size: %zu bytes, B size: %zu bytes\n",
                           (size_t)dataA.length, (size_t)dataB.length);

                    if (dataA.length == dataB.length) {
                        if ([dataA isEqualToData:dataB]) {
                            printf("  IDENTICAL (no differences)\n");
                        } else {
                            binary_diff(
                                (const uint8_t *)dataA.bytes,
                                (const uint8_t *)dataB.bytes,
                                dataA.length);
                        }
                    } else {
                        printf("  DIFFERENT SIZES — cannot diff directly\n");
                        printf("  Size difference: %zd bytes\n",
                               (ssize_t)dataA.length - (ssize_t)dataB.length);
                    }
                }
            }
        } else {
            printf("WARNING: Found %zu ANE dirs, expected >= 2\n", (size_t)aneDirs.count);
            printf("The ANE runtime may clean up temp dirs after load.\n");
            printf("Trying alternative approach: inspect OrionProgram internals...\n");
        }

        // Additional analysis: check if we can also diff via directly saving
        // the compiled artifact before it gets loaded
        printf("\n========== Weight data analysis ==========\n");
        printf("BLOBFILE A fp16 data (first 16 bytes): ");
        const uint8_t *ba = (const uint8_t *)blob_a.bytes;
        for (int i = 128; i < 128 + 16 && i < (int)blob_a.length; i++)
            printf("%02x ", ba[i]);
        printf("\n");

        printf("BLOBFILE B fp16 data (first 16 bytes): ");
        const uint8_t *bb = (const uint8_t *)blob_b.bytes;
        for (int i = 128; i < 128 + 16 && i < (int)blob_b.length; i++)
            printf("%02x ", bb[i]);
        printf("\n");

        // Test 2: Larger weight set (more realistic)
        printf("\n========== Test 2: Larger weights (768x768, Stories-scale) ==========\n");
        int large_in = 768;
        int large_out = 768;
        int large_count = large_in * large_out; // 589,824 elements
        printf("Weight count: %d (%.1f MB in fp16)\n", large_count,
               large_count * 2.0 / 1024 / 1024);

        float *large_a = (float *)malloc(large_count * sizeof(float));
        float *large_b = (float *)malloc(large_count * sizeof(float));
        for (int i = 0; i < large_count; i++) {
            large_a[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
            large_b[i] = large_a[i]; // Start identical
        }
        // Perturb only a few weights (simulating Adam step)
        for (int i = 0; i < 100; i++) {
            large_b[i] += 0.001f;
        }

        NSData *lblob_a = make_blobfile(large_a, large_count);
        NSData *lblob_b = make_blobfile(large_b, large_count);

        NSString *large_mil = mil_linear(large_in, large_out, 32);
        NSDictionary *lwdict_a = @{
            @"@model_path/weights/w.bin": @{@"offset": @0, @"data": lblob_a}
        };
        NSDictionary *lwdict_b = @{
            @"@model_path/weights/w.bin": @{@"offset": @0, @"data": lblob_b}
        };

        OrionProgram *lprog_a = orion_compile_mil(large_mil.UTF8String, lwdict_a, "large_A");
        OrionProgram *lprog_b = orion_compile_mil(large_mil.UTF8String, lwdict_b, "large_B");

        if (lprog_a && lprog_b) {
            printf("Both large programs compiled successfully.\n");

            // Scan temp dirs again for large programs
            tmpContents = [fm contentsOfDirectoryAtPath:tmpBase error:nil];
            NSMutableArray *largeDirs = [NSMutableArray array];
            for (NSString *item in tmpContents) {
                NSString *full = [tmpBase stringByAppendingPathComponent:item];
                BOOL isDir;
                if ([fm fileExistsAtPath:full isDirectory:&isDir] && isDir) {
                    NSString *milFile = [full stringByAppendingPathComponent:@"model.mil"];
                    if ([fm fileExistsAtPath:milFile]) {
                        // Check if this is one of our large compilations
                        // (it will have a different hexId from the small ones)
                        if (![aneDirs containsObject:full]) {
                            [largeDirs addObject:full];
                        }
                    }
                }
            }

            printf("Found %zu new ANE compile dirs for large test\n", (size_t)largeDirs.count);
            for (NSString *dir in largeDirs) {
                printf("\nDir: %s\n", dir.UTF8String);
                list_files(dir, 1);
            }

            if (largeDirs.count >= 2) {
                NSString *ld_a = largeDirs[0];
                NSString *ld_b = largeDirs[1];

                NSDirectoryEnumerator *e2 = [fm enumeratorAtPath:ld_a];
                NSString *rp;
                while ((rp = [e2 nextObject])) {
                    NSString *fA = [ld_a stringByAppendingPathComponent:rp];
                    NSString *fB = [ld_b stringByAppendingPathComponent:rp];
                    BOOL isDir;
                    [fm fileExistsAtPath:fA isDirectory:&isDir];
                    if (isDir) continue;
                    if ([rp.pathExtension isEqualToString:@"mil"]) continue;
                    if ([rp.pathExtension isEqualToString:@"bin"]) continue;

                    if ([fm fileExistsAtPath:fB]) {
                        NSData *dA = [NSData dataWithContentsOfFile:fA];
                        NSData *dB = [NSData dataWithContentsOfFile:fB];
                        printf("\n--- Large diff: %s ---\n", rp.UTF8String);
                        printf("  A: %zu bytes, B: %zu bytes\n",
                               (size_t)dA.length, (size_t)dB.length);

                        if (dA.length == dB.length && ![dA isEqualToData:dB]) {
                            binary_diff(dA.bytes, dB.bytes, dA.length);
                        } else if ([dA isEqualToData:dB]) {
                            printf("  IDENTICAL\n");
                        } else {
                            printf("  DIFFERENT SIZES\n");
                        }
                    }
                }
            }
        }

        // Cleanup
        printf("\n========== Cleanup ==========\n");
        orion_release_program(prog_a);
        orion_release_program(prog_b);
        if (lprog_a) orion_release_program(lprog_a);
        if (lprog_b) orion_release_program(lprog_b);
        free(weights_a); free(weights_b);
        free(large_a); free(large_b);

        printf("\n=== T149 Spike Complete ===\n");
        printf("Compile count: %d (of ~119 budget)\n", orion_compile_count());

        return 0;
    }
}
