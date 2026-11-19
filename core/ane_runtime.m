#import "ane_runtime.h"
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>

// T015: orion_compile_mil — compile MIL text to ANE program
// T016: orion_eval — evaluate a compiled program
// T017: orion_release_program — unload and release

#pragma mark - Private ANE Class References

static Class g_Desc;  // _ANEInMemoryModelDescriptor
static Class g_IMM;   // _ANEInMemoryModel
static Class g_AR;    // _ANERequest
static Class g_AIO;   // _ANEIOSurfaceObject
static bool  g_init;
static int   g_compile_count;

#pragma mark - OrionProgram

struct OrionProgram {
    void *model;       // _ANEInMemoryModel (retained via CFBridgingRetain)
    void *tmpDir;      // NSString* tmp directory (retained)
    char tag[256];
    bool loaded;
};


#pragma mark - BLOBFILE merge (macOS 27 workaround)

// macOS 27's ANECompilerService rejects program bundles containing more than
// one BLOBFILE weight file. Merge them into one Core ML storage-v2 style file
// and rewrite every BLOBFILE(path, offset) reference in the MIL text.
//
// Layout: 64-byte file header, then for each tensor a 64-byte metadata block
// followed immediately by its 64-byte-aligned payload. The MIL offset points
// at that tensor's metadata block.

static void orion_put_u32le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

static NSDictionary* orion_merge_weight_dict(NSString* mil,
                                             NSDictionary* wdict,
                                             NSString** out_mil) {
    *out_mil = mil;
    if (wdict.count < 2) return wdict;

    NSArray<NSString*> *keys = [[wdict allKeys] sortedArrayUsingSelector:@selector(compare:)];
    NSMutableData *merged = [NSMutableData data];
    uint8_t fileHeader[64] = {0};

    // Reuse the first blob's 64-byte file header for correct magic/version.
    for (NSString *key in keys) {
        NSData *blob = wdict[key][@"data"];
        if ([blob isKindOfClass:[NSData class]] && blob.length >= 128) {
            [blob getBytes:fileHeader range:NSMakeRange(0, 64)];
            break;
        }
    }
    // Core ML storage v2 file header: entry count at +0, version 2 at +4.
    orion_put_u32le(fileHeader + 0, (uint32_t)keys.count);
    orion_put_u32le(fileHeader + 4, 2);
    [merged appendBytes:fileHeader length:sizeof(fileHeader)];

    NSMutableDictionary<NSString*, NSNumber*> *offsets = [NSMutableDictionary dictionary];
    for (NSString *key in keys) {
        NSData *blob = wdict[key][@"data"];
        if (![blob isKindOfClass:[NSData class]] || blob.length < 128) {
            // Not a BLOBFILE; leave the dictionary untouched and let the
            // normal path report the error.
            *out_mil = mil;
            return wdict;
        }

        uint8_t meta[64];
        [blob getBytes:meta range:NSMakeRange(64, 64)];
        // Sanity check the per-chunk magic 0xDEADBEEF at +64.
        if (!(meta[0] == 0xEF && meta[1] == 0xBE && meta[2] == 0xAD && meta[3] == 0xDE)) {
            *out_mil = mil;
            return wdict;
        }

        NSUInteger chunkOffset = merged.length;
        uint32_t dataSize = (uint32_t)(blob.length - 128);
        // Per-chunk metadata: uint64 size at +8, uint64 data offset at +16.
        orion_put_u32le(meta + 8, dataSize);
        orion_put_u32le(meta + 12, 0);
        orion_put_u32le(meta + 16, (uint32_t)(chunkOffset + 64));
        orion_put_u32le(meta + 20, 0);

        [merged appendBytes:meta length:sizeof(meta)];
        [merged appendData:[blob subdataWithRange:NSMakeRange(128, dataSize)]];
        while ((merged.length & 63) != 0) {
            uint8_t zero = 0;
            [merged appendBytes:&zero length:1];
        }
        offsets[key] = @(chunkOffset);
    }

    NSString *mergedPath = @"@model_path/weights/merged.bin";
    NSMutableString *newMil = [mil mutableCopy];
    NSError *regexErr = nil;
    NSRegularExpression *re = [NSRegularExpression
        regularExpressionWithPattern:@"BLOBFILE\\(path=string\\(\"([^\"]+)\"\\), offset=uint64\\(([0-9]+)\\)\\)"
                             options:0 error:&regexErr];
    if (!re) {
        *out_mil = mil;
        return wdict;
    }

    NSArray<NSTextCheckingResult*> *matches =
        [re matchesInString:newMil options:0 range:NSMakeRange(0, newMil.length)];
    for (NSTextCheckingResult *m in [matches reverseObjectEnumerator]) {
        NSString *oldPath = [newMil substringWithRange:[m rangeAtIndex:1]];
        NSNumber *off = offsets[oldPath];
        if (!off) continue;
        NSString *replacement = [NSString stringWithFormat:
            @"BLOBFILE(path=string(\"%@\"), offset=uint64(%lu))",
            mergedPath, (unsigned long)off.unsignedLongValue];
        [newMil replaceCharactersInRange:m.range withString:replacement];
    }

    *out_mil = newMil;
    return @{mergedPath: @{@"offset": @0, @"data": merged}};
}

#pragma mark - Init

bool orion_ane_init(void) {
    if (g_init) return true;

    void *h = dlopen("/System/Library/PrivateFrameworks/"
                     "AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    if (!h) return false;

    g_Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_IMM  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR   = NSClassFromString(@"_ANERequest");
    g_AIO  = NSClassFromString(@"_ANEIOSurfaceObject");

    if (!g_Desc || !g_IMM || !g_AR || !g_AIO) return false;

    g_init = true;
    g_compile_count = 0;
    return true;
}

#pragma mark - T015: Compile

static OrionProgram* orion_compile_mil_internal(
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag,
    bool merge_blobs
) {
    if (!g_init || !mil_text) return NULL;

    @autoreleasepool {
        NSString *milString = @(mil_text);
        NSDictionary *wdict = weight_dict ?: @{};
        NSString *effectiveMil = milString;
        if (merge_blobs) {
            wdict = orion_merge_weight_dict(milString, wdict, &effectiveMil);
        }
        NSData *milData = [effectiveMil dataUsingEncoding:NSUTF8StringEncoding];

        // Step 1: Create descriptor
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) return NULL;

        // Step 2: Create in-memory model
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) return NULL;

        // Step 3: Pre-populate temp directory (ANE compiler reads from filesystem)
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];

        // Write weight blobs to temp directory
        for (NSString *path in wdict) {
            NSDictionary *entry = wdict[path];
            NSData *data = entry[@"data"];
            if (data) {
                // path is like "@model_path/weights/weight.bin"
                NSString *relPath = [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                                                    withString:@""];
                NSString *fullPath = [tmpDir stringByAppendingPathComponent:relPath];
                NSString *dir = [fullPath stringByDeletingLastPathComponent];
                [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
                [data writeToFile:fullPath atomically:YES];
            }
        }

        // Step 4: Compile
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            if (e) NSLog(@"ANE compile error: %@", e);
            [fm removeItemAtPath:tmpDir error:nil];
            return NULL;
        }

        // Step 5: Load into ANE
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            [fm removeItemAtPath:tmpDir error:nil];
            return NULL;
        }

        g_compile_count++;

        // Wrap in OrionProgram (manual retain for C struct)
        OrionProgram *prog = (OrionProgram *)calloc(1, sizeof(OrionProgram));
        prog->model = (void *)CFBridgingRetain(model);
        prog->tmpDir = (void *)CFBridgingRetain(tmpDir);
        prog->loaded = true;
        if (program_tag) {
            strlcpy(prog->tag, program_tag, sizeof(prog->tag));
        }

        return prog;
    }
}

OrionProgram* orion_compile_mil(
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag
) {
    return orion_compile_mil_internal(mil_text, weight_dict, program_tag, false);
}

OrionProgram* orion_compile_mil_merged(
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag
) {
    return orion_compile_mil_internal(mil_text, weight_dict, program_tag, true);
}

#pragma mark - T016: Eval

bool orion_eval(
    OrionProgram* prog,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
) {
    if (!prog || !prog->loaded || !inputs || !outputs) return false;
    if (num_inputs <= 0 || num_outputs <= 0) return false;

    @autoreleasepool {
        // Wrap inputs as _ANEIOSurfaceObject
        NSMutableArray *inArr = [NSMutableArray arrayWithCapacity:num_inputs];
        NSMutableArray *inIdx = [NSMutableArray arrayWithCapacity:num_inputs];
        for (int i = 0; i < num_inputs; i++) {
            id wrapped = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_AIO, @selector(objectWithIOSurface:), inputs[i]);
            if (!wrapped) return false;
            [inArr addObject:wrapped];
            [inIdx addObject:@(i)];
        }

        // Wrap outputs
        NSMutableArray *outArr = [NSMutableArray arrayWithCapacity:num_outputs];
        NSMutableArray *outIdx = [NSMutableArray arrayWithCapacity:num_outputs];
        for (int i = 0; i < num_outputs; i++) {
            id wrapped = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_AIO, @selector(objectWithIOSurface:), outputs[i]);
            if (!wrapped) return false;
            [outArr addObject:wrapped];
            [outIdx addObject:@(i)];
        }

        // Build request
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:
                      weightsBuffer:perfStats:procedureIndex:),
            inArr, inIdx, outArr, outIdx, nil, nil, @0);
        if (!req) return false;

        // Evaluate
        id model = (__bridge id)prog->model;
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &e);

        if (!ok && e) {
            NSLog(@"orion_eval ERROR [%s]: %@", prog->tag, e);
        }
        return ok;
    }
}

#pragma mark - T017: Release

void orion_release_program(OrionProgram* prog) {
    if (!prog) return;

    @autoreleasepool {
        // Unload from ANE
        if (prog->loaded && prog->model) {
            id model = (__bridge id)prog->model;
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &e);
        }

        // Clean up temp directory
        if (prog->tmpDir) {
            NSString *td = (__bridge id)prog->tmpDir;
            [[NSFileManager defaultManager] removeItemAtPath:td error:nil];
            CFRelease(prog->tmpDir);
            prog->tmpDir = NULL;
        }

        // Release model ref
        if (prog->model) {
            CFRelease(prog->model);
            prog->model = NULL;
        }

        free(prog);
    }
}

int orion_compile_count(void) {
    return g_compile_count;
}

#pragma mark - T151: Delta weight patching

OrionProgram* orion_program_patch_weights(
    OrionProgram* donor,
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag
) {
    if (!g_init || !donor || !donor->tmpDir || !mil_text) return NULL;

    @autoreleasepool {
        NSData *milData = [NSData dataWithBytes:mil_text length:strlen(mil_text)];
        NSDictionary *wdict = weight_dict ?: @{};
        NSFileManager *fm = [NSFileManager defaultManager];

        // Create descriptor + model with new weights
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) return NULL;

        id model = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) return NULL;

        // Get the new model's expected temp directory
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *newDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSString *donorDir = (__bridge NSString *)donor->tmpDir;
        bool sameDir = [newDir isEqualToString:donorDir];

        if (sameDir) {
            // Same hexId as donor (same MIL text + weight dict keys).
            // Directory and net.plist already exist — just update weight files.
            // Steal tmpDir ownership from donor so release won't delete the shared dir.
            CFRelease(donor->tmpDir);
            donor->tmpDir = NULL;
        } else {
            // Different hexId — set up new directory with donor's net.plist
            [fm removeItemAtPath:newDir error:nil];
            [fm createDirectoryAtPath:newDir withIntermediateDirectories:YES attributes:nil error:nil];

            NSString *srcPlist = [donorDir stringByAppendingPathComponent:@"net.plist"];
            NSString *dstPlist = [newDir stringByAppendingPathComponent:@"net.plist"];
            NSError *copyErr = nil;
            if (![fm copyItemAtPath:srcPlist toPath:dstPlist error:&copyErr]) {
                NSLog(@"patch_weights[%s]: net.plist copy failed: %@ -> %@ err=%@",
                      program_tag ?: "?", srcPlist, dstPlist, copyErr);
                [fm removeItemAtPath:newDir error:nil];
                return NULL;
            }
        }

        // Write new BLOBFILE(s) as the "data" file.
        // The compiled ANE "data" file IS the BLOBFILE — no transformation.
        // For single-weight programs, the data file = the BLOBFILE.
        // For multi-weight programs, we need to find which blob maps to "data".
        //
        // Strategy: write the BLOBFILE as the data file. If there are multiple
        // weight blobs, write the primary one (first entry) as data and others
        // to their weight paths. Also write model.mil for completeness.

        [milData writeToFile:[newDir stringByAppendingPathComponent:@"model.mil"]
                  atomically:YES];

        // Write weight blobs to weights/ and also as the data file
        [fm createDirectoryAtPath:[newDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];

        bool dataWritten = false;
        for (NSString *path in wdict) {
            NSDictionary *entry = wdict[path];
            NSData *data = entry[@"data"];
            if (data) {
                // Write to weights/ directory
                NSString *relPath = [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                                                    withString:@""];
                NSString *fullPath = [newDir stringByAppendingPathComponent:relPath];
                NSString *dir = [fullPath stringByDeletingLastPathComponent];
                [fm createDirectoryAtPath:dir withIntermediateDirectories:YES attributes:nil error:nil];
                [data writeToFile:fullPath atomically:YES];

                // First blob also becomes the data file
                if (!dataWritten) {
                    [data writeToFile:[newDir stringByAppendingPathComponent:@"data"]
                           atomically:YES];
                    dataWritten = true;
                }
            }
        }

        // If no weights, copy the donor's data file
        if (!dataWritten) {
            NSString *srcData = [donorDir stringByAppendingPathComponent:@"data"];
            [fm copyItemAtPath:srcData
                        toPath:[newDir stringByAppendingPathComponent:@"data"]
                         error:nil];
        }

        // Load WITHOUT compiling — the key delta patching optimization
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            if (e) NSLog(@"orion_program_patch_weights LOAD FAILED: %@", e);
            [fm removeItemAtPath:newDir error:nil];
            return NULL;
        }

        // Wrap in OrionProgram
        OrionProgram *prog = (OrionProgram *)calloc(1, sizeof(OrionProgram));
        prog->model = (void *)CFBridgingRetain(model);
        prog->tmpDir = (void *)CFBridgingRetain(newDir);
        prog->loaded = true;
        if (program_tag) {
            strlcpy(prog->tag, program_tag, sizeof(prog->tag));
        }

        // Note: does NOT increment g_compile_count — no compile happened
        return prog;
    }
}

bool orion_program_reload_weights(
    OrionProgram* prog,
    NSDictionary* weight_dict
) {
    if (!g_init || !prog || !prog->model || !prog->tmpDir) return false;

    @autoreleasepool {
        id model = (__bridge id)prog->model;
        NSString *dir = (__bridge NSString *)prog->tmpDir;
        NSDictionary *wdict = weight_dict ?: @{};

        // Unload from ANE
        if (prog->loaded) {
            NSError *e = nil;
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                model, @selector(unloadWithQoS:error:), 21, &e);
            prog->loaded = false;
        }

        // Update weight files on disk
        for (NSString *path in wdict) {
            NSDictionary *entry = wdict[path];
            NSData *data = entry[@"data"];
            if (data) {
                NSString *relPath = [path stringByReplacingOccurrencesOfString:@"@model_path/"
                                                                    withString:@""];
                NSString *fullPath = [dir stringByAppendingPathComponent:relPath];
                [data writeToFile:fullPath atomically:NO];
            }
        }

        // Reload with new weights
        NSError *e = nil;
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) {
            if (e) NSLog(@"orion_program_reload_weights LOAD FAILED: %@", e);
            return false;
        }
        prog->loaded = true;
        return true;
    }
}
