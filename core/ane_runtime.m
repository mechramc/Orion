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

OrionProgram* orion_compile_mil(
    const char* mil_text,
    NSDictionary* weight_dict,
    const char* program_tag
) {
    if (!g_init || !mil_text) return NULL;

    @autoreleasepool {
        NSData *milData = [NSData dataWithBytes:mil_text length:strlen(mil_text)];
        NSDictionary *wdict = weight_dict ?: @{};

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
