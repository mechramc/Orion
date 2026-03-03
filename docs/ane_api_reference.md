# ANE Private API Reference

> **T009**: API calling sequence documented from upstream maderix/ANE and ANEgpt.
> All classes loaded via `NSClassFromString()` after `dlopen()` of the private framework.

## Framework Loading

```objc
dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
```

Returns `NULL` if framework not available (non-Apple-Silicon, SIP issues).

## Class Resolution

```objc
Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
Class IMM  = NSClassFromString(@"_ANEInMemoryModel");
Class AR   = NSClassFromString(@"_ANERequest");
Class AIO  = NSClassFromString(@"_ANEIOSurfaceObject");
```

All four must be non-nil for the ANE pipeline to work.

## API Calling Sequence

### 1. Create Descriptor

```objc
id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
    Desc, @selector(modelWithMILText:weights:optionsPlist:),
    milData,    // NSData* — MIL text as UTF-8 bytes (NOT NSString*)
    weightsDict, // NSDictionary* — weight blob mapping (use @{} if no weights)
    nil);        // options (unused, pass nil)
```

**Weight dictionary format:**
```objc
@{
    @"@model_path/weights/weight.bin": @{
        @"offset": @(0),      // byte offset into blob
        @"data": weightBlob   // NSData* containing the full blob
    }
}
```

**Important**: The weights parameter must be a dictionary, not `nil`. Pass `@{}` for weight-free programs.

### 2. Create In-Memory Model

```objc
id model = ((id(*)(Class,SEL,id))objc_msgSend)(
    IMM, @selector(inMemoryModelWithDescriptor:), desc);
```

### 3. Pre-populate Temp Directory

The ANE compiler reads MIL text and weight files from a temp directory derived from the model's hex identifier. This directory **must** be pre-created before compilation.

```objc
id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
NSFileManager *fm = [NSFileManager defaultManager];

[fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
    withIntermediateDirectories:YES attributes:nil error:nil];
[milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
// Write each weight blob file too, if any:
[weightBlob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
```

### 4. Compile

```objc
NSError *e = nil;
BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
    model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
```

- **QoS 21** = high priority (always use this)
- Returns `YES` on success
- **WARNING**: Each process can compile ~119 programs before the ANE stops accepting new compilations. Use `exec()` restart to reset.

### 5. Load

```objc
ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
    model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
```

Loads the compiled E5 binary into the ANE hardware.

### 6. Create IOSurface Tensors

```objc
size_t bytes = channels * spatial * sizeof(float); // fp32: 4 bytes/elem
IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)@{
    (id)kIOSurfaceWidth: @(bytes),
    (id)kIOSurfaceHeight: @1,
    (id)kIOSurfaceBytesPerElement: @1,
    (id)kIOSurfaceBytesPerRow: @(bytes),
    (id)kIOSurfaceAllocSize: @(bytes),
    (id)kIOSurfacePixelFormat: @0
});
```

**Writing data:**
```objc
IOSurfaceLock(surface, 0, NULL);
memcpy(IOSurfaceGetBaseAddress(surface), data, bytes);
IOSurfaceUnlock(surface, 0, NULL);
```

**Reading data:**
```objc
IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
memcpy(output, IOSurfaceGetBaseAddress(surface), bytes);
IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
```

### 7. Build Request

```objc
id wIn  = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);

id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
    AR,
    @selector(requestWithInputs:inputIndices:outputs:outputIndices:
              weightsBuffer:perfStats:procedureIndex:),
    @[wIn1, wIn2],  // NSArray of _ANEIOSurfaceObject inputs
    @[@0, @1],       // NSArray of NSNumber input indices (match MIL func arg order)
    @[wOut],         // NSArray of _ANEIOSurfaceObject outputs
    @[@0],           // NSArray of NSNumber output indices
    nil,             // weightsBuffer (nil for in-memory models)
    nil,             // perfStats (nil unless profiling)
    @0);             // procedureIndex (always 0 for single-function programs)
```

### 8. Evaluate

```objc
ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
    model, @selector(evaluateWithQoS:options:request:error:),
    21, @{}, req, &e);
```

On success, output data is available in the output IOSurface(s).

### 9. Unload & Cleanup

```objc
((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
    model, @selector(unloadWithQoS:error:), 21, &e);
CFRelease(ioIn);
CFRelease(ioOut);
[[NSFileManager defaultManager] removeItemAtPath:tmpDir error:nil];
```

## MIL Text Format

MIL (Model Intermediate Language) is Apple's SSA IR for neural network programs.

### Structure

```
program(1.3)
[buildInfo = dict<string, string>({
    {"coremlc-component-MIL", "3510.2.1"},
    {"coremlc-version", "3505.4.1"},
    {"coremltools-component-milinternal", ""},
    {"coremltools-version", "9.0"}
})]
{
    func main<ios18>(tensor<dtype, [shape]> input_name, ...) {
        // operations
        type var = op(args...)[name = string("unique_id")];
    } -> (output_var);
}
```

### Tensor Layout

ANE tensors are always 4D: `[batch, channels, height, spatial]` → `[1, C, 1, S]`

### Supported Types

| MIL Type | Bytes | Notes |
|----------|-------|-------|
| `fp32`   | 4     | IOSurface I/O type |
| `fp16`   | 2     | Internal ANE compute type |
| `int32`  | 4     | Shape constants, indices |
| `bool`   | 1     | Flags |
| `string` | -     | Op parameters |

### Key Operations

| Op | Signature | Notes |
|----|-----------|-------|
| `add` | `add(x=a, y=b)` | Element-wise add |
| `mul` | `mul(x=a, y=b)` | Element-wise multiply |
| `cast` | `cast(dtype=dt, x=a)` | Type conversion |
| `conv` | `conv(weight=W, x=a, ...)` | 1×1 conv (3× faster than matmul on ANE) |
| `matmul` | `matmul(x=a, y=b, ...)` | Matrix multiply |
| `softmax` | `softmax(axis=ax, x=a)` | Softmax (WARNING: ignores causal masks) |
| `reshape` | `reshape(shape=s, x=a)` | Reshape tensor |
| `transpose` | `transpose(perm=p, x=a)` | Permute dimensions |
| `reduce_sum` | `reduce_sum(x=a, axes=ax, keep_dims=kd)` | Sum reduction |
| `pow` | `pow(x=a, y=b)` | Element-wise power |
| `sigmoid` | `sigmoid(x=a)` | Sigmoid activation |
| `concat` | `concat(axis=ax, interleave=b, values=(a,b,...))` | Concatenate tensors |
| `const` | `const()[name=..., val=...]` | Constant declaration |

### Weight Embedding (BLOBFILE)

Weights are baked into MIL as compile-time constants:

```
tensor<fp16, [out, in, 1, 1]> W = const()[
    name = string("W"),
    val = tensor<fp16, [out, in, 1, 1]>(
        BLOBFILE(path = string("@model_path/weights/weight.bin"),
                 offset = uint64(64)))
];
```

## BLOBFILE Format

128-byte header + fp16 weight data:

| Offset | Size | Value | Description |
|--------|------|-------|-------------|
| 0 | 1 | `0x01` | Magic byte 0 |
| 4 | 1 | `0x02` | Magic byte 4 |
| 64 | 4 | `0xDEADBEEF` | Chunk magic (LE: `EF BE AD DE`) |
| 68 | 1 | `0x01` | Version |
| 72 | 4 | varies | Data size in bytes |
| 80 | 4 | `128` | Data offset from file start |
| 128+ | N×2 | fp16 | Weight data (_Float16) |

## Constraints & Gotchas

1. **Minimum tensor size**: Very small tensors (e.g., [1,4,1,4]) fail at evaluation. Use at least [1,256,1,64].
2. **~119 compile limit**: ANE stops accepting compilations after ~119 per process. Workaround: `exec()` restart.
3. **SDPA ignores causal masks**: The `attn_mask` parameter to ANE's SDPA is silently ignored. Must decompose attention manually.
4. **Weights are baked**: Cannot update weights without full recompilation.
5. **Temp directory required**: ANE compiler reads from a filesystem temp directory, not just in-memory data.
6. **fp32 I/O, fp16 compute**: Use `cast()` between fp32 IOSurface data and fp16 internal operations.
7. **1×1 conv > matmul**: Prefer `conv` over `matmul` for linear layers (3× faster on ANE).
8. **Always call unload**: Failing to unload causes ANE resource leaks.
9. **QoS = 21**: Always use QoS value 21 for all compile/load/evaluate calls.

## Verified On

- **Hardware**: Mac Studio M4 Max 64GB
- **OS**: macOS 15 (Sequoia)
- **Date**: 2026-03-03
- **hello_mil.m**: z = x + y on [1,256,1,64] — PASS (compile 17.1ms, eval 0.223ms)
- **train_large**: 15 steps, loss 10.39→10.12, 75.3 ms/step
