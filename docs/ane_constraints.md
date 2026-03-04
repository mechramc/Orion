# ANE Constraints Reference

> Practical reference for anyone working with Apple Neural Engine private APIs.
> Every constraint below was discovered empirically during Orion development on M4 Max.

---

## 1. `concat` MIL Op Rejected by ANE Compiler

**What happens:** `ANECCompile() FAILED: err=()` for any program using `concat(axis=1, values=(...))`.

**Workaround:** Use multi-output MIL programs (`orion_mil_program_multi`) that return separate IOSurfaces per output instead of concatenating tensors into one.

**Discovered:** T064-T071 (ANE Training Kernels). All 7 training kernels using concat failed; `qkvBwd` (single summed output, no concat) was the only one that compiled.

---

## 2. Multi-Output Requires Uniform Output Buffer Sizes

**What happens:** `ANEProgramProcessRequestDirect() Failed with status=0x1d : Program Inference error` at eval time when output IOSurfaces have different allocation sizes.

**Workaround:** Pad all multi-output IOSurfaces to the max channel size across all outputs for that kernel. For example, if a kernel outputs `[d,s]` and `[h,s]` where `h > d`, allocate all output surfaces at `[h,s]`.

**Discovered:** T072 (Training Step). fwdFFN had mixed-size outputs `[d,s]` and `[h,s]`. Same fix applied to ffnBwd and sdpaBwd1 outputs.

---

## 3. Multi-Output Surfaces Ordered Alphabetically by MIL Variable Name

**What happens:** Output surfaces arrive in alphabetical order by their MIL variable name, NOT by their position in the MIL return tuple. Reading outputs in tuple order produces silently wrong data.

**Example:** MIL returns `(q32, k32, v32)` but actual output order is `k32, q32, v32`. You must provide output surfaces as `{ioK, ioQ, ioV}`.

**Workaround:** Always provide output IOSurfaces in alphabetical order of their MIL variable names. Or name your variables so alphabetical order matches your intended order.

**Discovered:** T101 (ANE Decode Step). Prefill's multi-output `(attn_res32, k32, v32)` had worked by coincidence -- it was already alphabetical.

---

## 4. Minimum IOSurface Allocation ~49KB for Eval

**What happens:** `ANEProgramProcessRequestDirect() Failed with status=0x1d` at eval time. Programs compile fine but fail at eval when IOSurface allocations are too small. seq=1 tensors with 768 channels = 3072 bytes -- too small.

**Minimum working size:** `[768, 16]` = 49,152 bytes (48KB). ANE internally uses a stride of 16 for the seq dimension in padded surfaces.

**Workaround:** Use `ORION_DECODE_SEQ = 16` as the minimum decode bucket. Place token data at seq position 0 and zero-pad positions 1-15.

**Discovered:** T100 (ANE Decode MIL). Invalidated T099 spike results which had reported seq=1 working (environment may have changed between sessions).

---

## 5. ~119 Compile Limit Per Process

**What happens:** After approximately 119 calls to `ANECCompile()`, the compiler silently fails or the process becomes unstable. The ANE compiler leaks internal state that cannot be reclaimed.

**Workaround:** Track compile count with `orion_compile_count()`. When approaching the budget, checkpoint state to disk and call `exec()` to restart the process. exec() overhead is negligible (~50ms).

**Discovered:** T004 (Upstream Validation). ANEgpt uses the same exec() restart strategy. Confirmed: 83% of wall time is compile, so a program cache (T084) is essential.

---

## 6. SDPA Causal Masks Ignored by ANE

**What happens:** If you pass a causal mask to ANE's SDPA operation, it compiles and runs but the mask has no effect. Attention scores are computed as if no mask exists, producing incorrect outputs for autoregressive generation.

**Workaround:** Decompose attention manually:
```
Q @ K^T  -->  apply causal mask (additive -inf)  -->  softmax  -->  @ V
```
Implemented via `orion_mil_causal_attention` using explicit matmul + mask + softmax + matmul.

**Discovered:** T022 (MIL Causal Attention). Also confirmed in upstream maderix/ANE and ANEgpt codebases.

---

## 7. Weights Baked at Compile Time

**What happens:** Overwriting BLOBFILE weight files on disk and reloading does NOT change model outputs. The weights are embedded into the compiled ANE program binary at compile time. There is no way to update weights without recompiling.

**Workaround:** Recompilation is mandatory for weight updates. For training: compile with new weights, cache the program, evict the old one. The program cache (T084) manages this lifecycle.

**Discovered:** T001-T007 (Upstream Validation). Verified on M4 and M5 hardware.

---

## 8. BLOBFILE Offset is `uint64(64)`, Not `uint64(128)`

**What happens:** MIL weight references with the wrong offset cause silent garbage reads. The compiled program reads weight data from the wrong position in the blob, producing incorrect outputs with no error.

**Details:** The BLOBFILE format has a 128-byte header, but the MIL `const()` weight reference offset points to the chunk header at byte 64, not the start of the file (byte 0) or end of the full header (byte 128).

```
MIL: const(name="w", val=blob(file="weight.blob", offset=uint64(64)))
```

The weight blob offset in the weight dict is 0 (start of blob data), not 64.

**Discovered:** T019-T023 (MIL Builder Helpers). Also confirmed in T099 (Single-Token Spike).

---

## 9. `milText` Must Be `NSData*`, Not `NSString*`

**What happens:** Passing an `NSString*` to `_ANEInMemoryModelDescriptor.milText` causes a crash or silent failure. The API expects raw UTF-8 bytes.

**Workaround:**
```objc
NSString *milString = [self generateMIL];
NSData *milData = [milString dataUsingEncoding:NSUTF8StringEncoding];
descriptor.milText = milData;
```

Always convert your MIL text string to `NSData*` with UTF-8 encoding before passing to the descriptor.

**Discovered:** T008 (Hello MIL Proof-of-Concept) and documented in T009 (ANE API Reference).

---

## 10. `gelu` Is Not a Valid MIL Op

**What happens:** Using `gelu(x)` in MIL text causes `ANECCompile() FAILED`. The `gelu` activation is not in the ANE compiler's supported op set despite appearing in MIL documentation.

**Workaround:** Decompose to the tanh approximation manually:

```
gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

Implemented as `orion_mil_gelu` using `tanh`, `mul`, `add`, and `pow` MIL ops which are all supported.

**Discovered:** T021 (MIL GELU + SiLU). `silu` also requires decomposition but was implemented alongside.

---

## 11. Weight Dict Must Be `@{}`, Not `nil`, for Weight-Free Programs

**What happens:** Passing `nil` as the weight dictionary to `ANEProgramProcessRequestDirect()` causes a crash, even for programs that have no weights (e.g., softmax, activation-only kernels).

**Workaround:**
```objc
NSDictionary *weights = @{};  // empty dict, NOT nil
ANEProgramProcessRequestDirect(program, request, weights);
```

Always pass an empty `NSDictionary` for weight-free programs.

**Discovered:** T008 (Hello MIL Proof-of-Concept). The `z = add(x, y)` proof-of-concept has no weights but still requires `@{}`.

---

## Quick Reference Table

| # | Constraint | Severity | Symptom |
|---|-----------|----------|---------|
| 1 | No `concat` op | Compile fail | `ANECCompile() FAILED` |
| 2 | Uniform output buffer sizes | Eval fail | `status=0x1d` |
| 3 | Alphabetical output ordering | Silent wrong data | Outputs swapped |
| 4 | Minimum ~49KB IOSurface | Eval fail | `status=0x1d` |
| 5 | ~119 compile limit | Process instability | Silent fail / crash |
| 6 | SDPA masks ignored | Silent wrong data | Unmasked attention |
| 7 | Weights baked at compile | Silent stale data | Old weights used |
| 8 | BLOBFILE offset is 64 | Silent wrong data | Garbage weights |
| 9 | milText must be NSData* | Crash | Immediate crash |
| 10 | No `gelu` MIL op | Compile fail | `ANECCompile() FAILED` |
| 11 | Weight dict must be `@{}` | Crash | Immediate crash |
