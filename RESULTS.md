# Orion — Results

Comprehensive benchmark results and technical findings from Orion on Mac Studio M4 Max 64GB.

---

## Inference (GPT-2 124M, M4 Max)

| Metric | Value |
|--------|-------|
| CPU decode throughput | **283 tok/s** |
| CPU decode latency (p50) | 3.5 ms/token |
| CPU decode latency (p90) | 3.7 ms/token |
| ANE full forward throughput | **170+ tok/s** |
| ANE decode latency | 5.78 ms/token |
| ANE prefill (first call) | 1399 ms (83% compile time) |
| ANE prefill (cached) | sub-100 ms |
| Accuracy | 100% top-1 argmax, exact 5-token greedy match vs CPU baseline |

Three inference modes: CPU-only, ANE prefill + CPU decode (hybrid), and ANE full forward. CPU decode is fastest per-token due to ANE dispatch overhead; ANE full forward demonstrates the hardware capability.

---

## Training (Stories110M, TinyStories vocab=32000)

### v2.0 — Delta Compilation (current)

| Metric | Value |
|--------|-------|
| Avg train time | **849 ms/step** |
| Avg recompile time | **494 ms/step** |
| Avg total step time | **1,345 ms/step** |
| Throughput | 0.656 TFLOPS |
| Recompile % of step | 36.8% |
| Gradient accumulation | 4 microbatches |
| exec() restarts needed | **0** (delta reload bypasses compile limit) |

**How it works**: ANE programs are compiled once at startup (72 programs, ~4.5s). Each subsequent training step updates weights via `orion_program_reload_weights` — unload the existing ANE program, update BLOBFILE on disk, reload. This bypasses `ANECCompile()` entirely, avoiding both the compilation cost and the ~119 compile limit.

#### v2.0 — 1000-Step Training Run (lr=3e-4, delta compilation)

| Metric | Value |
|--------|-------|
| Total steps | 1,000 |
| Initial loss | 11.83 |
| Wall time | **22.4 minutes** |
| NaN occurrences | **0 / 1,000** |
| Memory leak | None (stable RSS across all steps) |
| Checkpoints saved | 10 (every 100 steps) |

Training runs in a single process — no `exec()` restarts, no process boundaries. Consistent ~1.35s/step throughout.

### v2.0 vs v1.0 Comparison

| Metric | v1.0 (Full Recompile) | v2.0 (Delta Reload) | Improvement |
|--------|----------------------|---------------------|-------------|
| Avg train time | 908 ms/step | 849 ms/step | ~same |
| Avg recompile time | 4,200 ms/step | 494 ms/step | **8.5x** |
| Avg total step time | 5,108 ms/step | 1,345 ms/step | **3.8x** |
| Recompile % of step | 83.9% | 36.8% | -47.1 pp |
| 1000-step wall time | ~85 min | 22.4 min | **3.8x** |
| Process model | 1 step/process (exec restart) | Single process (no restart) | Eliminated |
| Compile limit risk | ~119 budget, must restart | N/A (0 compiles during training) | Eliminated |

The 8.5x recompile speedup comes from bypassing `ANECCompile()`. Instead of creating new model descriptors and compiling MIL text (4.2s for 60 weight-bearing kernels), delta reload does unload→file update→reload on existing model objects (~494ms for 60 kernels, or ~8ms/kernel).

---

### v1.0 — Full Recompile (baseline, for reference)

| Metric | Value |
|--------|-------|
| Step time (mean ± std) | **908 ± 16 ms** (compute only) |
| Recompile time | ~4,200 ms/step |
| Throughput | 0.612 TFLOPS |
| Compile budget | 72 programs/process → 1 step/process → auto-restart via `exec()` |

#### v1.0 — 1000-Step Training Run (lr=3e-4, full recompile)

| Metric | Value |
|--------|-------|
| Total steps | 1,000 |
| Initial loss | 12.29 |
| Minimum loss | 6.19 (step 888) |
| Final loss | 6.55 |
| NaN occurrences | **0 / 1,000** |
| Wall time | ~85 minutes |

Loss dropped 50% (12.3→6.2) with zero NaN across all 1,000 steps. Each step ran in a separate process via `exec()` restart due to the ~119 compile budget.

### Stability Stress Test (5 chains × 5 steps, lr=1e-5)

| Chain | Step 1 | Step 2 | Step 3 | Step 4 | Step 5 |
|-------|--------|--------|--------|--------|--------|
| 1 | 13.979 | 13.968 | 13.946 | 13.932 | 13.923 |
| 2 | 13.978 | 13.965 | 13.949 | 13.927 | 13.916 |
| 3 | 13.975 | 13.962 | 13.945 | 13.924 | 13.911 |
| 4 | 13.974 | 13.961 | 13.941 | 13.926 | 13.911 |
| 5 | 13.971 | 13.958 | 13.935 | 13.920 | 13.904 |

- **0 NaN** across all 25 steps (5 chains × 5 steps)
- All 5 chains monotonically decreasing
- Cross-chain loss std: 0.003 (step 1) → 0.007 (step 5)
- Each step runs in a fresh process (compile budget management via `exec()`)
- 25/25 `exec()` restarts successful, checkpoint resume verified

---

## ANE Kernel Performance

| Metric | Value |
|--------|-------|
| Single-token dispatch latency | ~0.03 ms |
| Weight swap compile time | 11.3 ms |
| Weight swap eval time | 0.15 ms |
| 100-iter swap endurance RSS growth | 1.41x (pass, threshold 2.0x) |

Kernel dispatch overhead is minimal — the ANE can service single-token requests well under the 5ms threshold, making full-forward decode viable.

---

## Compiler

| Metric | Value |
|--------|-------|
| Operations | 27 ops |
| Optimization passes | 5 (DCE, identity elimination, cast fusion, SRAM annotation, uniform output padding) |
| Compiler frontends | 13 (4 GPT-2 inference + 6 Stories training + 3 utility) |
| Structural equivalence | Verified against hand-written MIL for all 13 kernel types |

The compiler translates graph IR to ANE-compatible MIL text. All hand-written `milgen` generators have been replaced by compiler frontends. The MIL diff tool verifies structural equivalence between generated and reference programs.

---

## Numerical Stability (Training NaN Fix)

Three bugs caused NaN/Inf cascades during training. All fixed and verified.

### Bug 1: Stale ANE Programs on Resume

**Problem**: After checkpoint resume, ANE programs were compiled with stale (pre-resume) weights. The forward pass used old weights while the backward pass expected gradients from the new weights, causing divergence.

**Fix**: Deferred ANE compilation — programs are now compiled *after* checkpoint weights are loaded, not before. Each process compiles exactly once with the correct weights.

### Bug 2: fp16 Overflow Cascade

**Problem**: Large activations in fp16 (ANE native format) overflowed to Inf during the forward pass. These Infs propagated through softmax and cross-entropy, producing NaN losses.

**Fix**: fp16 clamping — activations are clamped to `[-65504, 65504]` (fp16 max) before operations that can amplify magnitude (softmax, layer norm). Applied in `stories_cpu_ops.m`.

### Bug 3: Corrupted BLOBFILE Weights

**Problem**: The BLOBFILE writer could produce corrupted weight data when the checkpoint's weight tensor layout didn't match the expected MIL weight dict format, causing silent numerical corruption.

**Fix**: Gradient sanitization — weight updates are sanitized (NaN→0, Inf→clamp) before writing to BLOBFILE format. Added validation in `stories_train.m` to detect corrupted weights early.

### Before/After

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| NaN rate | 100% (step 2+) | 0% (verified 5 steps) |
| Loss trajectory | 13.98 → NaN | 13.98 → 13.92 (monotonic) |
| Resume stability | Broken | Verified (5-step chain) |

---

## Hardware: Apple Neural Engine (M4 Max)

Hardware characteristics based on measurements by [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615), confirmed independently in Orion benchmarks.

| Property | Value | Source |
|----------|-------|--------|
| Generation | H16 | Apple spec |
| Peak throughput (Apple spec) | 38 TOPS (INT8) | Apple spec |
| Peak throughput (actual fp16) | ~19 TFLOPS | [maderix Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) |
| On-chip SRAM | 32 MB (30% perf drop when exceeded) | [maderix Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) |
| Dispatch overhead (XPC+IOKit) | ~0.095 ms | [maderix Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) |
| Data format | fp16 `[1, C, 1, S]` on IOSurface | maderix/hollance |
| Compile limit | ~119 programs per process | [maderix Part 2](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615) |

> Note: Apple claims 38 TOPS (INT8), but the ANE dequantizes INT8 to fp16 before computation — INT8 saves memory bandwidth, not compute. Actual measured peak is ~19 TFLOPS fp16.

### Key Constraints

Hardware-level constraints (compile limit, weight baking, conv vs matmul) were first documented by [maderix](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615). MIL IR and memory constraints were discovered during Orion development.

- **Compile budget**: ~119 compilations per process before the ANE stops accepting programs. Managed via program cache and `exec()` restart. *(maderix)*
- **Tensor layout**: All I/O must be fp16 `[1, C, 1, S]` on IOSurface-backed memory. CPU↔ANE transfers require transpose.
- **Minimum allocation**: ~49KB IOSurface minimum. `seq_len=1` compiles but fails at eval (status `0x1d`). Minimum decode bucket is `seq=16`. *(Orion)*
- **No concat**: ANE rejects the `concat` MIL op. Use multi-output programs with uniform buffer sizes instead. *(Orion)*
- **No causal masks in SDPA**: ANE ignores causal masks — attention must be manually decomposed (Q@K^T → mask → softmax → @V). *(maderix/ANEgpt, confirmed by Orion)*
- **Alphabetical output ordering**: Multi-output surfaces are ordered by MIL variable name, not return tuple order. *(Orion)*
