# Orion — Engineering Design Review

**Reviewer:** Technical design review against ORION v3 specification
**Date:** 2026-03-03
**Repository:** /Users/murai-labs/Github/Orion (local)
**Spec version:** ORION_v3_ANE_LLM_SPEC.md

---

## 1. Repository Overview

### Current Implementation State

The Orion repository contains a functional on-device LLM runtime for Apple Silicon ANE. Milestones M0–M3 are complete (83/98 tasks). The codebase is approximately 7,500 lines of Objective-C across 26 compilation units, with 141 tests passing across 13 test binaries.

### Module Summary

| Module | Files | LOC | Function |
|--------|-------|-----|----------|
| `core/` | 7 .h, 7 .m | ~1,100 | ANE runtime, IOSurface tensors, MIL builder, profiler, checkpoints, bucketing |
| `tokenizer/` | 2 .h, 2 .m | ~610 | GPT-2 BPE (50,257 vocab), SentencePiece (32,000 vocab) |
| `model/` | 1 .h, 1 .m, 2 configs | ~150 | Weight loading, GPT-2 and Stories110M config structs |
| `kernels/inference/` | 6 .h, 6 .m | ~1,150 | CPU forward, KV cache, ANE prefill runner, 3 MIL generators |
| `kernels/training/` | 5 .h, 5 .m | ~1,700 | Training orchestrator, CPU ops, data loader, 2 MIL generator files |
| `apps/cli/` | 4 .m | ~400 | CLI dispatch, infer, train, bench (bench is stub) |
| `tests/` | 13 .m | ~3,300 | Full test coverage for M0–M3 |

### Build System

No formal build system exists. Compilation uses direct `xcrun clang` invocation with 26 source files, 3 frameworks (Foundation, IOSurface, Accelerate), and `-ldl` for private framework loading. Test binaries are compiled individually. This is consistent with the upstream ANEgpt pattern and spec §3.5.

### Runtime Integration

The ANE runtime (`core/ane_runtime.m`) wraps the private `AppleNeuralEngine.framework` via `dlopen` + `objc_getClass`. It provides:
- `orion_ane_init` — singleton client initialization
- `orion_compile_mil` — MIL text → compiled program (via `_ANEInMemoryModelDescriptor`)
- `orion_eval` — program evaluation with IOSurface I/O
- `orion_release_program` — ARC-safe program release
- `orion_compile_count` — compile budget tracking

All private API interactions are correctly isolated in this module.

---

## 2. Architecture Alignment Table

| Spec Component | Implemented | Status | Notes |
|----------------|-------------|--------|-------|
| ANE runtime wrapper | Yes | **Complete** | `core/ane_runtime.{h,m}` — compile, eval, release, count |
| MIL kernel generators | Yes | **Complete** | 3 inference generators (attn, FFN, final), 2 training generator files (6 kernels + classifier/softmax) |
| MIL builder helpers | Yes | **Complete** | `core/mil_builder.{h,m}` — linear, layernorm, rmsnorm, gelu, silu, attention |
| IOSurface tensor I/O | Yes | **Complete** | `core/iosurface_tensor.{h,m}` — fp16/fp32 read/write, NEON conversions |
| Kernel scheduling system | No | **Missing** | No explicit scheduler; kernels are dispatched sequentially in hardcoded order |
| Training pipeline | Yes | **Complete** | `kernels/training/stories_train.{h,m}` — full orchestrator with GCD async dW |
| Weight swapping system | Partial | **M4 Pending** | Recompile-after-Adam works; no program cache or weights_id abstraction |
| Program cache | No | **Stub** | `core/ane_program_cache.{h,m}` exists but is ~40 LOC stub |
| KV cache | Yes | **Complete** | `kernels/inference/kv_cache.{h,m}` — create, store_prefill, append, free |
| GPT-2 tokenizer | Yes | **Complete** | `tokenizer/gpt2_bpe.{h,m}` — 20 golden tests passing |
| SentencePiece tokenizer | Yes | **Complete** | `tokenizer/sentencepiece_wrap.{h,m}` — 7 tests passing |
| CLI interface | Partial | **Mostly Done** | `infer` and `train` complete; `bench` is stub |
| Benchmark suite | Partial | **Training Only** | Per-step TFLOPS in train command; no `bench kernels`, `bench inference`, or `bench swap` |
| Model abstraction layer (OrionModel) | No | **Not Started** | GPT-2 and Stories110M are hardcoded via config structs, no common interface |
| Kernel abstraction layer (OrionKernel) | No | **Not Started** | Kernels are generator functions, not abstracted objects |
| Runtime abstraction layer (OrionRuntime) | No | **Not Started** | No centralized runtime; logic spread across stories_train.m and infer.m |
| Checkpoint save/load | Yes | **Complete** | `core/checkpoint.{h,m}` — binary format with header validation |
| Data loader | Yes | **Complete** | `kernels/training/data_loader.{h,m}` — mmap'd uint16 binary |
| Profiler | Yes | **Complete** | `core/profiler.{h,m}` — inference metrics; training profiler in train.m |

### Key Mismatches

**1. Inference architecture mismatch (Critical)**

The spec (v3) defines inference as "ANE full forward pass with CPU sampling loop." The implementation uses **ANE prefill + CPU decode**:
- `kernels/inference/prefill_ane.m` compiles per-layer ANE programs for a bucketed prompt
- `kernels/inference/decode_cpu.m` runs the entire forward pass on CPU for each decode step
- `apps/cli/commands/infer.m` calls ANE prefill once, then loops CPU decode

This is the most significant gap. The current architecture does not demonstrate sustained ANE utilization during generation — the ANE is idle during decode.

**2. No abstraction layers**

The spec defines OrionKernel, OrionModel, and OrionRuntime as first-class abstractions. The implementation has none — GPT-2 inference uses one code path (`decode_cpu.m` + `prefill_ane.m`), Stories110M training uses a completely separate code path (`stories_train.m`). There is no shared kernel interface, no model registry, and no centralized runtime.

**3. Benchmark suite incomplete**

The spec requires `orion bench kernels`, `orion bench inference`, `orion bench training`, and `orion bench swap`. Only basic per-step TFLOPS reporting exists in the train command. `apps/cli/commands/bench.m` is a 12-line stub.

---

## 3. Missing Systems

### Core Runtime Gaps

| Missing System | Why Required | Where It Fits | Complexity |
|----------------|--------------|---------------|------------|
| **Program cache** | Avoids recompilation on weight swap; key to M4 | `core/ane_program_cache.{h,m}` | Large |
| **Cache eviction** | Prevents compile budget exhaustion | Inside program cache | Medium |
| **weights_id abstraction** | Keys cache entries; enables swap endurance | `core/ane_program_cache.h` | Medium |
| **OrionKernel abstraction** | Decouples MIL generation from model-specific code | New `core/kernel.h` | Medium |
| **OrionModel abstraction** | Enables multi-model support without code duplication | New `core/model.h` | Medium |
| **OrionRuntime abstraction** | Centralizes scheduling, cache, and loop orchestration | New `core/runtime.h` | Large |

### Inference Pipeline Gaps

| Missing System | Why Required | Where It Fits | Complexity |
|----------------|--------------|---------------|------------|
| **ANE full forward for decode** | Spec requires ANE for all transformer computation, not just prefill | `kernels/inference/` — new decode_ane path | Large |
| **ANE single-token forward** | Decode step must run on ANE, not CPU | MIL generator for seq_len=1 forward | Medium |
| **Inference loop refactor** | Current infer.m hardcodes ANE-prefill + CPU-decode | `apps/cli/commands/infer.m` rewrite | Medium |

### Training Pipeline Gaps

No critical training gaps — M3 is functionally complete. Minor gaps:

| Missing System | Why Required | Where It Fits | Complexity |
|----------------|--------------|---------------|------------|
| **Training profiler detail** | Per-phase breakdown (fwd/bwd/dW/Adam/recompile) | Instrumentation inside stories_train.m | Small |

### Performance Tooling Gaps

| Missing System | Why Required | Where It Fits | Complexity |
|----------------|--------------|---------------|------------|
| **bench kernels** | Isolate per-kernel ANE latency | `apps/cli/commands/bench.m` | Medium |
| **bench inference** | End-to-end inference throughput measurement | `apps/cli/commands/bench.m` | Medium |
| **bench training** | Step time breakdown with controlled runs | `apps/cli/commands/bench.m` | Small |
| **bench swap** | Weight swap endurance with memory tracking | `apps/cli/commands/bench.m` | Medium |
| **SRAM spill detection** | Working set profiling for kernel tuning | New profiling instrumentation | Medium |

### Developer Tooling Gaps

| Missing System | Why Required | Where It Fits | Complexity |
|----------------|--------------|---------------|------------|
| **Build system (Makefile)** | Reproducible builds, test automation | Project root | Medium |
| **SwiftUI demo app** | End-user demo, not just CLI | `apps/macapp/` | Medium |
| **CI pipeline** | Automated test runs (limited by ANE hardware requirement) | GitHub Actions + self-hosted runner | Medium |
| **Auto-tuner (Stage 2)** | MIL graph variant exploration | New `tools/optimize/` | Large |
| **Model converter (Stage 3)** | PyTorch → Orion import | New `tools/convert/` | Large |

---

## 4. Risk Assessment

### Critical Risks

**1. ANE Private API Stability — HIGH RISK**

The entire system depends on 7 undocumented Objective-C classes in `AppleNeuralEngine.framework`. Any macOS update can break the API surface, method signatures, or behavior.

- **Current status:** Validated on macOS 15 / M4 Max
- **Mitigation:** Pin macOS version; maintain a compatibility test (`test_ane_runtime.m`)
- **Impact if it fails:** All ANE work blocked; fallback to CPU only

**2. Compile Budget Exhaustion — HIGH RISK (Validated)**

The ~119 compile limit per process is a hard constraint. The current training loop compiles 6 programs per layer × 12 layers = 72 per recompile step. With `STORIES_MAX_COMPILES=100`, only ~1.3 recompile cycles fit per process.

- **Current mitigation:** `orion_trainer_needs_restart` + checkpoint save before exec() restart
- **Residual risk:** Program cache (M4) must work correctly to avoid unnecessary recompiles during inference
- **Impact if cache fails:** Inference limited to ~1.6 full model loads per process

**3. ANE Full Forward for Decode — MEDIUM-HIGH RISK**

The spec requires ANE to run the full transformer for every decode step (single-token forward). This has not been implemented. Potential issues:
- Single-token (seq_len=1) MIL programs may not compile or may suffer high dispatch overhead relative to compute
- ANE minimum tensor size constraint (`[1,4,1,4]` fails) may prevent single-token execution
- KV cache management on ANE (vs CPU) adds complexity

- **Mitigation:** Test single-token ANE dispatch first as a spike. If latency is dominated by dispatch overhead, consider small-batch decode (4-8 tokens per ANE call).
- **Impact if it fails:** Must fall back to CPU decode (current implementation), which contradicts the spec's "full ANE forward" architecture

**4. Memory Management During Long Runs — MEDIUM RISK**

ANE compilation leaks memory through retained ObjC objects. The program cache must correctly release old programs. The swap endurance test (100 alternations) will validate this.

- **Current status:** ARC handles most cleanup; `orion_release_program` explicitly releases
- **Residual risk:** Cache eviction may not release all associated resources
- **Impact if it fails:** OOM during long training runs or swap tests

### Lower Risks

**5. MIL Graph Complexity — LOW-MEDIUM RISK**

MIL generation is currently done via string concatenation in C/ObjC functions. This is fragile but functional for the current kernel set. Adding new architectures or auto-tuning (Stage 2) will require a more structured approach.

**6. FP16 Numerical Drift — LOW RISK (Validated)**

Golden tests use two-tier tolerance (exact + top-k fallback). Inference and training both produce correct results within tolerance. Max logit error vs PyTorch: ~0.073 across 12 layers.

**7. Performance Bottlenecks — LOW RISK (Understood)**

ANE compile time dominates (~83% of wall time in benchmarks). This is well-understood and addressed by the program cache (M4). Training step time and TFLOPS are within expected ranges.

---

## 5. Prioritized Implementation Plan

### Phase 0 — Environment Validation (COMPLETE)

**Objective:** Confirm ANE private APIs work on target hardware.
**Tasks:** Clone upstream repos, build train_large, run hello MIL.
**Output:** Working ANE compilation and evaluation.
**Done:** M0 complete (11/11 tasks).

### Phase 1 — Minimal ANE Runtime (COMPLETE)

**Objective:** Core runtime primitives for ANE compilation, evaluation, and tensor I/O.
**Tasks:** IOSurface tensors, MIL compile/eval, program release, compile counting.
**Output:** `core/` module with full API surface.
**Done:** All runtime primitives implemented and tested (11/11 tests).

### Phase 2 — CPU Baseline Model (COMPLETE)

**Objective:** Full CPU inference for GPT-2 as correctness reference.
**Tasks:** Weight loading, CPU forward pass, tokenizer, KV cache, decode loop, CLI.
**Output:** `orion infer --model gpt2_124m` generates coherent text.
**Done:** M1 complete (35/35 tasks). Golden tests pass.

### Phase 3 — ANE Inference Integration (COMPLETE, with gap)

**Objective:** ANE prefill for GPT-2 inference.
**Tasks:** Bucket routing, MIL kernel generators, ANE prefill runner, hybrid inference.
**Output:** ANE-accelerated inference with measurable speedup.
**Done:** M2 complete (10/10 tasks). ANE prefill works.

**Gap:** Spec now requires ANE full forward for decode, not just prefill. This is not implemented. See Phase 3.5 below.

### Phase 3.5 — ANE Full Forward Inference (NOT STARTED — spec alignment)

**Engineering objective:** Replace CPU decode loop with ANE-based single-token forward pass.

**Tasks:**
1. Spike: Test single-token ANE dispatch latency (seq_len=1 MIL program)
2. Generate MIL forward kernels for seq_len=1 (or small batch)
3. Implement ANE decode step: ANE forward → CPU sampling → append KV cache
4. Refactor `infer.m` to use ANE forward for all steps (prefill + decode)
5. Verify inference golden tests still pass
6. Benchmark ANE decode vs CPU decode throughput

**Expected output:** `orion infer` uses ANE for both prefill and decode.
**Definition of done:** Inference golden tests pass with ANE-only forward pass. Per-token latency measured.

### Phase 4 — Training Pipeline Integration (COMPLETE)

**Objective:** Full training loop for Stories110M on ANE.
**Tasks:** 6 kernel types, gradient accumulation, checkpoint/resume, recompile, exec() restart, CLI.
**Output:** `orion train` runs training steps with loss decrease.
**Done:** M3 complete (27/27 tasks).

### Phase 5 — Weight Swapping System (NEXT — M4)

**Engineering objective:** Program cache enables weight swapping without process restart.

**Tasks:**
1. Implement program cache keyed by `(weights_id, kernel_type, layer_idx)`
2. Implement LRU cache eviction with correct program release
3. Implement `weights_id` abstraction for training (step-based) and inference (named)
4. Implement `orion bench swap` endurance test
5. Validate 100-swap endurance without memory growth
6. Integrate cache into training loop (avoid recompiling unchanged kernels)

**Expected output:** Weight swapping works for 100+ iterations without process restart.
**Definition of done:** Swap endurance test passes. Memory stable across 100 swaps.

### Phase 6 — Benchmark Harness (M5 partial)

**Engineering objective:** Comprehensive benchmark suite for ANE kernel and pipeline performance.

**Tasks:**
1. Implement `orion bench kernels` — per-kernel ANE latency profiling
2. Implement `orion bench inference` — end-to-end inference throughput
3. Implement `orion bench training` — step time breakdown
4. Implement `orion bench swap` — weight swap endurance with memory tracking
5. Add SRAM spill detection (working set size estimation)
6. Add benchmark regression tracking (baseline comparison)
7. Build system (Makefile) for reproducible builds and test automation

**Expected output:** All bench subcommands produce metrics. README benchmark table populated.
**Definition of done:** `orion bench kernels`, `bench inference`, `bench training`, `bench swap` all produce output. Makefile builds binary and runs tests.

### Phase 7 — Optimization Framework (Stage 2, future)

**Engineering objective:** Auto-tuning system for ANE kernel optimization.

**Tasks:**
1. Define MIL graph variant generation framework
2. Implement variant compilation and profiling
3. Implement variant selection (fastest compile+eval)
4. Add `orion optimize` CLI command
5. Integrate tuned kernels into inference and training pipelines

**Expected output:** Measurable speedup on at least one kernel type.
**Definition of done:** `orion optimize attention` produces a faster kernel than the hand-tuned baseline.

---

## 6. Engineering Quality Improvements

### Module Boundaries

The current codebase has clean file separation but lacks formal module boundaries. Specific improvements:

1. **Header-only public API:** Each module should export a single public header. Internal helpers should be in `_internal.h` files or static functions. Currently `stories_train.h` exposes both public API (`orion_trainer_create`) and internal types (`OrionLayerKernels`, `OrionLayerIO`).

2. **Separate training orchestrator from kernel code:** `stories_train.m` at ~900 lines handles both training orchestration and weight I/O. These should be separated: `stories_train.m` for orchestration, `stories_weight_io.m` for BLOBFILE read/write with transpose.

3. **Model registry:** Instead of `#include "model/configs/stories110m.h"` with hardcoded globals, provide a `orion_model_config_load(name)` function that returns `OrionModelConfig*` from a registry.

### Testing Strategy

Current state is strong (141 tests, golden vectors). Improvements:

1. **Benchmark regression tests:** Record baseline metrics and alert on degradation. Currently no automated regression detection.

2. **Integration tests:** The test suite is heavily unit-focused. Add integration tests that exercise the full `orion infer` and `orion train` CLI paths end-to-end (not just internal APIs).

3. **Memory leak tests:** Add tests that run N iterations of compile/eval/release and verify RSS does not grow. Critical for M4 program cache validation.

4. **Fuzz testing for MIL generation:** MIL text generation via string concatenation is fragile. Fuzz test with edge-case model configs (1 layer, 1 head, minimal dims) to catch generation bugs.

### CI / Build Reproducibility

1. **Makefile (T090):** Required. Should support `make`, `make test`, `make clean`, `make bench`. Currently every build requires remembering 26 .m files and correct flags.

2. **Self-hosted CI runner:** ANE tests cannot run in standard CI (requires Apple Silicon + macOS 15). A self-hosted GitHub Actions runner on the Mac Studio would enable automated testing on push. CPU-only tests could run in standard CI.

3. **Build reproducibility:** Pin clang version via `xcrun --show-sdk-version` in build output. Record macOS version and hardware in benchmark results.

### Benchmark Reproducibility

1. **Warmup runs:** All benchmarks should include configurable warmup iterations (discard first N runs). ANE compilation is expensive; first-run latency is not representative.

2. **Statistical rigor:** Report p50, p90, p99, and stddev, not just averages. The existing profiler does p50/p90 for decode; extend to all benchmarks.

3. **Hardware fingerprint:** Include machine model, macOS version, ANE core count, and memory in benchmark output. Results are not comparable across hardware.

### Documentation Structure

1. **`docs/architecture.md`:** A concise architecture document with module dependency diagram. Currently architecture knowledge is scattered across CHECKPOINT.md, STATUS.md, and the spec.

2. **`docs/ane_constraints.md`:** Consolidate all discovered ANE constraints (concat rejection, uniform buffer sizes, minimum tensor size, compile limit) into one reference document. Currently these are scattered in CHECKPOINT.md session notes.

3. **API documentation:** The header files serve as API docs but lack usage examples. Add a `docs/api_examples.md` showing common patterns (compile a kernel, run inference, save a checkpoint).

### Code Quality

1. **Static analysis:** Run `clang --analyze` on all source files. The codebase uses `-fobjc-arc` but manual memory management for C structs. Static analysis would catch leaks in C allocations (`malloc`/`free` pairs).

2. **Warning level:** Compile with `-Wall -Wextra -Wpedantic`. Currently only `-O2 -fobjc-arc` is used. Additional warnings would catch implicit conversions, unused variables, and missing prototypes.

3. **Consistent error handling:** Some functions return `NULL` on error, others return `bool`, others `fprintf(stderr)` and continue. Standardize on a pattern: return success/failure, log to stderr, clean up partial state.

---

## Summary

Orion is a technically ambitious project with strong M0–M3 execution: 83/98 tasks complete, 141 tests passing, functional ANE training and inference pipelines. The codebase is well-organized with clear module separation.

**Three critical gaps require attention before the project matches the v3 spec:**

1. **Inference architecture:** The spec requires ANE full forward for decode, not just prefill. This is the largest architectural change and carries technical risk (single-token ANE dispatch latency).

2. **Program cache (M4):** The weight swapping system is the next milestone. Without it, the compile budget limits both training (exec restart every ~1 step) and inference (1.6 model loads per process).

3. **Benchmark harness:** The bench command is a stub. Implementing `bench kernels`, `bench inference`, `bench training`, and `bench swap` will validate performance claims and enable optimization.

**Recommended execution order:**
1. Program cache (M4) — unblocks weight swapping and reduces compile overhead
2. Benchmark suite — validates performance before architectural changes
3. ANE full forward inference — requires careful spike to validate feasibility
4. Abstraction layers — can be introduced incrementally without blocking functionality
