# Orion — Checkpoint (Cross-Session Handoff)

> **Purpose**: This is the handoff document between sessions.
> Whichever tool picks up work next MUST read this file first.
> Updated at the end of every work session, and before every commit.

---

## Last Updated By
- **Tool**: Claude Code
- **Date**: 2026-03-03
- **Session**: 1 (continued)

## Current State
- **Phase**: M1 — CPU Baseline Inference (16/35 tasks complete)
- **Last completed**: T027 — Stories110M weight converter (110 files, 208.9 MB fp16)
- **Next task**: T028 (GPT-2 BPE tokenizer), T031 (GPT-2 weight loading), T032+ (CPU forward pass)
- **Branch**: `main`
- **Repo is green**: YES (test_ane_runtime: 11/11 pass)
- **Known issues**: ANE minimum tensor size — [1,4,1,4] fails, need [1,256,1,64]+
- **Tests passing**: test_ane_runtime 11/11, test_mil_builder 12/12, test_weight_convert 8/8

## What Just Happened (Session 1 continued — M1 Core Runtime)

### T027: Stories110M Weight Converter
1. Parses Karpathy's llama2.c binary format (7×int32 header + flat fp32 weights)
2. Produces 110 blob files: embed + 12 layers × 9 weights + rms_final (208.9 MB fp16)
3. Config verified: dim=768, hidden=2048, heads=12, seq=1024, vocab=32000

---

### T024-T026: BLOBFILE Writer + GPT-2 Weight Converter
1. **T024**: `make_blob_header()` + `convert_tensor_to_blob()` — 128-byte header + fp16 data
2. **T025**: Full GPT-2 converter: 196 files across 12 layers + embeddings + final LN (237.4 MB fp16)
   - GPT-2 uses Conv1D (weight is [in, out]) — transpose needed for ANE conv [out, in, 1, 1]
   - Fused QKV weight split into separate Q, K, V blobs
3. **T026**: Weight conversion tests: 8/8 pass (header format, round-trip, transpose, file size)
4. Venv created at `.venv/` with numpy, torch, transformers

---

### T019-T023: MIL Builder Helpers — ALL PASS
1. **T019**: `orion_mil_linear` — 1×1 conv with BLOBFILE weight refs (3× faster than matmul on ANE)
2. **T020**: `orion_mil_layernorm` + `orion_mil_rmsnorm` — both compile and eval correctly on ANE
3. **T021**: `orion_mil_gelu` (decomposed tanh approx, `gelu` MIL op not available) + `orion_mil_silu`
4. **T022**: `orion_mil_causal_attention` — explicit Q@K^T → mask → softmax → @V (no SDPA)
5. **T023**: `orion_mil_program` — header + function wrapper, `orion_mil_header()` for reuse
6. **Test**: `tests/test_mil_builder.m` — 12/12 pass (wrapper, SiLU, GELU, RMSNorm on ANE)

### Key Findings
- BLOBFILE offset in MIL must be `uint64(64)` (chunk header offset), not `uint64(128)`
- `gelu` is NOT a valid MIL op — must decompose to `tanh` approximation manually
- Weight blob offset in weight dict is 0 (start of blob), not 64

---

### T012-T018: Core Runtime Foundation — ALL PASS
1. **T012**: `orion_tensor_create` — creates IOSurface with ANE layout [1,C,1,S] fp16
2. **T013**: `orion_tensor_write/read` — raw fp16 lock/memcpy/unlock
3. **T014**: `orion_tensor_write_f32/read_f32` — NEON-accelerated fp16↔fp32 (max error 0.001875)
4. **T015**: `orion_compile_mil` — full pipeline: descriptor → temp dir → compile → load (returns OrionProgram*)
5. **T016**: `orion_eval` — wraps IOSurfaces as _ANEIOSurfaceObject, builds _ANERequest, evaluates
6. **T017**: `orion_release_program` — unloads from ANE, cleans temp dir, releases retained refs
7. **T018**: Integration test (`tests/test_ane_runtime.m`): 11/11 pass
   - Init (idempotent), tensor round-trip, add eval, scale eval, 10× compile+release loop

### API Design Decisions
- `OrionProgram` uses `void*` with `CFBridgingRetain/Release` for manual ARC in C struct
- Weight dict passed as `NSDictionary*` (matches upstream pattern directly)
- `orion_ane_init()` must be called once before any compile/eval
- `orion_compile_count()` tracks approach to ~119 limit

---

### T008: Hello MIL Proof-of-Concept — PASS
1. Wrote `experiments/hello_mil.m` — standalone program that compiles `z = add(x, y)` on ANE
2. First attempt with [1,4,1,4] tensors failed (ANE rejects small tensors at eval time)
3. Increased to [1,256,1,64] — **PASS**: all 16,384 elements correct
4. Key pattern: fp32 IOSurface I/O with cast to/from fp16 inside MIL
5. Must pass `@{}` (empty dict) for weights parameter, not `nil`
6. Compile: 17.1ms, Eval: 0.223ms

### T009: ANE API Reference
1. Wrote `docs/ane_api_reference.md` — complete API calling sequence
2. Covers: framework loading, class resolution, 9-step compile→eval pipeline
3. MIL text format, tensor layout, BLOBFILE format, all key operations
4. Constraints & gotchas section with verified findings

### M0 Upstream Validation (T001-T007, T010, T011)
1. **T001 + T002**: Cloned maderix/ANE and ANEgpt into `vendor/maderix-ane/` and `vendor/anegpt/`
2. **T003**: Built `train_large` from maderix/ANE — compiles clean with `xcrun clang`
3. **T004**: Ran `train_large` for 15 steps on M4 Max:
   - Loss: 10.39 → 10.12 (decreasing as expected)
   - 75.3 ms/step, 1.24 TFLOPS sustained, 7.8% ANE utilization
   - Compile time: 83.3% of wall time
   - exec() restart at step 10 works seamlessly
4. **T005**: Built ANEgpt binaries (`train_large`, `train_large_ane`) — 9 warnings (cosmetic)
5. **T006**: Built `libane_bridge.dylib` via ANEgpt Python bridge Makefile
6. **T007**: Ran ANEgpt `train_large` — identical loss trajectory to maderix/ANE
7. **T010**: Downloaded TinyStories pretokenized data (41.3 MB, 20.6M tokens, uint16 Llama2 BPE 32K vocab)
8. **T011**: Downloaded Stories110M weights (418 MB) to `model/weights/stories110M.bin`

### Key Performance Observations
- ANE compile dominates wall time (83%) — validates need for program cache (M4)
- exec() restart overhead is negligible (~50ms) — safe for 119-compile workaround
- Loss trajectory matches between maderix/ANE and ANEgpt — codebases are functionally equivalent for training
- 7.8% ANE utilization suggests significant headroom if compile overhead reduced

---

## What Happened Earlier (Session 1 — Init)

### Repo Initialization
1. Initialized git repo at `/Users/murai-labs/github/Orion`
2. Created full directory structure matching spec §7 (48 files across 11 directories)
3. Created project `CLAUDE.md` with tech stack, conventions, build commands, do's/don'ts

### Spec Rewrite
1. Researched upstream repos:
   - **maderix/ANE** (github.com/maderix/ANE) — Core ANE private API reverse-engineering, MIT
   - **ANEgpt** (github.com/vipuldivyanshu92/ANEgpt) — Full LLM training harness, MIT
2. Updated `ORION_v2_ANE_LLM_SPEC.md` with:
   - Full reference table (5 repos + Apple coremltools docs)
   - Verified private ANE API surface (6 ObjC classes, workflow, gotchas)
   - Hardware facts (16 cores, 32MB SRAM, 15.8 TFLOPS, dispatch overhead)
   - Complete dependency list (system frameworks, Python packages, build commands)
   - BLOBFILE weight blob format (128-byte header spec)
   - 6 kernel types per layer (from ANEgpt) with CPU/ANE division table
   - ANEgpt offload benchmarks (classifier 10.2x, softmax 33.8x)
   - Checkpoint format (CkptHdr struct)
   - Risk table with 8 entries and concrete mitigations
   - Removed all `citeturn` placeholder artifacts

### Task List
1. Generated `TASKS.md` with 94 atomic tasks across 7 phases
2. Each task has: ID, size (S/M/L/XL), acceptance criteria, dependencies, file references
3. Critical path identified: T001 → T008 → T015 → T019 → T047 → T052 → T054

### Key Findings from Research
- **ANEgpt is NOT a fork of maderix/ANE** — it's an independent repo that incorporates `ane-training/` (credited MIT to maderix) and adds a Python bridge + nanochat training harness
- **Weights are truly baked**: Overwriting blob files and reloading does NOT change outputs — recompilation is mandatory (verified on M4 and M5)
- **~119 compile limit per process**: ANEgpt works around this via exec() restart with checkpoint/resume
- **SDPA causal masks are ignored by ANE hardware**: Must decompose Q@K^T → CPU mask+softmax → scores@V
- **ANE is a convolution engine**: 1x1 convolutions deliver 3x better throughput than matmul
- **MIL weight blob format**: 128-byte header with specific magic bytes at offsets 0, 4, 64-68, 68, 72-76, 80-84
- **`_ANEInMemoryModelDescriptor.milText` must be `NSData*`** (UTF-8 bytes), NOT `NSString*`

## Files Created This Session

### Project Infrastructure
- `CLAUDE.md` — Project-level instructions
- `ORION_v2_ANE_LLM_SPEC.md` — Updated spec (was already present, rewritten)
- `README.md` — Project overview
- `LICENSE` — MIT (Murai Labs)
- `.gitignore` — Build, blobs, Python, OS, IDE
- `TASKS.md` — 94-task atomic task list
- `STATUS.md` — Project status dashboard
- `CHECKPOINT.md` — This file

### Skeleton Source Files (all stubs with TODO markers)
- `core/` — ane_runtime, ane_program_cache, mil_builder, iosurface_tensor, profiler (.h + .m)
- `model/configs/` — gpt2_124m.h, stories110m.h
- `model/convert/` — hf_to_blobs_gpt2.py, hf_to_blobs_llama.py
- `model/weights/download.md`, `model/blobs/.gitkeep`
- `kernels/inference/` — gpt2_prefill_attn.milgen.h, gpt2_prefill_ffn.milgen.h, kv_cache, decode_cpu
- `kernels/training/` — stories_train_kernels.milgen.h, rmsnorm_bwd.milgen.h, classifier_softmax.milgen.h, stories_cpu_ops
- `tokenizer/` — gpt2_bpe, sentencepiece_wrap
- `apps/cli/` — main.m, commands/ (infer.m, train.m, bench.m)
- `apps/macapp/` — OrionApp.swift, ModelRunner.swift
- `scripts/download_data.sh`
- `tests/` — test_tokenizer.m, test_weight_convert.py, test_infer_golden.json, test_train_smoke.m, test_program_swap.m

## What To Pick Up Next

### Immediate — Continue M1
**MIL Builder Helpers** (T019-T023):
1. **T019** (M): `orion_mil_linear` — generate MIL for Y = conv(W, X) with BLOBFILE weights
2. **T020** (M): `orion_mil_layernorm` + `orion_mil_rmsnorm`
3. **T021** (S): `orion_mil_gelu` + `orion_mil_silu`
4. **T022** (L): `orion_mil_causal_attention` — decomposed (no SDPA)
5. **T023** (M): `orion_mil_program` — wrapper to compose ops

**Weight Format** (T024-T027, parallel):
1. **T024** (M): BLOBFILE writer (128-byte header + fp16 data)
2. **T025** (M): GPT-2 HF→BLOBFILE converter
3. **T027** (M): Stories110M converter

**Tokenizer** (T028-T030, parallel):
1. **T028** (L): GPT-2 BPE tokenizer in Obj-C

## Staged But Uncommitted Changes
None — all changes committed.

## Commits This Session
1. `71119b0` — Initial repo scaffold with full project structure and updated spec
2. `1e9f330` — Add atomic task list with 94 tasks across 7 phases
3. `d9fea4a` — M0 upstream validation: 9/11 tasks complete
4. `2002a50` — M0 complete: hello MIL + API reference (T008, T009)
5. `01c3199` — M1 core runtime: IOSurface tensors + ANE compile/eval/release (T012-T018)
6. `d2b4924` — M1 MIL builder helpers: linear, norms, activations, attention (T019-T023)
7. `(pending)` — M1 weight format: BLOBFILE writer + GPT-2 converter (T024-T026)

## Warnings for Next Session
- `vendor/` is gitignored — upstream repos must be cloned locally (`git clone` into `vendor/`)
- `data/` is gitignored — TinyStories data must be downloaded locally via `scripts/download_data.sh`
- Weight files are gitignored — `model/weights/stories110M.bin` (418MB) must be downloaded locally
- No build system yet — `xcrun clang` single-file compilation (match upstream)
- Weight blobs are gitignored (`model/blobs/*`) — must run converters locally
- Private ANE APIs require macOS 15+ on Apple Silicon — cannot be tested in CI
- SIP may need to be considered for dlopen of private frameworks
- Python deps (torch, transformers) needed for weight conversion — install in venv
- ANE compile time dominates (83% of wall time) — plan for this in benchmarks
