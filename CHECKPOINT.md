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
- **Phase**: M0 — Upstream Validation (9/11 tasks complete)
- **Last completed**: T001-T007, T010, T011 — both upstreams cloned, built, validated; data + weights downloaded
- **Next task**: T008 (Hello MIL proof-of-concept), T009 (ANE API reference docs)
- **Branch**: `main`
- **Repo is green**: YES (upstream builds verified on M4 Max)
- **Known issues**: None
- **Tests passing**: N/A (no Orion tests yet — upstream training validated)

## What Just Happened (Session 1 continued — M0 Execution)

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

### Immediate — Finish M0 (2 tasks remaining)
1. **T008** (M): Hello MIL proof-of-concept — compile trivial MIL program, eval on ANE, read result back → `experiments/hello_mil.m`
   - This is the **critical path** gate for all M1+ ANE work
   - Use upstream `ane_runtime.h` patterns from maderix/ANE as reference
2. **T009** (S): Document ANE API calling sequence → `docs/ane_api_reference.md`
   - Can run in parallel with T008

### After M0
- Begin M1 Phase 1 (core runtime): T012-T018 (IOSurface + ANE runtime)
- Parallel: T028 (GPT-2 tokenizer — no dependencies)

## Staged But Uncommitted Changes
None — all changes committed.

## Commits This Session
1. `71119b0` — Initial repo scaffold with full project structure and updated spec
2. `1e9f330` — Add atomic task list with 94 tasks across 7 phases
3. `(pending)` — M0 upstream validation: 9/11 tasks complete

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
