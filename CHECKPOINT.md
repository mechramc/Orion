# Orion — Checkpoint (Cross-Session Handoff)

> **Purpose**: This is the handoff document between sessions.
> Whichever tool picks up work next MUST read this file first.
> Updated at the end of every work session, and before every commit.

---

## Last Updated By
- **Tool**: Claude Code
- **Date**: 2026-03-03
- **Session**: 6

## Current State
- **Phase**: M3 — Training (19/27 done)
- **Last completed**: T074 — Gradient accumulation
- **Next task**: T075 (Checkpoint save)
- **Branch**: `main`
- **Repo is green**: YES (all tests pass)
- **Known issues**: ANE compile dominates prefill time (~83%) — program cache (M4) will fix; HuggingFace auth needed for TinyStories data download; ANE rejects `concat` MIL op — use multi-output instead; ANE multi-output requires uniform output buffer sizes
- **Tests passing**: test_ane_runtime 11/11, test_mil_builder 12/12, test_weight_convert 8/8, test_cpu_forward 6/6, test_tokenizer 20/20, test_decode 4/4, test_infer_golden 3/3, test_ane_prefill 34/34, test_cpu_training_ops 19/19, test_sp_tokenizer 7/7, test_data_loader 7/7, test_train_kernels 16/16, test_train_smoke 4/4

## What Just Happened (Session 6 — M3 Training Step)

### T072: Single Training Step (XL)
1. Implemented `stories_train.h` — full trainer data structures: `OrionTrainer`, `OrionLayerKernels`, `OrionLayerWeights`, `OrionLayerGrads`, `OrionLayerAdam`, `OrionLayerIO`
2. Implemented `stories_train.m` (~712 lines) — complete training step:
   - `orion_trainer_create` — compiles all ANE programs (6 per layer), loads weights from BLOBFILE format, allocates all buffers
   - `orion_train_step` — embedding → fwd layers (ANE) → final RMSNorm + classifier (CPU cblas_sgemm) → cross-entropy loss → backward classifier + RMSNorm (CPU) → backward layers (ANE + CPU dW) → embedding backward
   - `orion_trainer_adam_update` — Adam with bias correction for all parameters
   - `orion_trainer_zero_grads` — zero all gradient buffers
   - `orion_trainer_free` — cleanup all resources
3. CPU↔ANE data flow uses `io_read_transpose` (ANE [C,S] → CPU [S,C]) and `io_write_transpose` (CPU [S,C] → ANE [C,S])
4. Backward input assembly uses `orion_tensor_copy_into` for zero-copy fp16 IOSurface region copies

### T073: GCD Async dW Overlap
1. Replaced inline `orion_cpu_dw_accum` calls with `dispatch_group_async` on serial GCD queue
2. Heap-captured buffers (`c_dh1`, `c_xn`, etc.) for block closure safety, freed inside block
3. Deferred `dispatch_group_wait` to end of training step (after embedding backward)
4. Serial queue ensures thread safety on gradient accumulators
5. Verified identical loss values vs synchronous version (5.545182 → 5.544421)

### Key Finding: ANE Multi-Output Requires Uniform Buffer Sizes
- `ANEProgramProcessRequestDirect() Failed with status=0x1d : Program Inference error`
- Root cause: fwdFFN has mixed-size outputs ([d,s] and [h,s]). ANE eval requires ALL output IOSurfaces to have the same allocation size.
- **Fix**: Pad all multi-output IOSurfaces to max channel size per kernel. e.g., fwdFFN outputs all use max(d,h)=h channels.
- Same fix applied to ffnBwd outputs and sdpaBwd1 outputs.

### T082: Training Smoke Test
- `test_train_smoke.m` — 4 tests with 1-layer config (d=768, h=2048, vocab=256, seq=256)
- Creates synthetic weight blobs on disk (BLOBFILE format), runs full training loop
- Verifies: trainer creation, single step finite loss, Adam update, loss decrease over 5 steps
- Loss: 5.545 → 5.544 (monotonic decrease with synthetic data)

### Also in this session (uncommitted from Session 5)
- Changed all training kernel outputs from fp32 to fp16 (zero-copy backward input assembly)
- Added `orion_cpu_rmsnorm_bwd` and `orion_cpu_embedding_bwd` to stories_cpu_ops
- Added `orion_tensor_copy_into` and `orion_tensor_write_f32_at` to iosurface_tensor
- Added error logging to `orion_eval` (logs ANE error domain/code on failure)

---

## What Just Happened (Session 5 — M3 ANE Training Kernels)

### T064-T071: ANE Training Kernels
1. **T064**: `orion_milgen_fwd_attn` — RMSNorm → QKV → causal attention → Wo. 6 multi-output taps (oo, qf, kf, vf, af, xn) for backward pass activation reuse.
2. **T065**: `orion_milgen_fwd_ffn` — RMSNorm → W1, W3 parallel → SiLU(h1) * h3 → W2. 5 multi-output taps (y, h1, h3, gt, xn).
3. **T066**: `orion_milgen_ffn_bwd` — SiLU backward chain rule: sig(h1)*(1+h1*(1-sig(h1))) through W2^T, W1^T, W3^T. 3 outputs (dx, dh1, dh3).
4. **T067**: `orion_milgen_sdpa_bwd1` — Wo^T → reshape to multi-head → recompute attention forward → dV = P^T @ da, dp = da @ V^T. 3 outputs (dvf, pf, dpf).
5. **T068**: `orion_milgen_sdpa_bwd2` — Softmax backward: ds = P * (dp - sum(P*dp)) scaled → dQ = ds @ K, dK = ds^T @ Q. Weight-free. 2 outputs (dqf, dkf).
6. **T069**: `orion_milgen_qkv_bwd` — Wq^T + Wk^T + Wv^T → sum → dx. Single output.
7. **T070**: `orion_milgen_classifier_fwd` — embed^T @ hidden → logits via conv [vocab, dim, 1, 1].
8. **T071**: `orion_milgen_vocab_softmax` — softmax(axis=1) over 32000 vocab classes.

### Key Finding: ANE Rejects concat MIL Op
- Initial implementation used `concat(axis=1, values=(...))` to bundle multiple output tensors into one IOSurface. All kernels using concat failed with `ANECCompile() FAILED: err=()`.
- `qkvBwd` (single summed output, no concat) was the only kernel that compiled.
- **Fix**: Replaced concat with `orion_mil_program_multi()` — returns separate IOSurface per output. All 7 kernels now compile and eval successfully.
- Added `orion_tensor_create_f32` and `orion_tensor_read_f32_direct` for fp32 output surfaces (training kernels cast to fp32 for gradient precision).

### Test Results
- **test_train_kernels**: 15/15 pass — 7 milgen tests, 7 compile tests, 1 eval test (fwdAttn with 6 fp32 outputs verified finite)
- 7 successful ANE compiles in one process

---

## What Just Happened (Session 4 — M3 CPU Training Foundation)

### T057-T061: CPU Training Ops
1. **T057**: `orion_cpu_rmsnorm` — vDSP_dotpr for sum-of-squares, normalize by 1/sqrt(ss/dim + eps), then scale by weight
2. **T058**: `orion_cpu_cross_entropy` — softmax with max-subtraction for numerical stability, NLL loss, gradient = softmax - one_hot (averaged over seq_len)
3. **T059**: `orion_cpu_embedding` — simple memcpy table lookup per token
4. **T060**: `orion_cpu_adam_step` — Adam with bias correction: lr_t = lr * sqrt(1-beta2^t) / (1-beta1^t), then m,v update + param step
5. **T061**: `orion_cpu_dw_accum` — cblas_sgemm with CblasTrans for x^T @ dy, beta=1.0 for accumulation
6. **Test**: 19/19 pass — RMSNorm (4 tests), cross-entropy (5 tests), embedding (2 tests), Adam (4 tests including convergence on quadratic), dW accum (4 tests including 256×768×768 dimensions)

### T062: SentencePiece Tokenizer
7. **T062**: Reads Karpathy's tokenizer.bin format (max_token_len + N*(score+len+chars)). Implements BPE encoding: preprocess text with ▁ prefix and space→▁ replacement, initialize per-character tokens, greedy merge by highest score. Decode handles ▁→space and <0xNN> byte tokens.
8. Downloaded llama2_tokenizer.model from Karpathy's repo, exported to tokenizer_32k.bin (466KB, 32000 tokens)
9. **Test**: 7/7 pass — exact match with Python SentencePiece reference (encode "Once upon a time..." → [9038, 2501, 263, 931, 727, 471, 263, 2217, 7826]), roundtrip verified

### T063: Data Loader
10. **T063**: Memory-mapped data loader for pretokenized uint16 binary files. Produces (input, target) pairs where target = input shifted by 1. Supports reset and wrap-around.
11. Created synthetic test data (1010 tokens, 10 documents) since HuggingFace download requires auth
12. **Test**: 7/7 pass — open, batch production, target shift verification, multiple batches, reset, boundary checks

---

## What Happened Earlier (Session 3 — ANE Prefill Kernels)

### T047, T048, T050: ANE Milgen Kernels + Bucket Selection
1. **T050**: Bucket selection — `orion_select_bucket()` picks smallest bucket ≥ seq_len from {32,64,128,256,512,1024}
2. **T047**: MIL GPT-2 attention prefill generator — composes mil_builder helpers into complete per-layer attention program with 3 outputs (hidden, K, V for cache). Added `orion_mil_program_multi` for multi-output MIL programs. Added `orion_make_causal_mask_blob` for causal mask generation.
3. **T048**: MIL GPT-2 FFN prefill generator — LN2 → FC(768→3072) → GELU → Proj(3072→768) → Residual

### T049, T051-T053: ANE Prefill Pipeline
4. **T049**: Final LayerNorm MIL kernel — logits projection done on CPU (wte blob is 73MB, too large for ANE SRAM)
5. **T051**: Prompt padding — computes wte+wpe embeddings, pads to bucket size, transposes CPU [seq, d_model] → ANE [d_model, bucket]
6. **T052**: Full ANE prefill runner — compiles 25 programs (12 attn + 12 FFN + 1 final LN), evals sequentially, CPU logits projection via cblas_sgemv
7. **T053**: K,V extraction — transposes ANE output [d_model, bucket] → CPU [seq, d_model], stores into KV cache with head-split layout
8. **Test**: 31/31 pass — Full 12-layer ANE prefill: "The quick brown fox" → argmax=274 ("jumps"), exact CPU match, 5/5 top-5 overlap. K max error vs CPU: 0.0025. 30 compiles used.

### T054-T056: Hybrid Inference + Golden Tests + Benchmark
9. **T054**: Wired hybrid inference into CLI — `orion infer --ane` uses ANE prefill → CPU decode. Falls back to CPU if ANE fails. Output is identical between CPU and ANE modes.
10. **T055**: ANE golden vectors — "Hello" → 5 token exact match, "fox" → "jumps" exact match via ANE prefill + CPU decode.
11. **T056**: Benchmark — CPU prefill 13.5ms (3.4ms/tok) vs ANE 1399ms (349.8ms/tok, compile-dominated). ANE eval alone ~5ms. Program cache (M4) will eliminate recompilation.
12. **Test**: 34/34 pass — full M2 test suite. 105 compiles used across all tests.

---

## What Happened Earlier (Session 2 — CPU Forward Pass)

### T031-T036: CPU GPT-2 Forward Pass
1. **T031**: `weight_loader.h/m` — reads 196 BLOBFILE blobs (fp16→fp32), loads all GPT-2 124M weights
2. **T032**: Token + positional embedding lookup using vDSP_vadd
3. **T033**: CPU LayerNorm using vDSP (mean, variance, normalize, scale+shift)
4. **T034**: Multi-head self-attention with explicit causal mask (cblas_sgemm for Q@K^T and scores@V)
5. **T035**: FFN: Linear → GELU (tanh approx) → Linear using cblas_sgemm
6. **T036**: Full 12-layer GPT-2 forward pass: embed → 12×(LN→Attn→Residual→LN→FFN→Residual) → final LN → logits
7. All weight matrices use `CblasTrans` (converter transposes Conv1D [in,out] → blob [out,in])
8. Test: 6/6 pass — correct argmax for "Hello"→`,`, "The quick brown fox"→`jumps`, "Hello, world"→`!`
9. Max logit error vs PyTorch: ~0.073 (acceptable fp16 drift across 12 layers)

### T043-T046: Golden Vectors + Profiler
1. **T043**: Inference golden test — 5-token greedy generation matches PyTorch exactly, determinism verified
2. **T044**: Profiler core — p50/p90 percentile computation, RSS tracking, TFLOPS calculation
3. **T045**: Profiler print — formatted table output to stderr
4. **T046**: Wired into CLI — `orion infer` now prints profiler stats after generation

---

### T037-T042: KV Cache + Decode Loop + CLI
1. **T037**: KV cache store_prefill — rearranges [seq, n_head*head_dim] → [n_head, seq, head_dim] layout
2. **T038**: KV cache append — writes single position K,V to all heads
3. **T039**: Single-token decode step using cached K,V + cblas_sgemv for per-head attention
4. **T040**: Token sampling: argmax (temp=0) + temperature scaling + top-p nucleus sampling (xorshift64 RNG)
5. **T042**: Full CLI: `orion infer --prompt "..." --max_tokens N --temperature T --top_p P --seed S`
6. Performance: GPT-2 124M on M4 Max — prefill 3.1 ms/tok, decode 3.5 ms/tok (283 tok/s)
7. Test: test_decode 4/4 pass (prefill_kv matches forward, decode_step matches prefill, multi-step cross-check)

---

### T028-T030: GPT-2 BPE Tokenizer
1. **T028**: Full BPE tokenizer in Obj-C: byte-to-unicode mapping, NSRegularExpression pre-tokenization, iterative BPE merge
2. **T029**: 20 golden vectors generated from tiktoken (diverse prompts: contractions, Unicode, whitespace, numbers)
3. **T030**: Test runner loads golden JSON, verifies encode+decode for all 20 → 20/20 pass
4. ARC gotcha: `@autoreleasepool` inside encode releases the static regex from `dispatch_once` — removed inner pool
5. C struct + ObjC objects: must use `void*` + `CFBridgingRetain/Release` (same pattern as ane_runtime.m)
6. Tokenizer data: `vocab.json` (50257 entries) + `merges.txt` (49992 merges) stored in `tokenizer/data/`

---

### T027: Stories110M Weight Converter (previous session)
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

### Immediate — Continue M3 (Training Step Wiring)
**M3 — Training on ANE** (T057-T083, 15/27 done):
1. **T072** (XL): Single training step — forward (ANE) → loss (CPU) → backward (ANE) → dW (CPU) → Adam
2. **T073** (M): GCD async dW overlap (compute dW on CPU while ANE runs next layer)
3. **T074** (M): Gradient accumulation across microbatches
4. **T075-T076** (L): Checkpoint save/resume
5. **T077** (L): exec() restart at ~119 compile limit
6. **T078** (L): Recompile ANE programs after Adam weight update

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
