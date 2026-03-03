# Orion v2 — Atomic Task List

> Generated from `ORION_v2_ANE_LLM_SPEC.md`. Each task is completable in a single session, independently testable, and tagged with milestone, size, and dependencies.

---

## Phase 0 — Upstream Validation (M0)

### P0.1 — Clone & Build Upstream

- [ ] **T001** (S) — Clone maderix/ANE repo into `vendor/maderix-ane/` and verify directory structure matches expected layout [no dependencies]
  - *Acceptance:* `vendor/maderix-ane/training/train_large.m` exists
  - *File:* `vendor/`

- [ ] **T002** (S) — Clone ANEgpt repo into `vendor/anegpt/` and verify directory structure [no dependencies]
  - *Acceptance:* `vendor/anegpt/ane-training/training/train_large.m` and `vendor/anegpt/ane-training/bridge/` exist
  - *File:* `vendor/`

- [ ] **T003** (M) — Build maderix/ANE `train_large` binary using `xcrun clang` on macOS 15 / M4 Max [blocked by: T001]
  - *Acceptance:* Binary compiles with zero errors; `./train_large --help` or default invocation prints usage/starts
  - *Verify:* `xcrun clang -O2 -framework Foundation -framework IOSurface -framework CoreML -framework Accelerate -ldl -lobjc -o train_large vendor/maderix-ane/training/train_large.m`

- [ ] **T004** (M) — Run maderix/ANE `train_large` for 10+ steps and capture output [blocked by: T003]
  - *Acceptance:* Loss values printed for 10 consecutive steps; no crash or compiler exhaustion
  - *Record:* macOS version, clang version, output log → `docs/m0_upstream_log.txt`

- [ ] **T005** (M) — Build ANEgpt `train_large.m` and `train_large_ane.m` binaries [blocked by: T002]
  - *Acceptance:* Both binaries compile; `./train_large` starts training loop
  - *Verify:* Makefile in `vendor/anegpt/ane-training/training/`

- [ ] **T006** (S) — Build ANEgpt bridge `libane_bridge.dylib` [blocked by: T002]
  - *Acceptance:* `libane_bridge.dylib` built from source (not using checked-in binary)
  - *Verify:* `cd vendor/anegpt/ane-training/bridge && make`

- [ ] **T007** (M) — Run ANEgpt `train_large` for 10+ steps with TinyStories data [blocked by: T005, T010]
  - *Acceptance:* Loss printed, decreases over steps; checkpoint file created
  - *Record:* Output log → `docs/m0_anegpt_log.txt`

### P0.2 — Hello MIL Proof-of-Concept

- [ ] **T008** (M) — Implement minimal "hello MIL" program: compile a trivial MIL matmul, eval on ANE, read back result [blocked by: T004]
  - *Acceptance:* A standalone `.m` file that compiles MIL text → loads on ANE → evaluates with IOSurface input → reads output → prints result matching expected value
  - *File:* `experiments/hello_mil.m`

- [ ] **T009** (S) — Document the private ANE API calling sequence with code snippets extracted from upstream [blocked by: T004, T008]
  - *Acceptance:* `docs/ane_api_reference.md` with: dlopen, objc_getClass, compile, load, eval sequence; IOSurface setup; known gotchas
  - *File:* `docs/ane_api_reference.md`

### P0.3 — Data Acquisition

- [ ] **T010** (S) — Implement `scripts/download_data.sh` to fetch pretokenized TinyStories [no dependencies]
  - *Acceptance:* Script downloads `tinystories_data00.bin` (~41MB); validates file size; idempotent (skips if exists)
  - *File:* `scripts/download_data.sh` (update existing stub)

- [ ] **T011** (S) — Add `scripts/download_weights.sh` to fetch Stories110M pretrained weights [no dependencies]
  - *Acceptance:* Downloads `stories110M.bin` from HuggingFace to `model/weights/`; validates file
  - *File:* `scripts/download_weights.sh`

---

## Phase 1 — Core Runtime Foundation (M0→M1)

### P1.1 — IOSurface Tensor Layer

- [ ] **T012** (M) — Implement `orion_tensor_create` with correct ANE-compatible IOSurface properties [blocked by: T008]
  - *Acceptance:* Creates IOSurface with fp16 pixel format, `[1, C, 1, S]` layout, correct alignment; returns valid `IOSurfaceRef`
  - *Test:* Create tensor, write known values, read back, verify match
  - *File:* `core/iosurface_tensor.m`

- [ ] **T013** (S) — Implement `orion_tensor_write` and `orion_tensor_read` (raw fp16 copy) [blocked by: T012]
  - *Acceptance:* Lock/unlock + memcpy round-trips data exactly
  - *File:* `core/iosurface_tensor.m`

- [ ] **T014** (M) — Implement `orion_tensor_write_f32` and `orion_tensor_read_f32` with NEON fp16↔fp32 conversion [blocked by: T012]
  - *Acceptance:* Convert 1024 random floats → fp16 → back; max abs error < 1e-3 for values in [-10, 10]
  - *Reference:* ANEgpt `stories_io.h` NEON intrinsics
  - *File:* `core/iosurface_tensor.m`

### P1.2 — ANE Runtime Wrapper

- [ ] **T015** (L) — Implement `orion_compile_mil`: dlopen private framework, create `_ANEInMemoryModelDescriptor`, compile, load [blocked by: T008, T009]
  - *Acceptance:* Compile a known-good MIL program (matmul from T008) using the Orion API; returns non-NULL `OrionProgram*`
  - *Critical:* milText as `NSData*` (UTF-8), not `NSString*`; weight blobs as `NSDictionary*`
  - *File:* `core/ane_runtime.m`

- [ ] **T016** (M) — Implement `orion_eval`: build `_ANERequest` with IOSurface inputs/outputs and execute [blocked by: T015, T012]
  - *Acceptance:* Eval the matmul program from T015 with known input → verify output matches expected result
  - *File:* `core/ane_runtime.m`

- [ ] **T017** (M) — Implement `orion_release_program` with explicit ObjC object release under ARC [blocked by: T015]
  - *Acceptance:* Compile + release 20 programs in a loop; no crash; RSS does not grow unbounded
  - *File:* `core/ane_runtime.m`

- [ ] **T018** (S) — Write standalone ANE runtime integration test [blocked by: T015, T016, T017]
  - *Acceptance:* `tests/test_ane_runtime.m`: compile matmul → eval → verify → release → repeat 10×; all pass
  - *File:* `tests/test_ane_runtime.m`

### P1.3 — MIL Builder Helpers

- [ ] **T019** (M) — Implement `orion_mil_linear`: generate MIL text for `Y = X @ W + b` with BLOBFILE weight references [blocked by: T015]
  - *Acceptance:* Generated MIL text compiles via `orion_compile_mil`; eval produces correct linear transform output
  - *File:* `core/mil_builder.m`

- [ ] **T020** (M) — Implement `orion_mil_layernorm` (GPT-2 style) and `orion_mil_rmsnorm` (Llama style) [blocked by: T019]
  - *Acceptance:* Each produces MIL that compiles and evals to match CPU reference within fp16 tolerance
  - *File:* `core/mil_builder.m`

- [ ] **T021** (S) — Implement `orion_mil_gelu` and `orion_mil_silu` activations [blocked by: T019]
  - *Acceptance:* MIL compiles; output matches CPU reference for 256-element test vector
  - *File:* `core/mil_builder.m`

- [ ] **T022** (L) — Implement `orion_mil_causal_attention` with explicit decomposition (no SDPA) [blocked by: T019, T020]
  - *Acceptance:* Q@K^T / sqrt(d) with explicit causal mask → softmax → @V; output matches CPU reference
  - *Critical:* ANE ignores attn_mask — this MUST use decomposed ops
  - *File:* `core/mil_builder.m`

- [ ] **T023** (M) — Implement `orion_mil_program` wrapper to compose ops into complete MIL function [blocked by: T019]
  - *Acceptance:* Wraps body + inputs + outputs into valid MIL program text that compiles
  - *File:* `core/mil_builder.m`

### P1.4 — BLOBFILE Weight Format

- [ ] **T024** (M) — Implement BLOBFILE writer: 128-byte header (magic, size, offset) + fp16 data [blocked by: T009]
  - *Acceptance:* Write a known weight tensor → read back → header fields correct; data matches within fp16 tolerance
  - *Reference:* Header format from spec §4.3
  - *File:* `model/convert/hf_to_blobs_gpt2.py` (implement `make_blob_header` + `convert_tensor_to_blob`)

- [ ] **T025** (M) — Implement GPT-2 HF→BLOBFILE converter: load all weights from HuggingFace, write per-layer blobs [blocked by: T024]
  - *Acceptance:* `python hf_to_blobs_gpt2.py --output model/blobs/gpt2_124m/` creates blob files for all 12 layers + embeddings
  - *File:* `model/convert/hf_to_blobs_gpt2.py`

- [ ] **T026** (S) — Implement weight conversion tests: round-trip fidelity, header format validation [blocked by: T025]
  - *Acceptance:* `tests/test_weight_convert.py` passes: header magic/size correct; fp16 max error < 1e-3 vs fp32
  - *File:* `tests/test_weight_convert.py`

- [ ] **T027** (M) — Implement Llama2/Stories HF→BLOBFILE converter for Stories110M [blocked by: T024, T011]
  - *Acceptance:* `python hf_to_blobs_llama.py --checkpoint model/weights/stories110M.bin --output model/blobs/stories110m/` creates per-layer blobs (Wq, Wk, Wv, Wo, W1, W2, W3, rms_norm)
  - *File:* `model/convert/hf_to_blobs_llama.py`

---

## Phase 2 — CPU Baseline Inference (M1)

### P2.1 — GPT-2 BPE Tokenizer

- [ ] **T028** (L) — Implement GPT-2 BPE tokenizer in Obj-C: load vocab.json + merges.txt, encode, decode [no dependencies]
  - *Acceptance:* Encode "Hello, world!" → token ids matching tiktoken reference; decode back to original string
  - *File:* `tokenizer/gpt2_bpe.m`

- [ ] **T029** (M) — Generate 20 golden test vectors using tiktoken Python reference, store in test fixture [blocked by: T028]
  - *Acceptance:* `tests/tokenizer_golden.json` with 20 diverse prompts + expected token ids (from tiktoken)
  - *File:* `tests/tokenizer_golden.json`

- [ ] **T030** (M) — Implement tokenizer golden test runner [blocked by: T028, T029]
  - *Acceptance:* `tests/test_tokenizer.m` loads golden vectors, runs encode/decode, all 20 pass
  - *File:* `tests/test_tokenizer.m`

### P2.2 — CPU GPT-2 Forward Pass

- [ ] **T031** (M) — Implement GPT-2 weight loading: read BLOBFILE blobs into contiguous fp32 arrays [blocked by: T025]
  - *Acceptance:* Load all GPT-2 124M weights; total parameter count matches 124M; spot-check embedding values
  - *File:* `kernels/inference/decode_cpu.m` or new `model/weight_loader.{h,m}`

- [ ] **T032** (M) — Implement token embedding + positional embedding lookup [blocked by: T031]
  - *Acceptance:* Given token ids [15496, 11, 995, 0] (="Hello, world!"), produces correct embedding vectors matching PyTorch reference
  - *File:* `kernels/inference/decode_cpu.m`

- [ ] **T033** (M) — Implement CPU LayerNorm (GPT-2 style) using vDSP [no dependencies]
  - *Acceptance:* Output matches PyTorch `nn.LayerNorm` within 1e-5 for a 768-dim test vector
  - *File:* `kernels/inference/decode_cpu.m`

- [ ] **T034** (M) — Implement CPU attention: Q,K,V projections + scaled dot-product with causal mask + output projection [blocked by: T033]
  - *Acceptance:* Single-layer attention output matches PyTorch reference within 1e-4 for a 4-token input
  - *File:* `kernels/inference/decode_cpu.m`

- [ ] **T035** (M) — Implement CPU FFN: Linear → GELU → Linear [blocked by: T033]
  - *Acceptance:* FFN output matches PyTorch reference within 1e-4
  - *File:* `kernels/inference/decode_cpu.m`

- [ ] **T036** (L) — Implement full CPU GPT-2 prefill forward pass: embed → 12× (Attn + FFN) → final LN → logits [blocked by: T032, T034, T035]
  - *Acceptance:* Given "The quick brown fox", top-5 logits match PyTorch reference; deterministic output with seed
  - *File:* `kernels/inference/decode_cpu.m`

### P2.3 — KV Cache & Decode Loop

- [ ] **T037** (M) — Implement `orion_kv_cache_store_prefill`: populate KV cache from prefill K,V outputs [blocked by: T036]
  - *Acceptance:* After prefill of 4 tokens, `cache->current_len == 4`; K,V data retrievable and matches prefill output
  - *File:* `kernels/inference/kv_cache.m`

- [ ] **T038** (M) — Implement `orion_kv_cache_append`: add single-token K,V during decode [blocked by: T037]
  - *Acceptance:* Append 3 tokens sequentially; `cache->current_len` increments correctly; data at each position correct
  - *File:* `kernels/inference/kv_cache.m`

- [ ] **T039** (M) — Implement single-step CPU decode: embed token → attention with KV cache → FFN → logits [blocked by: T036, T038]
  - *Acceptance:* After prefill "The", decode step produces logits; top token is plausible continuation
  - *File:* `kernels/inference/decode_cpu.m`

- [ ] **T040** (M) — Implement `orion_sample_token`: temperature scaling → softmax → top-p nucleus sampling [blocked by: T039]
  - *Acceptance:* temp=0 always returns argmax; temp=1.0 + top_p=0.9 produces varied but plausible tokens; seed reproducibility
  - *File:* `kernels/inference/decode_cpu.m`

### P2.4 — Inference CLI Wiring

- [ ] **T041** (M) — Implement CLI argument parsing for `orion infer`: --model, --weights, --prompt, --max_new_tokens, --temperature, --top_p, --seed, --bucket, --metrics [blocked by: T040]
  - *Acceptance:* Parses all flags correctly; prints error for missing required args; --help works
  - *File:* `apps/cli/commands/infer.m`

- [ ] **T042** (L) — Wire `orion infer` end-to-end on CPU: tokenize → prefill → decode loop → detokenize → print [blocked by: T041, T030, T036, T040]
  - *Acceptance:* `./orion infer --model gpt2_124m --prompt "Once upon a time" --max_new_tokens 50 --seed 42` prints coherent English text
  - *File:* `apps/cli/commands/infer.m`

- [ ] **T043** (M) — Generate inference golden test vectors: run CPU inference with fixed prompts/seeds, record expected tokens [blocked by: T042]
  - *Acceptance:* `tests/test_infer_golden.json` populated with 5+ test cases, each with prompt + seed + expected first 10 tokens
  - *File:* `tests/test_infer_golden.json`

### P2.5 — Profiler

- [ ] **T044** (M) — Implement profiler: mach_absolute_time timing, sample ring buffer, p50/p90 percentile computation [no dependencies]
  - *Acceptance:* Record 100 samples → p50/p90 within expected range; `orion_get_rss()` returns non-zero
  - *File:* `core/profiler.m`

- [ ] **T045** (S) — Implement `orion_prof_print`: formatted table output (prefill ms, decode tok/s, TFLOPS, RSS) [blocked by: T044]
  - *Acceptance:* Prints clean table to stdout matching ANEgpt reporting style
  - *File:* `core/profiler.m`

- [ ] **T046** (S) — Wire profiler into `orion infer --metrics` flag [blocked by: T044, T045, T042]
  - *Acceptance:* `./orion infer --model gpt2_124m --prompt "Hello" --max_new_tokens 20 --metrics` prints text + metrics table
  - *File:* `apps/cli/commands/infer.m`

---

## Phase 3 — ANE Prefill Inference (M2)

### P3.1 — MIL Kernel Generation for GPT-2

- [ ] **T047** (L) — Implement MIL generator for GPT-2 attention prefill kernel (per-layer) [blocked by: T019, T022, T023]
  - *Acceptance:* Generated MIL compiles via `orion_compile_mil` with GPT-2 layer 0 weights; eval output matches CPU reference within fp16 tolerance
  - *File:* `kernels/inference/gpt2_prefill_attn.milgen.h` → implement in new `.m`

- [ ] **T048** (L) — Implement MIL generator for GPT-2 FFN prefill kernel (per-layer) [blocked by: T019, T021, T023]
  - *Acceptance:* Generated MIL compiles; FFN output matches CPU reference within fp16 tolerance
  - *File:* `kernels/inference/gpt2_prefill_ffn.milgen.h` → implement in new `.m`

- [ ] **T049** (M) — Implement MIL generator for final LayerNorm + logits projection [blocked by: T020, T019]
  - *Acceptance:* Final LN + embed^T → logits; matches CPU reference
  - *File:* new `kernels/inference/gpt2_final.milgen.h`

### P3.2 — Bucket Routing

- [ ] **T050** (S) — Implement bucket selection: given prompt length, choose smallest bucket ≥ length from {32, 64, 128, 256, 512, 1024} [no dependencies]
  - *Acceptance:* `select_bucket(17) == 32`, `select_bucket(64) == 64`, `select_bucket(1025)` → error
  - *File:* new `core/bucket.{h,m}` or inline in inference runner

- [ ] **T051** (M) — Implement prompt padding + position encoding for bucketed ANE input [blocked by: T050, T014]
  - *Acceptance:* Pad 17-token prompt to bucket 32; write to IOSurface; positions correct; padding tokens zeroed
  - *File:* `kernels/inference/` or `core/`

### P3.3 — ANE Prefill Pipeline

- [ ] **T052** (XL) — Implement ANE prefill runner: compile all 12 layers (attn + FFN) for chosen bucket, eval sequentially, collect logits + K,V [blocked by: T047, T048, T049, T051, T031]
  - *Acceptance:* Prefill "The quick brown fox" on ANE → top-5 logits match CPU prefill within fp16 tolerance (top-k fallback OK)
  - *File:* new `kernels/inference/prefill_ane.{h,m}`

- [ ] **T053** (M) — Extract K,V from ANE prefill output into KV cache for CPU decode [blocked by: T052, T037]
  - *Acceptance:* After ANE prefill, KV cache populated correctly; subsequent CPU decode step produces plausible token
  - *File:* `kernels/inference/prefill_ane.m`

- [ ] **T054** (L) — Wire hybrid inference: ANE prefill → CPU decode loop [blocked by: T052, T053, T039, T040]
  - *Acceptance:* `orion infer --model gpt2_124m --prompt "Once upon a time" --max_new_tokens 50` produces coherent text using ANE prefill + CPU decode
  - *File:* `apps/cli/commands/infer.m`

- [ ] **T055** (M) — Run inference golden tests with ANE prefill path [blocked by: T054, T043]
  - *Acceptance:* All golden tests pass (exact match or top-k fallback for fp16 drift)
  - *File:* `tests/test_infer_golden.json` + test runner

- [ ] **T056** (S) — Benchmark ANE prefill vs CPU-only prefill; record speedup [blocked by: T054, T046]
  - *Acceptance:* `orion infer --metrics` shows ANE prefill time < CPU prefill time; speedup documented
  - *Record:* Results → `docs/m2_benchmarks.md`

---

## Phase 4 — Training on ANE (M3)

### P4.1 — CPU Training Ops

- [ ] **T057** (M) — Implement `orion_cpu_rmsnorm`: RMSNorm forward using vDSP [no dependencies]
  - *Acceptance:* Output matches PyTorch `rms_norm` within 1e-5 for dim=768 input
  - *File:* `kernels/training/stories_cpu_ops.m`

- [ ] **T058** (M) — Implement `orion_cpu_cross_entropy`: NLL loss + softmax gradient [blocked by: T057]
  - *Acceptance:* Loss value matches PyTorch `F.cross_entropy`; gradient matches `autograd` within 1e-4
  - *File:* `kernels/training/stories_cpu_ops.m`

- [ ] **T059** (S) — Implement `orion_cpu_embedding`: token lookup from embedding table [no dependencies]
  - *Acceptance:* Lookup tokens [0, 100, 31999] → correct rows from embedding matrix
  - *File:* `kernels/training/stories_cpu_ops.m`

- [ ] **T060** (M) — Implement `orion_cpu_adam_step` with bias correction [no dependencies]
  - *Acceptance:* 100 steps on a quadratic function converge to minimum; matches PyTorch Adam output
  - *File:* `kernels/training/stories_cpu_ops.m`

- [ ] **T061** (M) — Implement `orion_cpu_dw_accum` via cblas_sgemm [no dependencies]
  - *Acceptance:* dW = X^T @ dY matches numpy `X.T @ dY` within 1e-5 for m=256, n=768, k=768
  - *File:* `kernels/training/stories_cpu_ops.m`

### P4.2 — SentencePiece Tokenizer

- [ ] **T062** (M) — Implement SentencePiece tokenizer wrapper for Llama2/Stories vocab [no dependencies]
  - *Acceptance:* Encode/decode round-trip for "Once upon a time there was a little girl" using Llama2 32k vocab
  - *Note:* For training, pretokenized data is primary; this is for interactive inference with Stories model
  - *File:* `tokenizer/sentencepiece_wrap.m`

### P4.3 — Training Data Loader

- [ ] **T063** (M) — Implement pretokenized data loader: read `tinystories_data00.bin`, produce (input, target) batches [blocked by: T010]
  - *Acceptance:* Load file → iterate batches of (seq_len=256) → targets = input shifted by 1; no OOB reads
  - *File:* new `kernels/training/data_loader.{h,m}`

### P4.4 — ANE Training Kernels (MIL Generation)

- [ ] **T064** (L) — Implement MIL generator for `fwdAttn` kernel: RMSNorm → QKV → attention → Wo, with forward taps (Q, K, V, scores) [blocked by: T019, T020, T022, T023]
  - *Acceptance:* MIL compiles with Stories110M dim=768 weights; output matches CPU reference within fp16 tolerance
  - *Reference:* ANEgpt `stories_mil.h`, `forward.h`
  - *File:* `kernels/training/stories_train_kernels.milgen.h` → new `.m`

- [ ] **T065** (L) — Implement MIL generator for `fwdFFN` kernel: RMSNorm → SwiGLU (W1, W3, SiLU, W2) [blocked by: T019, T020, T021, T023]
  - *Acceptance:* MIL compiles; FFN output matches CPU reference
  - *File:* `kernels/training/stories_train_kernels.milgen.h`

- [ ] **T066** (L) — Implement MIL generator for `ffnBwd` kernel: W2^T → SiLU_bwd → W1^T, W3^T [blocked by: T065]
  - *Acceptance:* Backward dx matches PyTorch autograd FFN backward within fp16 tolerance
  - *File:* `kernels/training/stories_train_kernels.milgen.h`

- [ ] **T067** (L) — Implement MIL generator for `sdpaBwd1` kernel: Wo^T → SDPA backward part 1 (dV, attn probs, dp) [blocked by: T064]
  - *Acceptance:* dV and intermediate gradients match PyTorch reference
  - *File:* `kernels/training/stories_train_kernels.milgen.h`

- [ ] **T068** (M) — Implement MIL generator for `sdpaBwd2` kernel: softmax grad → dQ, dK (weight-free) [blocked by: T067]
  - *Acceptance:* dQ, dK match PyTorch reference; no weight blobs needed (weight-free kernel)
  - *File:* `kernels/training/stories_train_kernels.milgen.h`

- [ ] **T069** (L) — Implement MIL generator for `qkvBwd` kernel: Wq^T + Wk^T + Wv^T → dx [blocked by: T068]
  - *Acceptance:* Final dx matches PyTorch attention backward
  - *File:* `kernels/training/stories_train_kernels.milgen.h`

- [ ] **T070** (M) — Implement MIL generator for classifier forward on ANE (optional offload) [blocked by: T019]
  - *Acceptance:* embed @ x produces logits matching CPU; 10.2× speedup target
  - *File:* `kernels/training/classifier_softmax.milgen.h`

- [ ] **T071** (M) — Implement MIL generator for vocab softmax on ANE (optional offload) [blocked by: T019]
  - *Acceptance:* Softmax over 32000 classes matches CPU; 33.8× speedup target
  - *File:* `kernels/training/classifier_softmax.milgen.h`

### P4.5 — Training Loop

- [ ] **T072** (XL) — Implement single training step: forward (ANE) → loss (CPU) → backward (ANE) → dW (CPU) [blocked by: T064, T065, T066, T067, T068, T069, T058, T061]
  - *Acceptance:* One step completes; loss is finite; gradients are non-zero
  - *File:* `apps/cli/commands/train.m` or new `kernels/training/train_step.{h,m}`

- [ ] **T073** (M) — Implement GCD-async dW overlap: dispatch cblas_sgemm dW to background queue, deferred wait [blocked by: T072]
  - *Acceptance:* Training step with async dW is faster than synchronous; results identical
  - *File:* `kernels/training/train_step.m`

- [ ] **T074** (M) — Implement gradient accumulation over N micro-batches before Adam update [blocked by: T072, T060]
  - *Acceptance:* Accumulate 4 micro-batches → single Adam step; equivalent to batch_size × 4 single step
  - *File:* `kernels/training/train_step.m`

- [ ] **T075** (L) — Implement checkpoint save: write `OrionCkptHdr` + all weights + Adam state to file [blocked by: T072]
  - *Acceptance:* Checkpoint file written with correct magic (0x424C5A54), version, step, loss; file loadable
  - *File:* new `core/checkpoint.{h,m}`

- [ ] **T076** (L) — Implement checkpoint resume: load checkpoint → restore weights + Adam state → continue training [blocked by: T075]
  - *Acceptance:* Train 10 steps → save → resume → train 10 more; loss trajectory is smooth (no discontinuity)
  - *File:* `core/checkpoint.m`

- [ ] **T077** (L) — Implement exec() restart: save checkpoint → exec() self → resume → continue training [blocked by: T076]
  - *Acceptance:* Train through exec() restart boundary; loss continues decreasing; ~119 compile limit avoided
  - *File:* `apps/cli/commands/train.m`

- [ ] **T078** (L) — Implement recompilation of weight-bearing kernels after Adam update (Mode A training loop) [blocked by: T072, T015]
  - *Acceptance:* After Adam step, all 60 weight-bearing kernels recompiled with new weights; next forward pass uses updated weights
  - *File:* `kernels/training/train_step.m`

### P4.6 — Training CLI Wiring

- [ ] **T079** (M) — Implement CLI argument parsing for `orion train`: --model, --dataset, --seq, --batch, --grad_accum, --lr, --steps, --checkpoint_every, --resume [blocked by: T072]
  - *Acceptance:* Parses all flags; validates ranges; --help works
  - *File:* `apps/cli/commands/train.m`

- [ ] **T080** (L) — Wire `orion train` end-to-end: data loader → training loop → checkpoint → exec() restart [blocked by: T079, T063, T072, T073, T074, T075, T076, T077, T078]
  - *Acceptance:* `./orion train --model stories110m --dataset tinystories --steps 50 --checkpoint_every 25` completes; loss decreases; 2 checkpoints saved
  - *File:* `apps/cli/commands/train.m`

- [ ] **T081** (S) — Wire profiler into training: ms/step, compile time %, loss, TFLOPS [blocked by: T080, T044]
  - *Acceptance:* Training loop prints per-step metrics matching ANEgpt reporting style
  - *File:* `apps/cli/commands/train.m`

### P4.7 — Training Tests

- [ ] **T082** (M) — Implement training smoke test: 50 steps, assert loss decreases [blocked by: T080]
  - *Acceptance:* `tests/test_train_smoke.m` runs 50 steps; `loss[49] < loss[0]`; no crash
  - *File:* `tests/test_train_smoke.m`

- [ ] **T083** (M) — Implement checkpoint resume test: train → save → resume → verify continuity [blocked by: T080]
  - *Acceptance:* Loss after resume matches expected trajectory; Adam state restored correctly
  - *File:* `tests/test_train_smoke.m` (extend)

---

## Phase 5 — Weight Swapping (M4)

- [ ] **T084** (L) — Implement `orion_get_or_compile` program cache with composite key: (kernel_name, layer_idx, weights_id, bucket, config_hash) [blocked by: T015, T017]
  - *Acceptance:* First call compiles; second call with same key returns cached program (no recompile); different weights_id triggers recompile
  - *File:* `core/ane_program_cache.m`

- [ ] **T085** (M) — Implement `orion_cache_evict` and `orion_cache_clear` with safe ObjC release [blocked by: T084]
  - *Acceptance:* Evict by weights_id removes correct entries; clear removes all; RSS does not grow after eviction
  - *File:* `core/ane_program_cache.m`

- [ ] **T086** (M) — Implement `weights_id` abstraction: auto-incrementing for training, named for inference [blocked by: T084]
  - *Acceptance:* Training: weights_id = "step_00001", "step_00002"...; Inference: "base", "finetune_chat"
  - *File:* `core/ane_program_cache.m`

- [ ] **T087** (M) — Implement CLI for `orion bench swap`: --model, --ckpt_a, --ckpt_b, --iters [blocked by: T084, T085]
  - *Acceptance:* Parses args; --help works
  - *File:* `apps/cli/commands/bench.m`

- [ ] **T088** (XL) — Implement swap endurance test: alternate two checkpoints 100×, verify stability [blocked by: T087, T076]
  - *Acceptance:* 100 iterations of (load ckpt A → compile → eval → evict → load ckpt B → compile → eval → evict); process alive; RSS stable (< 2× initial)
  - *File:* `tests/test_program_swap.m`

- [ ] **T089** (S) — Wire swap benchmark into `orion bench swap` CLI [blocked by: T088, T087]
  - *Acceptance:* `./orion bench swap --model stories110m --ckpt_a ckpt_01000 --ckpt_b ckpt_01100 --iters 100` runs and prints per-swap timing + memory report
  - *File:* `apps/cli/commands/bench.m`

---

## Phase 6 — Demo App & Polish (M5)

- [ ] **T090** (M) — Create Xcode project or Makefile for building the full `orion` CLI binary with all source files [blocked by: T042, T080]
  - *Acceptance:* Single build command produces `orion` binary with infer + train + bench all functional
  - *File:* `Makefile` or `orion.xcodeproj/`

- [ ] **T091** (M) — Implement SwiftUI `ContentView` with prompt input, generate button, streaming text output [blocked by: T042]
  - *Acceptance:* App launches; user types prompt; taps Generate; text appears progressively
  - *File:* `apps/macapp/OrionApp.swift`

- [ ] **T092** (M) — Implement `ModelRunner` bridging header: Swift → C API calls for inference [blocked by: T091, T042]
  - *Acceptance:* Swift calls `orion_infer_init` / `orion_prefill_ane` / `orion_decode_cpu_step` via bridging header
  - *File:* `apps/macapp/ModelRunner.swift` + bridging header

- [ ] **T093** (S) — Add metrics overlay to SwiftUI app: tokens/sec, prefill ms, ANE utilization, RSS [blocked by: T092, T044]
  - *Acceptance:* Metrics display updates live during generation
  - *File:* `apps/macapp/OrionApp.swift`

- [ ] **T094** (M) — Write README benchmarks table: prefill ms, decode tok/s, TFLOPS, memory for GPT-2 + Stories [blocked by: T056, T081]
  - *Acceptance:* README.md has a formatted table with actual measured numbers
  - *File:* `README.md`

- [ ] **T095** (S) — Final wiring audit: every component imported, every IPC handler registered, all types match [blocked by: T090, T091, T092]
  - *Acceptance:* Checklist: all headers imported → all .m files compiled → CLI commands dispatch → app builds → no dead code

---

## Phase 7 — Stretch: LoRA Adapter-as-Input (M6)

- [ ] **T096** (L) — Implement MIL generator for LoRA-fused linear: `Y = X@W_base + scale * ((X@A)@B)` where A,B are IOSurface inputs [blocked by: T019, T015]
  - *Acceptance:* MIL compiles; A,B passed as IOSurface inputs (not baked); output matches CPU reference
  - *File:* new `kernels/inference/lora_fused.milgen.h`

- [ ] **T097** (M) — Implement LoRA adapter loading: read A,B matrices from file → IOSurface [blocked by: T096, T014]
  - *Acceptance:* Load two different LoRA adapters; each produces distinct IOSurface tensors
  - *File:* new `core/lora_adapter.{h,m}`

- [ ] **T098** (L) — Demonstrate hot-swap: run base+LoRA_A → swap to base+LoRA_B without recompilation [blocked by: T096, T097]
  - *Acceptance:* Same compiled base program; swap IOSurface inputs for A,B; output changes; no recompile
  - *File:* `tests/` or `experiments/`

---

## Dependency Graph (Critical Path)

```
T001/T002 → T003/T005 → T004/T007 → T008 → T015 → T016 → T018
                                       ↓
                                      T009
                                       ↓
T010/T011 ──→ T024 → T025 → T026/T027
                              ↓
T028 → T029 → T030           T031
                               ↓
T033 → T034/T035 → T036 → T037 → T038 → T039 → T040
                                                   ↓
                                          T041 → T042 → T043
                                                   ↓
T012 → T013/T014    T019 → T020/T021/T022/T023    ↓
                      ↓                             ↓
               T047/T048/T049 → T052 → T053 → T054 → T055/T056
                      ↓
               T064-T069 → T072 → T073/T074/T075 → T076 → T077
                                    ↓                        ↓
                              T078 → T080 → T081/T082/T083
                                       ↓
                              T084 → T085/T086 → T088 → T089
                                                   ↓
                                          T090 → T091 → T092 → T093
```

**Critical path:** T001 → T003 → T004 → T008 → T015 → T016 → T019 → T047 → T052 → T054 (ANE inference) **or** → T064 → T072 → T080 (training)

---

## Risks & Unknowns

| # | Risk | Impact | Mitigation | Related Tasks |
|---|------|--------|------------|---------------|
| R1 | Private ANE API breaks on macOS update | All ANE work blocked | Pin macOS 15.x; document exact versions in T009 | T003, T008, T015 |
| R2 | ~119 compile limit per process | Training halts mid-epoch | exec() restart in T077; compile budget tracking | T077, T078 |
| R3 | Compiler memory leak | OOM during swap endurance | Strict ARC release in T017; cache eviction in T085 | T017, T085, T088 |
| R4 | SDPA ignores causal masks | Wrong attention outputs | Explicit decomposition in T022; verified in T008 | T022, T047, T064 |
| R5 | GPT-2 BPE tokenizer complexity | Large implementation effort (T028=L) | Consider wrapping tiktoken via Python subprocess as fallback | T028 |
| R6 | 32000-channel conv rejected by ANE | Classifier backward on CPU only | Accept CPU fallback; optional ANE offload in T070/T071 | T070, T071, T072 |
| R7 | fp16 numerical drift | Golden tests fail | Two-tier tolerance: exact + top-k fallback (spec §10.2) | T043, T055 |
| R8 | IOSurface alignment requirements unknown | Tensor creation fails | Extract from upstream in T008/T009; trial-and-error | T012 |
| R9 | SRAM spill (>32MB working set) | 30% perf degradation | Profile per-kernel working set; keep under 32MB | T052, T064 |
| R10 | LoRA-as-input (Mode B) may not be feasible | M6 not achievable | Marked as stretch; T096 is first validation point | T096, T097, T098 |

---

## Size Summary

| Size | Count | Estimated effort per task |
|------|-------|--------------------------|
| S | 18 | < 1 hour |
| M | 52 | 1-3 hours |
| L | 20 | 3-6 hours |
| XL | 4 | 6-12 hours |
| **Total** | **94** | |

## Phase Summary

| Phase | Tasks | Milestone | Focus |
|-------|-------|-----------|-------|
| 0 — Upstream Validation | T001-T011 | M0 | Clone, build, run upstream; hello MIL; data |
| 1 — Core Runtime | T012-T027 | M0→M1 | IOSurface, ANE runtime, MIL builder, weight format |
| 2 — CPU Inference | T028-T046 | M1 | Tokenizer, CPU forward, KV cache, decode, CLI, profiler |
| 3 — ANE Prefill | T047-T056 | M2 | MIL kernels, bucket routing, hybrid inference |
| 4 — ANE Training | T057-T083 | M3 | CPU ops, training kernels, training loop, checkpoint |
| 5 — Weight Swapping | T084-T089 | M4 | Program cache, swap endurance |
| 6 — Demo & Polish | T090-T095 | M5 | Build system, SwiftUI app, README, audit |
| 7 — LoRA Stretch | T096-T098 | M6 | Adapter-as-input hot swap |
