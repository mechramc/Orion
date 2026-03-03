# ORION v3 — End-to-End On-Device LLM on Apple Silicon (ANE Training + Inference)

**Target machine:** Mac Studio **M4 Max, 64 GB unified memory**
**Target OS:** macOS 15+ (Sequoia)

---

## References

### Primary upstream repos

| Repo | Author | License | URL | Role |
|------|--------|---------|-----|------|
| **ANEgpt** | Vipul Divyanshu | MIT | [github.com/vipuldivyanshu92/ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt) | Full LLM training harness: 12-layer Stories110M on ANE, Python bridge, nanochat fork |
| **maderix/ANE** | Manjeet Singh | MIT | [github.com/maderix/ANE](https://github.com/maderix/ANE) | Core private API reverse-engineering, MIL compilation, IOSurface patterns, benchmarks |

### Additional references

| Resource | URL | Relevance |
|----------|-----|-----------|
| hollance/neural-engine | [github.com/hollance/neural-engine](https://github.com/hollance/neural-engine) | Community docs on ANE internals (2.4k stars) |
| apple/ml-ane-transformers | [github.com/apple/ml-ane-transformers](https://github.com/apple/ml-ane-transformers) | Official Apple CoreML-optimized transformer reference |
| smpanaro/more-ane-transformers | [github.com/smpanaro/more-ane-transformers](https://github.com/smpanaro/more-ane-transformers) | GPT inference on ANE via CoreML (~5 words/sec gpt2-xl) |
| Anemll/Anemll | [github.com/Anemll/Anemll](https://github.com/Anemll/Anemll) | ANE ML library, Gemma 3 port (Beta 0.3.5) |
| Apple coremltools MIL docs | [apple.github.io/coremltools](https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html) | Official MIL documentation (public layer) |

> This spec is written to be handed to a coding agent. It defines milestones, repo layout, APIs, and acceptance tests.

---

## 0) Executive goal

Build a **fully offline**, **end-to-end** LLM system on macOS that supports:

1. **Inference on device** (CLI + simple macOS app):
   - Prompt → tokenize → **ANE full forward pass** → **CPU sampling** → ANE next-token forward → repeat until done.
   - The entire transformer executes on ANE. CPU handles only token sampling and embedding lookup.
2. **Training on device** (ANEgpt-style):
   - Forward + backward **dx on ANE**, **dW on CPU** (cblas_sgemm via GCD async), **Adam + checkpoint/resume**, and sustained training via exec() restart to work around compiler limits.
3. **Runtime weight swapping** (two modes):
   - **Mode A (guaranteed):** swap weights by **fast recompile + program cache + safe release**. Weights are baked into ANE programs as BLOBFILE constants — "swapping" means recompiling with new weight blobs.
   - **Mode B (stretch):** true runtime swapping via **adapter-as-input** (LoRA/IA3) without recompiling base weights.

---

## 1) Non-goals (v3)

- iOS build (macOS only for v3).
- Large-scale distributed training.
- Perfect parity with PyTorch for all ops (fp16 numerics tolerated; token-level golden tests are required).

---

## 2) Target model(s)

### 2.1 Inference demo model (v3)
- **GPT-2 124M** (fast to demo; widely reproducible).
- Buckets: **32 / 64 / 128 / 256 / 512 / 1024** prompt tokens.

### 2.2 Training demo model (v3)
- **Stories110M (Llama2-style)** matching ANEgpt's `stories_config.h`:

```c
#define DIM     768       // model dimension
#define HIDDEN  2048      // FFN hidden size (SwiGLU)
#define HEADS   12        // attention heads
#define HD      64        // per-head dimension (DIM/HEADS)
#define SEQ     256       // sequence length
#define NLAYERS 12        // transformer layers
#define VOCAB   32000     // Llama2 BPE vocabulary
```

Total parameters: ~109.53M (84.95M transformer + 24.58M embedding).

> Both models are **reference implementations** of the Orion runtime abstraction layer (see §5.1). Additional architectures can be added by implementing the `OrionModel` interface.

---

## 3) Dependencies & build requirements

### 3.1 System requirements
- macOS 15+ (Sequoia) on Apple Silicon (M4 Max target)
- Xcode Command Line Tools: `xcode-select --install`
- No third-party compiled dependencies required for core runtime

### 3.2 System frameworks (linked via `-framework`)

| Framework | Usage |
|-----------|-------|
| **Foundation** | ObjC runtime, NSData, NSDictionary for ANE API |
| **IOSurface** | Zero-copy shared memory tensors between CPU and ANE |
| **CoreML** | Required for private ANE framework loading |
| **Accelerate** | cblas_sgemm (dW), vDSP (RMSNorm, softmax, Adam) |

Additional linker flags: `-ldl` (dlopen for private framework), `-lobjc`

### 3.3 Private framework (runtime loaded)

```
/System/Library/PrivateFrameworks/AppleNeuralEngine.framework
```

Loaded via `dlopen()` + `objc_getClass()` at runtime. Not linked at build time.

### 3.4 Python dependencies (weight conversion + optional training path)

| Package | Version | Purpose |
|---------|---------|---------|
| torch | >=2.9 | Weight extraction from HuggingFace checkpoints |
| transformers | >=4.57 | GPT-2 tokenizer + model loading |
| tiktoken | >=0.11 | GPT-2 BPE encoding |
| sentencepiece | latest | Llama2/Stories tokenizer |
| numpy | latest | Tensor manipulation in converters |

### 3.5 Build commands

```bash
# Core Obj-C training binary (single-file compilation, no Xcode project needed)
xcrun clang -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK \
  -framework Foundation -framework IOSurface \
  -framework Accelerate \
  -ldl -I . \
  -o orion apps/cli/main.m \
  apps/cli/commands/*.m \
  core/*.m \
  kernels/inference/*.m \
  kernels/training/*.m \
  tokenizer/*.m \
  model/*.m

# Weight conversion
python model/convert/hf_to_blobs_gpt2.py
python model/convert/hf_to_blobs_llama.py

# Download training data (TinyStories, ~41MB pretokenized)
bash scripts/download_data.sh
```

### 3.6 Optional: pre-trained weights

```bash
# Stories110M (Karpathy's tinyllamas)
mkdir -p model/weights
curl -L -o model/weights/stories110M.bin \
  https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin

# GPT-2 124M (via HuggingFace transformers)
python model/convert/hf_to_blobs_gpt2.py --output model/blobs/gpt2_124m/
```

---

## 4) Core technical constraints (must respect)

### 4.1 Private ANE API surface

The ANE is accessed via reverse-engineered private Objective-C classes discovered by `maderix/ANE`:

| Class | Role |
|-------|------|
| `_ANEClient` | Main entry point. `sharedConnection` → compile → load → evaluate pipeline |
| `_ANECompiler` | Compiles MIL text into E5 binary format |
| `_ANEInMemoryModel` | In-memory compilation (no filesystem temp files for model itself) |
| `_ANEInMemoryModelDescriptor` | Accepts MIL as `NSData*` (UTF-8) + weight blobs as `NSDictionary*` |
| `_ANEModel` | Model reference; holds `programHandle` after load |
| `_ANERequest` | Evaluation request with IOSurface input/output specs |
| `_ANEIOSurfaceObject` | Wraps IOSurface memory for tensor I/O |

**API workflow:**
1. `[_ANEClient sharedConnection]` — obtain hardware connection
2. Create `_ANEInMemoryModelDescriptor` with MIL text (`NSData*`) + weight blobs (`NSDictionary*`)
3. `compileModel:options:qos:error:` — compile MIL → E5 binary
4. `loadModel:options:qos:error:` — load to ANE (assigns `programHandle`, queue depth = 127)
5. Allocate IOSurface buffers for tensor data
6. Build `_ANERequest` with input/output IOSurface specs
7. `evaluateWithModel:options:request:qos:error:` — execute

**Critical gotchas:**
- `_ANEInMemoryModelDescriptor.milText` must be `NSData*` (UTF-8 bytes), **NOT** `NSString*`
- Despite "in-memory" naming, compilation still writes to a temp directory
- QoS parameter has zero measurable effect on ANE frequency
- Multi-output programs require all output IOSurfaces to have the same allocation size

### 4.2 Tensor layout
- All ANE I/O uses **IOSurface** fp16 tensors in **[1, C, 1, S]** layout (channel-first, NCDHW)
- Zero-copy shared memory between CPU and ANE
- NEON ARM intrinsics for vectorized fp16 ↔ fp32 conversion (see ANEgpt `stories_io.h`)

### 4.3 Weights as BLOBFILE constants
- Weights are embedded into compiled ANE programs, not passed as inputs
- **Overwriting weight files and reloading does NOT change outputs** — recompilation is mandatory
- Weight blob format: 128-byte header + fp16 data

```
header[0]     = 0x01
header[4]     = 0x02
header[64:68] = [0xEF, 0xBE, 0xAD, 0xDE]  // magic
header[68]    = 0x01
header[72:76] = little-endian uint32 data_size
header[80:84] = little-endian uint32 128     // offset to data
// followed by fp16 weight data
```

### 4.4 Causal attention masking
- **ANE SDPA hardware ignores `attn_mask`** — verified on M4 and M5
- Must decompose: Q@K^T (ANE conv) → mask+softmax (CPU vDSP) → scores@V (ANE conv)
- Alternative: decompose entirely in MIL using explicit matmul + add + softmax ops

### 4.5 Compiler resource limits
- **~119 kernel compile limit per process** — hard ceiling enforced by runtime
- ANEgpt workaround: `exec()` restart with checkpoint/resume
- With 72 kernels/batch × 10 accumulation steps = ~1 batch per process life
- Strict ARC scopes + explicit object release to maximize budget

### 4.6 ANE hardware facts (M4 Max)

| Property | Value |
|----------|-------|
| ANE cores | 16 |
| Evaluation queue depth | 127 |
| Peak FP16 | ~15.8 TFLOPS (Apple claims 38 TOPS INT8, but hardware dequantizes to FP16) |
| On-chip SRAM | 32 MB (perf drops ~30% when working set exceeds this) |
| Idle power | Zero (hard power gating) |
| Dispatch overhead | ~0.095 ms per XPC+IOKit roundtrip |
| Optimal dispatch | Chain 16-64 ops per MIL program (single ops waste 70% capacity) |

**Key insight:** ANE is fundamentally a convolution engine. 1×1 convolutions deliver **3× better throughput** than matmul for identical computations.

### 4.7 ANE MIL constraints (discovered)
- `concat` MIL op is rejected by ANE compiler for multi-output bundling — use separate outputs via `orion_mil_program_multi()`
- Minimum tensor size enforced at eval time: `[1,4,1,4]` fails; `[1,256,1,64]` works
- 32000-channel convolutions rejected by ANE — classifier must run on CPU or be chunked

---

## 5) System architecture (end-to-end)

### 5.1 Runtime abstraction layers

Orion is a **general ANE model runtime**, not a single-model runner. GPT-2 and Stories110M are reference implementations. The architecture is organized around three abstraction layers:

#### OrionKernel

Represents a single compiled ANE kernel (one MIL program).

Responsibilities:
- MIL text generation from model parameters
- Weight blob binding (BLOBFILE references in MIL)
- Compilation via `orion_compile_mil`
- Evaluation via `orion_eval`

A kernel maps to one `OrionProgram*`. Each layer in a transformer has multiple kernels (e.g., 6 for training: fwdAttn, fwdFFN, ffnBwd, sdpaBwd1, sdpaBwd2, qkvBwd).

#### OrionModel

Represents a transformer architecture configuration.

Responsibilities:
- Layer configuration (`OrionModelConfig`: n_layer, n_head, d_model, etc.)
- Kernel graph construction (which kernels per layer, in what order)
- Weight mapping (BLOBFILE directory → per-kernel weight references)
- Tokenizer selection (GPT-2 BPE vs SentencePiece)

Reference implementations:
- `gpt2_124m.h` — GPT-2 with LayerNorm, GELU, 50257 vocab
- `stories110m.h` — Llama2-style with RMSNorm, SwiGLU, 32000 vocab

#### OrionRuntime

Represents the execution engine that schedules kernels on hardware.

Responsibilities:
- Program cache management (compile once, reuse across forward passes)
- Kernel scheduling (ANE eval ordering, CPU fallback decisions)
- Training loop orchestration (forward → loss → backward → dW → Adam → recompile)
- Inference loop orchestration (forward → sample → repeat)
- Weight swapping (Mode A: recompile; Mode B: adapter-as-input)
- Resource management (compile budget tracking, exec() restart)

### 5.2 Layers

**A) Frontends**
- CLI (`orion`) for training, inference, and benchmarking
- Minimal SwiftUI app for demo (optional but recommended)

**B) Runtime core**
- Tokenizer (GPT-2 BPE + SentencePiece for Stories/Llama2 vocab)
- Weight store + converter (HF → BLOBFILE blobs)
- ANE runtime (compile + eval via private APIs)
- CPU ops (cblas/vDSP for dW + optimizer + sampling)

**C) Model execution**
- Inference:
  - **ANE full forward pass** for each token generation step
  - CPU sampling (temperature, top-p, argmax)
  - No CPU fallback for transformer computation — the entire forward pass runs on ANE
- Training:
  - Per-layer kernel suite (6 kernels per layer — see §5.4)
  - CPU dW accumulation overlapped with ANE eval via GCD async dispatch
  - Adam, gradient accumulation, checkpoint/resume, exec() restart

### 5.3 Data flow (inference)

```
Prompt → Tokenizer → Embedding lookup (CPU)
  → ANE full forward pass (all layers) → Logits
  → CPU sampling (temperature, top-p) → Next token
  → ANE full forward pass (single token) → Logits
  → Repeat until done
```

This architecture demonstrates that the **entire transformer runs on the Apple Neural Engine**. CPU handles only:
- Token embedding lookup (table indexing — not a compute bottleneck)
- Logits → token sampling (argmax / nucleus sampling)
- KV cache management (append new K,V per layer)

Why this is superior to "ANE prefill + CPU decode":
- Shows the full transformer executing on Apple's neural accelerator
- Avoids unnecessary CPU fallback for decode-phase computation
- Better demonstrates ANE capability for sustained workloads
- Achieves higher sustained throughput by keeping ANE occupied

### 5.4 Data flow (training)

ANEgpt demonstrates the full pipeline:

```
For each training step:
  1) Prepare input tokens + targets (CPU)
  2) Forward kernels per layer (ANE)
     - fwdAttn: RMSNorm → QKV → SDPA → Wo
     - fwdFFN:  RMSNorm → SwiGLU (W1, W3, SiLU, W2)
  3) CPU: NLL loss + gradient (requires gather — not in MIL)
  4) Backward kernels per layer (ANE, dx path)
     - ffnBwd:   W2^T → SiLU_bwd → W1^T, W3^T
     - sdpaBwd1: Wo^T → SDPA bwd part 1 (dV, attn probs, dp)
     - sdpaBwd2: softmax grad → dQ, dK (weight-free)
     - qkvBwd:   Wq^T + Wk^T + Wv^T → dx
  5) CPU dW (async cblas_sgemm via GCD) + Adam update
  6) Recompile weight-bearing kernels with updated weights
  7) Checkpoint periodically
  8) exec() restart when approaching compile limit
```

### 5.5 ANE kernel suite (per layer)

From ANEgpt — 6 kernel types per layer, 12 layers = 72 total (60 weight-bearing, 12 weight-free):

| Kernel | Direction | Weight-bearing | Description |
|--------|-----------|----------------|-------------|
| `fwdAttn` | Forward | Yes | RMSNorm → QKV projection → SDPA → output Wo |
| `fwdFFN` | Forward | Yes | RMSNorm → SwiGLU (W1, W3, SiLU, W2) |
| `ffnBwd` | Backward | Yes | W2^T → SiLU_bwd → W1^T, W3^T |
| `sdpaBwd1` | Backward | Yes | Wo^T → SDPA backward (dV, attn probs) |
| `sdpaBwd2` | Backward | No | Softmax grad → dQ, dK |
| `qkvBwd` | Backward | Yes | Wq^T + Wk^T + Wv^T → dx |

**Forward taps:** Q, K, V, attention scores are exposed via multi-output IOSurfaces so backward pass can reuse them without CPU recomputation.

### 5.6 CPU/ANE division of labor

| Component | Location | Reason |
|-----------|----------|--------|
| Transformer forward/backward (dx) | ANE | Compute-bound, fits convolution engine |
| Token sampling (inference) | CPU | Sequential, branching logic |
| Adam optimizer | CPU | Weights are baked constants, cannot mutate on-chip |
| dW gradient accumulation | CPU (cblas_sgemm, GCD async) | Weight updates require recompilation |
| NLL loss + gradient | CPU | Requires `gather` (per-position token indexing, not in MIL) |
| Classifier backward | CPU | ANE rejects 32000-input-channel convolutions |

### 5.7 ANEgpt `train_large_ane` additional offloads

These operations can optionally be moved to ANE for speedup:

| Operation | CPU time | ANE time | Speedup |
|-----------|----------|----------|---------|
| Classifier forward (embed @ x) | 10.77ms | 1.06ms | **10.2×** |
| Softmax over vocab=32000 | 81.11ms | 2.40ms | **33.8×** |
| RMSNorm backward | 0.18ms | 0.21ms | ~1× |
| Total per step | ~106ms | ~93ms | ~1.14× |

This increases kernels from 72 to 86 per compile.

---

## 6) Runtime weight swapping (explicit goals)

### 6.1 Mode A (required): fast recompile + program cache

Because weights are baked into programs, "swapping" means "compile new programs with new weight blobs."

**Verified constraint:** Overwriting weight blob files on disk and calling unload+reload produces identical outputs — the E5 binary embeds weights at compile time. Only recompilation changes behavior.

**Requirements**
- Provide an API: `OrionProgramSet *compilePrograms(weights_id, bucket, model_config)`
- Maintain a cache keyed by: `(model_config_hash, bucket, weights_id, kernel_type, layer_idx)`
- Enforce safe release of ObjC objects to avoid compiler exhaustion
- Provide a `weights_id` abstraction:
  - For training: incrementing step id / checkpoint id
  - For inference: "base", "finetune-X", etc.

**Acceptance criteria**
- Load checkpoint A → compile → eval → load checkpoint B → compile → eval without process restart, for at least 100 swaps.

### 6.2 Mode B (stretch): adapter-as-input (true runtime swap)

Goal: keep base weights baked; pass small adapters (LoRA) as IOSurface inputs and fuse:
```
Y = X @ W_base + scale * ((X @ A) @ B)
```
This enables hot-swapping without recompiling base programs.

**Acceptance criteria (stretch)**
- Demonstrate adapter swap between two LoRA sets within one process without recompilation, for at least one projection (e.g., Wq) and one layer.

---

## 7) Benchmarking harness

### 7.1 Orion Benchmark Suite

Orion includes a built-in benchmark suite for measuring ANE kernel performance, end-to-end pipeline throughput, and system-level resource usage.

#### CLI commands

```bash
# Kernel-level benchmarks (individual ANE program latency)
orion bench kernels \
  --model stories110m \
  --layers 0,5,11 \
  --iters 100

# Inference benchmarks (full forward pass throughput)
orion bench inference \
  --model gpt2_124m \
  --prompt "The quick brown fox" \
  --max_tokens 128 \
  --warmup 3

# Training benchmarks (step time breakdown)
orion bench training \
  --model stories110m \
  --steps 20 \
  --grad_accum 4

# Weight swap endurance (Mode A validation)
orion bench swap \
  --model stories110m \
  --ckpt_a ckpt_01000 \
  --ckpt_b ckpt_01100 \
  --iters 100
```

#### 7.2 Metrics collected

| Metric | Unit | Source |
|--------|------|--------|
| Tokens/sec | tok/s | Wall time / tokens generated |
| TFLOPS estimate | TFLOPS | Estimated FLOPs / wall time |
| ANE utilization | % | Time in `orion_eval` / total wall time |
| SRAM spill detection | bool | Working set size vs 32MB threshold |
| Dispatch overhead | ms | XPC+IOKit roundtrip per eval call |
| Step time breakdown | ms | Forward, backward, dW, Adam, recompile |
| Memory (RSS) | MB | `mach_task_basic_info` resident size |
| Compile time | ms | Time in `orion_compile_mil` |
| Compile budget | count | Remaining compiles before exec() restart |

#### 7.3 Benchmark-driven tuning

Benchmark results directly inform kernel optimization decisions:

| Tuning dimension | What benchmarks reveal |
|------------------|----------------------|
| Kernel fusion | Whether combining ops into one MIL program reduces dispatch overhead |
| Conv vs matmul mapping | 1×1 conv delivers 3× throughput; benchmarks verify this for each op |
| Batch sizes | Optimal micro-batch size before SRAM spill |
| MIL program size | Trade-off between op count per program and compile time |
| Sequence length bucketing | Which bucket sizes minimize padding waste |

---

## 8) Repo layout (required)

```
orion/
  CLAUDE.md                          # Project instructions for coding agents
  ORION_v3_ANE_LLM_SPEC.md          # This spec
  README.md
  LICENSE                            # MIT
  .gitignore

  core/
    ane_runtime.{h,m}               # Private API wrapper: compile MIL, eval, IOSurface
    ane_program_cache.{h,m}         # (weights_id, bucket, layer, kernel) → compiled program
    mil_builder.{h,m}               # MIL text generator helpers
    iosurface_tensor.{h,m}          # fp16 tensors, layout conversions, NEON fp16↔fp32
    profiler.{h,m}                  # Warmup, p50/p90, tokens/sec, TFLOPS, memory
    checkpoint.{h,m}                # Training checkpoint save/load (OrionCkptHdr format)
    bucket.{h,m}                    # Sequence length bucket selection

  model/
    configs/
      gpt2_124m.h                   # GPT-2 124M model config (OrionModel reference impl)
      stories110m.h                 # Stories110M config (OrionModel reference impl)
    weights/
      download.md                   # How to obtain weights (no blobs committed)
    convert/
      hf_to_blobs_gpt2.py          # HuggingFace GPT-2 → BLOBFILE format
      hf_to_blobs_llama.py          # HuggingFace Llama2/Stories → BLOBFILE format
    blobs/
      .gitkeep                      # Generated blobs go here (gitignored)
    weight_loader.{h,m}            # BLOBFILE reader + weight struct population

  kernels/
    inference/
      gpt2_prefill_attn.milgen.{h,m}  # MIL generator: GPT-2 attention forward
      gpt2_prefill_ffn.milgen.{h,m}   # MIL generator: GPT-2 FFN forward
      gpt2_final.milgen.{h,m}         # MIL generator: final LayerNorm + logits
      kv_cache.{h,m}                   # KV cache management
      decode_cpu.{h,m}                 # CPU forward pass (reference / fallback)
      prefill_ane.{h,m}               # ANE forward pass runner
    training/
      stories_train_kernels.milgen.{h,m}  # MIL generators for 6 kernel types per layer
      classifier_softmax.milgen.{h,m}     # Classifier + vocab softmax on ANE
      stories_train.{h,m}                 # Training orchestrator (OrionTrainer)
      stories_cpu_ops.{h,m}               # CPU ops: RMSNorm, XEnt, embedding, Adam, dW
      data_loader.{h,m}                   # Pretokenized data loader (uint16 binary)

  tokenizer/
    gpt2_bpe.{h,m}                 # GPT-2 BPE tokenizer
    sentencepiece_wrap.{h,m}       # SentencePiece wrapper for Llama2/Stories vocab
    data/                           # Tokenizer vocabulary files

  apps/
    cli/
      main.m                       # CLI entry point + command dispatch
      commands/
        infer.m                    # `orion infer` command
        train.m                    # `orion train` command
        bench.m                    # `orion bench` command (kernels, inference, training, swap)
    macapp/
      OrionApp.swift               # SwiftUI app entry point
      ModelRunner.swift            # Bridge between UI and Orion runtime

  scripts/
    download_data.sh               # Download TinyStories pretokenized data

  tests/
    test_ane_runtime.m             # ANE compile/eval/release tests
    test_mil_builder.m             # MIL generation tests
    test_cpu_forward.m             # CPU forward pass tests
    test_tokenizer.m               # GPT-2 BPE golden tests (20 prompts)
    test_sp_tokenizer.m            # SentencePiece tokenizer tests
    test_decode.m                  # KV cache + decode tests
    test_infer_golden.m            # Inference golden vectors
    test_ane_prefill.m             # ANE forward pass tests
    test_cpu_training_ops.m        # CPU training op tests
    test_data_loader.m             # Data loader tests
    test_train_kernels.m           # ANE training kernel tests
    test_train_smoke.m             # E2E training smoke test
    test_program_swap.m            # 100-iteration swap endurance test

  docs/
    ane_api_reference.md           # Private ANE API documentation
```

---

## 9) Public API contracts (coding agent must implement)

### 9.1 ANE runtime

```c
typedef struct {
    int n_layer, n_head, d_model, head_dim, hidden_dim, vocab;
    int max_seq;
} OrionModelConfig;

typedef struct OrionProgram OrionProgram;

OrionProgram* orion_compile_mil(
    const char* mil_text,
    const void* const* weight_blobs, const size_t* weight_sizes, int num_blobs,
    const OrionModelConfig* cfg,
    const char* program_tag  // for debugging + caching
);

void orion_eval(
    OrionProgram* prog,
    IOSurfaceRef* inputs, int num_inputs,
    IOSurfaceRef* outputs, int num_outputs
);

void orion_release_program(OrionProgram* prog);
int orion_compile_count(void);
```

### 9.2 Program cache + weight swapping

```c
typedef struct {
    const char* weights_id; // e.g., "ckpt_00012000"
    int bucket;
} OrionWeightsBinding;

OrionProgram* orion_get_or_compile(
    const char* kernel_name,
    int layer_idx,
    const OrionWeightsBinding* wb,
    const OrionModelConfig* cfg
);

void orion_cache_evict(const char* weights_id);
void orion_cache_clear(void);
```

### 9.3 Inference runner

```c
typedef struct {
    OrionModelConfig cfg;
    void* kv_cache;
} OrionInferState;

void orion_infer_init(OrionInferState* st, OrionModelConfig cfg, int max_new_tokens);

// ANE full forward pass — runs entire transformer on ANE, returns logits
int  orion_forward_ane(OrionInferState* st, const int* tokens, int n_tokens);

// CPU sampling — selects next token from logits
int  orion_sample_cpu(OrionInferState* st, float temperature, float top_p, uint64_t seed);
```

### 9.4 Training runner

```c
typedef struct {
    OrionModelConfig cfg;
    const char* dataset_path;
    int batch_size;
    int grad_accum_steps;
    float lr;
    int checkpoint_every;
} OrionTrainConfig;

int orion_train_run(const OrionTrainConfig* tc);
```

### 9.5 Checkpoint format

Matching ANEgpt's `CkptHdr` for compatibility:

```c
typedef struct {
    int magic;           // 0x424C5A54 "BLZT"
    int version;         // 2
    int step, total_steps;
    int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches, adam_t;
} OrionCkptHdr;
```

---

## 10) CLI requirements

### 10.1 Inference

```bash
orion infer \
  --model gpt2_124m \
  --weights base \
  --prompt "..." \
  --max_new_tokens 128 \
  --temperature 0.8 \
  --top_p 0.9 \
  --seed 42 \
  --metrics
```

### 10.2 Training

```bash
orion train \
  --model stories110m \
  --dataset tinystories \
  --seq 256 \
  --batch 4 \
  --grad_accum 8 \
  --lr 3e-4 \
  --steps 20000 \
  --checkpoint_every 500 \
  --resume               # resume from latest checkpoint
```

### 10.3 Benchmarks

```bash
# Kernel-level profiling
orion bench kernels --model stories110m --iters 100

# Inference throughput
orion bench inference --model gpt2_124m --prompt "Hello" --max_tokens 64

# Training step breakdown
orion bench training --model stories110m --steps 10

# Weight swap endurance
orion bench swap \
  --model stories110m \
  --ckpt_a ckpt_01000 \
  --ckpt_b ckpt_01100 \
  --iters 100
```

---

## 11) Correctness & test plan (non-negotiable)

### 11.1 Tokenizer golden tests
- GPT-2 BPE: encode/decode must match reference vectors for 20 prompts.
- SentencePiece: encode/decode must match Python reference for Llama2 vocabulary.

### 11.2 Inference golden tests
- Fixed prompts + seed → expected first N tokens (store in `tests/test_infer_golden.json`).
- Tolerances:
  - Primary: token-exact match for deterministic path
  - Secondary: top-k contains expected token for fp16 drift

### 11.3 Training smoke tests
- Run 50 steps on tiny dataset shard and assert:
  - Loss decreases vs step 0
  - Checkpoints load and continue (including across exec() restart)
  - No compiler exhaustion / crash

### 11.4 Program swap endurance test
- Alternate two checkpoints 100 times:
  - Compile + eval
  - Verify process stays alive and memory does not grow unbounded

### 11.5 Benchmark regression tests
- Record baseline metrics for each bench subcommand
- Alert when TFLOPS drops >10% or step time increases >15% vs baseline

---

## 12) Performance & metrics to print

At minimum print:
- Inference: tokens/sec (p50/p90 per token), ANE forward ms
- ANE utilization proxy: time in `orion_eval` vs total wall time
- TFLOPS estimate (based on op counts / wall time)
- Memory: resident set size
- Training: ms/step, compile time %, recompile time %, loss, TFLOPS
- Compile budget: remaining compiles before exec() restart

**Reference benchmarks (maderix/ANE on M4):**
- 9.3 ms/step, 1.78 TFLOPS sustained, 11.2% ANE utilization
- 6 ANE kernel dispatches per training step

---

## 13) Milestones (agent execution order)

### M0 — Build + run upstream reference
- Clone ANEgpt and confirm `train_large` builds and runs on macOS 15 / M4 Max.
- Extract and understand minimal ANE runtime patterns.
- Reproduce "hello MIL" — compile a trivial MIL program and eval on ANE.
- **Exit criteria:** `train_large` completes 10+ steps, loss printed.

### M1 — CPU baseline inference
- Implement CPU GPT-2 forward + KV decode.
- Implement GPT-2 BPE tokenizer.
- Pass tokenizer + inference golden tests.
- **Exit criteria:** `orion infer --model gpt2_124m` generates coherent text on CPU.

### M2 — ANE inference for GPT-2
- Implement bucket routing (32/64/128/256/512/1024).
- Generate MIL for full forward pass kernels (attention + FFN).
- Compile forward kernels and run end-to-end ANE inference with CPU sampling.
- **Exit criteria:** Inference golden tests pass with ANE forward pass; measurable speedup over CPU-only.

### M3 — Training integration (Stories110M)
- Port/adapt kernel suite (6 per layer) from ANEgpt.
- Implement checkpoint/resume with exec() restart.
- Implement GCD-async dW overlap.
- **Exit criteria:** `orion train` runs 50+ steps, loss decreases, checkpoint resumes correctly.

### M4 — Weight swapping (Mode A)
- Implement `weights_id` compilation and program cache.
- Implement swap endurance test.
- **Exit criteria:** Alternate two checkpoints 100× without crash; memory stable.

### M5 — Benchmark suite + demo app
- Implement `orion bench kernels`, `orion bench inference`, `orion bench training`.
- SwiftUI app: prompt → generate; show metrics overlay.
- Record demo video + include benchmarks table in README.
- **Exit criteria:** All bench subcommands produce metrics; app launches and generates text.

### M6 (stretch) — Adapter-as-input (Mode B)
- Implement LoRA injection for one projection in one layer, with hot swap.
- **Exit criteria:** Two LoRA adapters swapped within one process without recompilation.

---

## 14) Risks & mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Private API breaks on macOS update | Build fails | Pin macOS version; document known-good build (macOS 15.x) |
| Compiler ~119 load limit per process | Training halts | exec() restart + checkpoint/resume (proven in ANEgpt) |
| Compiler memory leakage | OOM during long runs | Strict ARC scopes; explicit object release; program cache size limits |
| SDPA ignores causal masks | Wrong attention | Explicit decomposition: Q@K^T → mask+softmax → scores@V |
| ANE rejects concat MIL op | Multi-output fails | Use `orion_mil_program_multi()` for separate output IOSurfaces |
| Multi-output buffer size mismatch | Eval status 0x1d | Pad all output IOSurfaces to uniform allocation size per kernel |
| 32000-channel conv rejected by ANE | Classifier on CPU | Accept CPU fallback (or use chunked approach) |
| FP16 numerical drift | Token mismatch vs reference | Two-tier golden tests: exact + top-k fallback |
| SRAM spill (>32MB working set) | 30% perf drop | Keep per-kernel working set under 32MB; profile with `sram_bench` |
| Single-op dispatch overhead (~0.1ms) | Low utilization | Chain 16-64 ops per MIL program |

---

## 15) Project Evolution Roadmap

This section describes how Orion evolves beyond the initial runtime in three stages.

### Stage 1 — Orion Core

**Goal:** A stable end-to-end ANE LLM runtime.

**Capabilities:**
- ANEgpt-style training engine (forward/backward on ANE, dW/Adam on CPU)
- Full inference pipeline on ANE with CPU sampling
- Checkpoint save/load with weight recompilation
- CLI interface (`orion train`, `orion infer`, `orion bench`)
- Benchmark suite with kernel-level and pipeline-level profiling

**Deliverable:** A developer can run:

```bash
orion train --model stories110m --dataset tinystories --steps 1000
orion infer --model gpt2_124m --prompt "Hello" --max_tokens 64
orion bench kernels --model stories110m --iters 100
```

**Definition of done:** All milestones M0–M5 complete. All tests pass. Benchmark table in README.

### Stage 2 — Orion Compiler

**Goal:** Automated ANE graph optimization.

The Orion Compiler generates, compiles, profiles, and selects the fastest MIL graph variant for each kernel. This is an auto-tuning system for ANE kernels, analogous to CUDA autotuners or Triton's compilation model.

**Architecture:**

```
OrionKernel definition
  → Generate N MIL graph variants (different fusion, layout, conv/matmul choices)
  → Compile all variants via orion_compile_mil
  → Profile each variant (latency, SRAM usage, throughput)
  → Select fastest variant
  → Cache selected program
```

**Example transformations explored:**
- **Operator fusion:** Merge LayerNorm + linear into one MIL program vs separate programs
- **Layout changes:** Channel-first vs channel-last for intermediate tensors
- **Conv vs matmul mapping:** Test whether 1×1 conv or explicit matmul is faster for each weight shape
- **Sequence chunking:** Split long sequences into chunks that fit in SRAM

**CLI interface:**

```bash
# Auto-tune attention kernel for Stories110M layer 0
orion optimize attention --model stories110m --layer 0 --variants 8

# Auto-tune all kernels (full sweep)
orion optimize all --model stories110m --output tuned_kernels/
```

**Output:** A `tuned_kernels/` directory containing the fastest MIL variant for each (model, layer, kernel_type) combination, plus a JSON manifest with profiling results.

**Definition of done:** At least 3 transformation axes explored. Measurable speedup on at least one kernel vs hand-tuned baseline.

### Stage 3 — Orion Platform

**Goal:** Turn Orion into a developer platform for on-device AI.

**Capabilities:**
- Build small transformer models optimized for ANE from a model definition
- Convert PyTorch model weights to Orion runtime format (BLOBFILE)
- Generate optimized MIL graphs automatically using the Stage 2 compiler
- Export Swift bindings for embedding Orion models in macOS applications

**CLI interface:**

```bash
# Build a complete deployment package from a model definition
orion build --model tinygpt --weights trained_weights/ --output dist/

# Convert PyTorch checkpoint to Orion format
orion convert --source model.pt --arch gpt2 --output model/blobs/custom/

# Generate Swift package for embedding
orion export-swift --model tinygpt --output OrionModel.swift
```

**`orion build` produces:**
- Trained weights in BLOBFILE format
- Pre-compiled, auto-tuned ANE kernels for target hardware
- A deployment package (directory or archive) ready for integration

**Definition of done:** End-to-end pipeline from PyTorch weights → optimized ANE runtime → working inference in a Swift app. At least one model besides GPT-2/Stories110M successfully converted and running.

---

## 16) Definition of Done (v3)

- [ ] `orion infer` works offline and generates coherent text (GPT-2) with ANE full forward pass.
- [ ] `orion train` runs Stories110M training steps and saves checkpoints on device.
- [ ] Weight swapping Mode A works (load two checkpoints repeatedly, 100× endurance).
- [ ] All golden tests pass (tokenizer, inference, training smoke, swap endurance).
- [ ] `orion bench` produces kernel-level, inference, and training metrics.
- [ ] README includes benchmark table + demo instructions.
- [ ] CLI prints performance metrics (tokens/sec, TFLOPS, memory, compile overhead).
