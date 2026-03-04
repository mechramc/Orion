# Orion

**A local AI runtime that enables training and running small LLMs directly on Apple Silicon using the Neural Engine.**

No CoreML. No Metal. No GPU. No cloud.

```
┌─────────────────────────────────────────────────────────────┐
│  Prompt → Tokenizer → ANE Runtime → CPU Sampling → Output  │
│                                                             │
│  "The meaning of life is"  →  170+ tok/s on M4 Max         │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Build (no Xcode required)
make

# Run inference
./orion infer --prompt "The meaning of life is" --max_tokens 128 --ane

# Train a model
./orion train --weights model/blobs/stories110m --dataset data/tinystories.bin --steps 1000
```

Everything runs offline. No data leaves your device.

---

## Why ANE?

Apple ships a dedicated Neural Engine (NPU) in every iPhone, iPad, and Mac sold since 2020 — over **2 billion devices**. Each has a dedicated ML accelerator capable of **15+ TFLOPS**, but Apple only exposes it through CoreML, which:

- adds significant compilation and dispatch overhead
- restricts operations to Apple-approved model formats
- does not support training at all

Orion bypasses CoreML entirely. It talks to the ANE through private frameworks (`_ANEClient`, `_ANECompiler`, MIL IR), giving full control over compilation, memory layout, and scheduling. This opens up capabilities Apple never intended:

- **Training on the ANE** — forward and backward passes on hardware Apple restricted to inference
- **Direct program caching** — avoid recompilation overhead that plagues CoreML
- **Custom kernel pipelines** — compiler-generated MIL from graph IR with optimization passes
- **Budget-aware compilation** — track and manage the ~119 compile limit per process

Orion builds on prior work (ANEgpt, maderix/ANE) and extends it into a **production-quality LLM training and inference runtime** with a compiler, program caching, checkpointing, and benchmark harness.

---

## How Orion Compares

| Framework | Focus | Training | ANE Direct | Models |
|-----------|-------|----------|------------|--------|
| [MLX](https://github.com/ml-explore/mlx) | GPU inference + training | Yes (GPU) | No | Many |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm) | Portable inference | No | No | Many |
| [Ollama](https://ollama.com) | Easy model serving | No | No | Many |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | Portable CPU/GPU inference | No | No | Many |
| **Orion** | **ANE training + inference** | **Yes (ANE)** | **Yes** | GPT-2 124M, Stories110M |

**Orion's niche**: the only framework that targets Apple's Neural Engine directly for both training and inference. Others use GPU (Metal) or CPU. Orion uses the NPU — dedicated silicon that's idle in most workloads.

---

## What You Can Do

### Inference (3 modes)

```bash
./orion infer --prompt "Hello, world" --ane            # Full ANE forward pass
./orion infer --prompt "Hello, world" --ane-prefill     # ANE prefill → CPU decode
./orion infer --prompt "Hello, world"                   # CPU-only baseline
```

### Training

```bash
# Start training (1 step per process due to ANE compile budget)
./orion train --weights model/blobs/stories110m \
  --dataset data/tinystories.bin \
  --steps 100 --grad_accum 4 --lr 1e-5

# Resume from checkpoint
./orion train --weights model/blobs/stories110m \
  --dataset data/tinystories.bin \
  --steps 100 --grad_accum 4 --lr 1e-5 \
  --resume checkpoints/step_50.bin
```

Training runs: ANE forward/backward → CPU weight updates → Adam optimizer → checkpoint → `exec()` restart with updated weights. Each process compiles ANE programs once (72 programs per step), then restarts to stay within the ~119 compile budget. Resume from any checkpoint is verified stable (5-step chain, 0 NaN).

### Benchmarking

```bash
./orion bench kernels                        # Per-kernel ANE latency (5 kernel types)
./orion bench inference --prompt "Hello"     # End-to-end tok/s, TFLOPS, ANE utilization
./orion bench training --steps 10            # Training step breakdown (fwd/bwd/dW/Adam)
./orion bench kernels --save-baseline        # Save baseline for regression tracking
./orion bench kernels                        # Compare against saved baseline (>15% = WARN)
```

The benchmark suite measures:
- **TFLOPS** — actual compute throughput on ANE
- **Tokens/sec** — end-to-end inference speed
- **ANE utilization** — time in `orion_eval` vs total wall time
- **Dispatch overhead** — compile + eval scheduling cost
- **Memory** — RSS growth across compiles and swaps

This turns Orion into a **measurement tool** for ANE performance, not just an inference engine.

---

## Architecture

```
                         orion CLI
                ┌──────────────────────────┐
                │   infer   train   bench   │
                └─────┬───────┬───────┬────┘
                      │       │       │
           ┌──────────┴───────┴───────┴──────────┐
           │          Model Layer                  │
           │  OrionModelConfig (GPT-2, Stories)    │
           │  Tokenizers (BPE, SentencePiece)      │
           │  Weight loading (BLOBFILE format)      │
           └──────────────┬───────────────────────┘
                          │
           ┌──────────────┴───────────────────────┐
           │          Compiler + Kernel Layer         │
           │  Graph IR → optimized MIL text           │
           │  Weight dict builders → BLOBFILE refs    │
           │  Program cache (store/lookup/evict)      │
           └──────────────┬───────────────────────┘
                          │
           ┌──────────────┴───────────────────────┐
           │          Core Runtime                  │
           │  orion_compile_mil → OrionProgram      │
           │  orion_eval (IOSurface I/O)            │
           │  orion_release_program                 │
           └──────────────┬───────────────────────┘
                          │
           ┌──────────────┴───────────────────────┐
           │    Apple Neural Engine (private APIs)  │
           │  _ANEClient → _ANECompiler             │
           │  MIL IR → ANE microcode                │
           │  IOSurface-backed fp16 [1,C,1,S]       │
           └────────────────────────────────────────┘
```

### Abstraction layers

| Layer | Role | Status |
|-------|------|--------|
| `OrionKernel` | Wraps MIL generation + weight dict + cache + compile + eval into one call | Done |
| `OrionModelConfig` | Model architecture description (layers, heads, dims, vocab) | Done |
| `OrionRuntime` | ANE init + compile budget tracking + kernel dispatch | Done |
| `OrionCompiler` | Graph IR → optimized MIL text (validate, optimize, codegen) | Done |

These abstractions make Orion a **general ANE model runtime** rather than a single-model demo. Adding a new model means implementing its compiler frontend (graph builder) and weight layout — the compile-cache-eval machinery is shared.

### Data flow

**Inference**: Tokenize → embed (CPU) → 12 transformer layers on ANE (bucketed prefill or per-token decode) → final layernorm (ANE) → logits projection (CPU, wte is 73MB — too large for ANE SRAM) → sample → repeat.

**Training**: Embed (CPU) → forward layers (ANE, 6 outputs per layer for backward reuse) → loss (CPU) → backward layers (ANE, dx path) → dW accumulation (CPU, GCD async) → Adam (CPU) → recompile ANE programs with updated weights.

**Tensor layout**: All ANE I/O uses `fp16 [1, C, 1, S]` on IOSurface-backed memory. CPU↔ANE data transfer transposes between `[seq, d_model]` and `[d_model, seq]`.

---

## Use Cases

- **Local AI assistants** — private, offline language models on Mac, iPad, or iPhone
- **Educational robotics** — a toy robot running a local LLM on an iPad, no cloud dependency
- **Offline copilots** — code completion, writing assistance, and reasoning without internet
- **Privacy-first applications** — medical, legal, or personal AI where data must never leave the device
- **Edge deployment** — any Apple device with an ANE becomes an inference/training endpoint
- **ML research** — benchmark and profile the Neural Engine with direct kernel-level control

---

## Project Roadmap

### Stage 1 — Orion Core *(complete)*

ANE training runtime + inference engine. Direct MIL compilation, program caching, weight swapping, benchmark harness. Two models: GPT-2 124M (inference) and Stories110M (training). Runtime abstractions (`OrionKernel`, `OrionRuntime`, model registry) make Orion a general toolkit rather than a single-model demo.

**Status**: Complete. 116/116 tasks.

### Stage 2 — Orion Compiler *(complete)*

Graph-level IR and optimization pipeline that replaces hand-written MIL generators. Models are now defined as compiler frontends (pure C graph builders) instead of hand-written MIL text. The compiler validates, optimizes, and codegens to ANE-compatible MIL automatically.

Capabilities:
- **Graph IR** — ~27 ops, fluent builder API, topological ordering
- **Optimization passes** — DCE, identity elimination, cast fusion, SRAM annotation, uniform output padding
- **ANE validation** — constraint checking before compilation (shape limits, op support)
- **13 compiler frontends** — 4 GPT-2 inference + 6 Stories training + 3 utility kernels
- **MIL diff tool** — structural comparison between generated MIL programs

**Status**: Complete. 32/32 compiler tasks + milgen fully replaced. 21/21 tests pass.

### Stage 3 — Orion Platform *(future)*

Developer toolchain for building local AI applications on Apple devices. Model packaging, deployment, and a simple CLI/API:

```bash
pip install orion

orion train tiny mydata.txt
orion chat
```

### Packaging Roadmap

| Phase | Distribution | Status |
|-------|-------------|--------|
| 1 | Native binary (`clang && ./orion`) | Done |
| 2 | Makefile (`make && ./orion`) | Current |
| 3 | Python wrapper (`import orion`) | Planned |
| 4 | pip distribution (`pip install orion`) | Planned |

---

## Technical Discoveries

Key findings from building on the ANE:

- ANE multi-output surfaces are ordered **alphabetically by MIL variable name**, not by return tuple order
- ANE requires minimum ~49KB IOSurface allocation — `seq_len=1` compiles but fails at eval with status `0x1d`
- ANE rejects the `concat` MIL op entirely — must use multi-output programs instead
- Multi-output programs require **all output IOSurfaces to have the same allocation size** — pad to max
- `gelu` is not a valid MIL op — decompose to tanh approximation
- BLOBFILE header is 128 bytes; the MIL offset `uint64(64)` is the chunk header offset, not the data offset
- ~119 compile limit per process before ANE stops accepting programs — managed via cache + `exec()` restart
- SDPA causal masks are ignored by the ANE — must decompose attention manually (Q@K^T → mask → softmax → @V)
- Weight dict must be `@{}` (empty dict), not `nil`, for weight-free programs
- `milText` must be `NSData*` (UTF-8 bytes), not `NSString*`
- ANE MIL requires named const refs for matmul `transpose_x`/`transpose_y` — inline bool literals (`true`/`false`) are rejected
- ANE MIL `conv` op does NOT support `bias=` parameter — bias must be a separate `add` op
- After MIL optimization passes, output variable names must reference live nodes — dead node names in return tuples cause InvalidMILProgram

---

## How Orion Differs from Upstream

Orion builds on two open-source projects that reverse-engineered Apple's ANE:

| | [maderix/ANE](https://github.com/maderix/ANE) | [ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt) | **Orion** |
|---|---|---|---|
| **What it is** | ANE private API driver | Training demo | Production LLM runtime |
| **Language** | C/ObjC | C + Python | Pure ObjC (no Python at runtime) |
| **Inference** | N/A | ANE prefill only | ANE full forward (prefill + decode) |
| **Training** | N/A | Stories110M only | Stories110M + CLI + checkpoint + resume |
| **Models** | N/A | 1 | 2 (GPT-2 124M + Stories110M) |
| **Tokenizer** | N/A | Python-side | Native BPE + SentencePiece in C |
| **Program cache** | N/A | None | Composite key cache with eviction |
| **Benchmarking** | N/A | N/A | 4-mode suite with regression tracking |
| **MIL generation** | Hardcoded | Hardcoded | Compiler (graph IR → optimized MIL) |

---

## Requirements

- macOS 15+ (Sequoia) on Apple Silicon (M1 or later)
- Xcode Command Line Tools
- Python 3.10+ with `torch`, `transformers` (weight conversion only — not needed at runtime)

## Building

```bash
# Build with Makefile (~50 source files, no Xcode project required)
make            # Build the orion binary
make test       # Build and run all 21 test suites
make bench      # Run benchmark suite
make clean      # Remove all build artifacts
```

## Weight Setup

```bash
# GPT-2 124M (inference)
python model/convert/hf_to_blobs_gpt2.py    # → model/blobs/gpt2_124m/

# Stories110M (training)
python model/convert/hf_to_blobs_llama.py    # → model/blobs/stories110m/
```

## Project Status

148 of 148 tasks complete across 13 phases. See [STATUS.md](STATUS.md) for the full dashboard and [RESULTS.md](RESULTS.md) for comprehensive benchmarks.

| Phase | Status | Highlights |
|-------|--------|------------|
| M0: Upstream Validation | Complete | Verified ANE private APIs on M4 Max |
| M1: CPU Baseline | Complete | Full GPT-2 forward pass, 283 tok/s CPU decode |
| M2: ANE Prefill | Complete | 12-layer ANE prefill, exact match vs CPU |
| M3: Training | Complete | Full Stories110M training loop, checkpointing |
| M4: Weight Swapping | Complete | Program cache, 100-iter swap endurance (RSS 1.41x) |
| Phase 8: ANE Full Forward | Complete | ANE decode at 5.8ms/tok, tokens match CPU exactly |
| Phase 9: Benchmark Harness | Complete | Per-kernel, inference, training, regression tracking |
| Phase 10: Runtime Abstractions | Complete | OrionKernel, OrionRuntime, model registry |
| Phase 11: Build & Quality | Complete | Makefile, zero warnings, constraints docs |
| Phase 12: Orion Compiler | Complete | Graph IR, 5 optimization passes, 13 frontends, MIL codegen |
| Phase 13: Milgen→Compiler | Complete | All MIL generators replaced by compiler, 21/21 tests pass |

## Acknowledgements

Built on research from:
- [maderix/ANE](https://github.com/maderix/ANE) — ANE private API reverse-engineering (MIT)
- [ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt) — LLM training harness on ANE (MIT)
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — ANE community documentation

## License

MIT — Murai Labs
