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
- **Custom kernel pipelines** — hand-tuned MIL generators instead of opaque graph optimization
- **Budget-aware compilation** — track and manage the ~119 compile limit per process

This is the first open-source project to demonstrate **LLM training on Apple's Neural Engine**.

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
./orion train --weights model/blobs/stories110m \
  --dataset data/tinystories.bin \
  --steps 20000 --grad_accum 10 \
  --checkpoint_every 1000 --lr 3e-4
```

Training runs: ANE forward/backward → CPU weight updates → Adam optimizer → checkpoint → recompile ANE programs with updated weights. Resume from any checkpoint.

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
           │          Kernel Layer                  │
           │  MIL generators → MIL text             │
           │  Weight dict builders → BLOBFILE refs   │
           │  Program cache (store/lookup/evict)     │
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

### Abstraction layers (Phase 10, in progress)

| Layer | Role | Status |
|-------|------|--------|
| `OrionKernel` | Wraps MIL generation + weight dict + cache + compile + eval into one call | Planned |
| `OrionModelConfig` | Model architecture description (layers, heads, dims, vocab) | Done |
| `OrionRuntime` | ANE init + compile budget tracking + kernel dispatch | Planned |

These abstractions are what make Orion a **general ANE model runtime** rather than a single-model demo. Adding a new model means implementing its `OrionKernel` definitions and weight layout — the compile-cache-eval machinery is shared.

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

### Stage 1 — Orion Core *(finishing)*

ANE training runtime + inference engine. Direct MIL compilation, program caching, weight swapping, benchmark harness. Two models: GPT-2 124M (inference) and Stories110M (training). Runtime abstractions (`OrionKernel`, `OrionRuntime`, model registry) in progress — these make Orion a general toolkit rather than a single-model demo.

**Status**: 100/116 tasks complete. 11 remaining (runtime abstractions + build quality).

### Stage 2 — Orion Compiler *(upcoming)*

Automatic optimization of MIL graphs for ANE. The goal: hand-tuned ANE performance without hand-written MIL generators.

Planned capabilities:
- **Kernel fusion** — merge adjacent ops into single ANE programs to reduce dispatch overhead
- **Operator scheduling** — optimal ordering of MIL ops for ANE's convolution engine
- **SRAM tiling** — keep working sets under 32MB to avoid spill-to-DRAM performance collapse
- **Auto-bucketing** — automatically select optimal sequence length buckets per model
- **Graph profiling** — identify bottlenecks and generate optimization reports

This is what turns Orion from "a runtime that runs models" into "a compiler that makes models fast on ANE."

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
| **MIL generation** | Hardcoded | Hardcoded | Parameterized generators |

---

## Requirements

- macOS 15+ (Sequoia) on Apple Silicon (M1 or later)
- Xcode Command Line Tools
- Python 3.10+ with `torch`, `transformers` (weight conversion only — not needed at runtime)

## Building

```bash
# Build with Makefile (30 source files, no Xcode project required)
make            # Build the orion binary
make test       # Build and run all 17 test suites
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

108 of 116 tasks complete across 11 phases. See [STATUS.md](STATUS.md) for the full dashboard.

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

## Acknowledgements

Built on research from:
- [maderix/ANE](https://github.com/maderix/ANE) — ANE private API reverse-engineering (MIT)
- [ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt) — LLM training harness on ANE (MIT)
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — ANE community documentation

## License

MIT — Murai Labs
