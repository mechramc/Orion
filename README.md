# Orion

**Run LLMs entirely on Apple's Neural Engine. No CoreML. No Metal. No GPU. No cloud.**

Orion is the first open-source runtime that drives Apple's Neural Engine directly via private APIs (`_ANEClient`, `_ANECompiler`, MIL) to run full LLM inference and training — bypassing CoreML's overhead and restrictions entirely.

GPT-2 124M runs at **170+ tok/s** on an M4 Max with the entire forward pass on the ANE. Training a 110M-parameter model works end-to-end: ANE forward/backward, CPU weight updates, Adam optimizer, checkpointing, and automatic weight recompilation into ANE programs.

## Why This Matters

Apple Neural Engines ship in every iPhone, iPad, and Mac sold since 2020 — over 2 billion devices. Each one has a dedicated ML accelerator capable of 15+ TFLOPS, but Apple only exposes it through CoreML, which adds significant overhead and restricts what you can run.

Orion removes that restriction. By talking to the ANE directly, it opens up:

- **On-device LLM inference** without GPU contention or thermal throttling
- **On-device fine-tuning** on consumer hardware (no cloud, no data leaves the device)
- **Full control over compilation, memory, and scheduling** — things CoreML abstracts away
- **A foundation for custom ML runtimes** that target the ANE as a first-class compute backend

## What You Can Do With Orion

### Run inference
```bash
./orion infer --prompt "The meaning of life is" --max_tokens 128
./orion infer --prompt "Hello, world" --ane          # ANE full forward
./orion infer --prompt "Hello, world" --ane-prefill   # ANE prefill, CPU decode
```

### Train a model
```bash
./orion train --weights model/blobs/stories110m --dataset data/tinystories.bin \
  --steps 20000 --grad_accum 10 --checkpoint_every 1000
```

### Benchmark everything
```bash
./orion bench kernels                        # Per-kernel ANE latency (5 kernel types)
./orion bench inference --prompt "Hello"     # End-to-end throughput
./orion bench training --steps 10            # Training step breakdown
./orion bench kernels --save-baseline        # Save baseline for regression tracking
./orion bench kernels                        # Compare against baseline (>15% = WARN)
```

## How Orion Differs from Upstream Projects

Orion builds on two open-source projects that reverse-engineered Apple's ANE:

| | [maderix/ANE](https://github.com/maderix/ANE) | [ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt) | **Orion** |
|---|---|---|---|
| **What it is** | ANE private API driver | Training demo on that driver | Production LLM runtime |
| **Language** | C/ObjC | C + Python bridge | Pure ObjC (no Python at runtime) |
| **Inference** | N/A | ANE prefill only | ANE full forward (prefill + decode) |
| **Training** | N/A | Stories110M (single binary) | Stories110M with full CLI, checkpointing, resume |
| **Models** | N/A | Stories110M only | GPT-2 124M + Stories110M |
| **Tokenizer** | N/A | Python-side | Native BPE + SentencePiece in C |
| **Program cache** | N/A | None (recompiles every call) | Cache with composite key (kernel, layer, weights, bucket) |
| **Weight swap** | N/A | N/A | Recompile-on-update + cache eviction |
| **Benchmarking** | N/A | N/A | Per-kernel, inference, training, regression tracking |
| **MIL generation** | Hardcoded strings | Hardcoded strings | Parameterized generators (layer, seq, config) |
| **CLI** | Single-purpose binaries | Single binary | Unified `orion` with subcommands |

### Key technical discoveries beyond upstream

- ANE multi-output surfaces are ordered **alphabetically by MIL variable name**, not by return tuple order
- ANE requires minimum ~49KB IOSurface allocation — seq_len=1 compiles but fails at eval
- ANE rejects the `concat` MIL op entirely — must use multi-output programs
- `gelu` is not a valid MIL op — must decompose to tanh approximation
- BLOBFILE header is 128 bytes (not 64); the MIL offset `uint64(64)` is the chunk header offset
- ~119 compile limit per process before ANE stops accepting programs — managed via cache + `exec()` restart

## Architecture

```
                      orion CLI
                    ┌─────────────────────────┐
                    │  infer  train  bench     │
                    └────┬───────┬───────┬─────┘
                         │       │       │
              ┌──────────┴───────┴───────┴──────────┐
              │           Kernel Layer               │
              │  milgen generators → MIL text         │
              │  weight dict builders → BLOBFILE refs  │
              │  program cache (store/lookup/evict)    │
              └──────────────┬──────────────────────┘
                             │
              ┌──────────────┴──────────────────────┐
              │           Core Runtime               │
              │  orion_compile_mil → OrionProgram     │
              │  orion_eval (IOSurface I/O)           │
              │  orion_release_program                │
              └──────────────┬──────────────────────┘
                             │
              ┌──────────────┴──────────────────────┐
              │     Apple Neural Engine (private)     │
              │  _ANEClient  _ANECompiler             │
              │  MIL IR → ANE microcode               │
              │  IOSurface-backed fp16 tensors         │
              └─────────────────────────────────────┘
```

**Inference flow**: Tokenize → embed (CPU) → 12 transformer layers on ANE (bucketed prefill or per-token decode) → final layernorm on ANE → logits projection on CPU (wte is 73MB, too large for ANE SRAM) → sample → repeat.

**Training flow**: Embed (CPU) → forward layers (ANE) → loss (CPU) → backward layers (ANE) → dW accumulation (CPU, GCD async) → Adam (CPU) → recompile ANE programs with updated weights.

**Tensor layout**: All ANE I/O uses `fp16 [1, C, 1, S]` on IOSurface-backed memory. CPU↔ANE data transfer is a transpose between `[seq, d_model]` and `[d_model, seq]`.

## Requirements

- macOS 15+ (Sequoia) on Apple Silicon (M1 or later)
- Xcode Command Line Tools
- Python 3.10+ with `torch`, `transformers` (weight conversion only — not needed at runtime)

## Building

```bash
# Full build (26 source files)
xcrun clang -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK \
  -framework Foundation -framework IOSurface -framework Accelerate \
  -ldl -I . \
  apps/cli/main.m apps/cli/commands/*.m \
  core/*.m kernels/inference/*.m kernels/training/*.m \
  model/*.m tokenizer/*.m \
  -o orion

# Run tests
for t in tests/test_*; do [ -x "$t" ] && ./"$t"; done
```

## Weight Setup

```bash
# GPT-2 124M (inference)
python model/convert/hf_to_blobs_gpt2.py    # → model/blobs/gpt2_124m/

# Stories110M (training)
python model/convert/hf_to_blobs_llama.py    # → model/blobs/stories110m/
```

## Project Status

99 of 116 tasks complete across 9 phases. See [STATUS.md](STATUS.md) for the full dashboard.

| Phase | Status | Highlights |
|-------|--------|------------|
| M0: Upstream Validation | Complete | Verified ANE APIs work on M4 Max |
| M1: CPU Baseline | Complete | Full GPT-2 forward pass, 283 tok/s CPU decode |
| M2: ANE Prefill | Complete | 12-layer ANE prefill, exact match vs CPU |
| M3: Training | Complete | Full Stories110M training loop, checkpointing |
| M4: Weight Swapping | Complete | Program cache, 100-iter endurance test (RSS 1.41x) |
| Phase 8: ANE Full Forward | Complete | ANE decode at 5.8ms/tok, tokens match CPU exactly |
| Phase 9: Benchmark Harness | Complete | Per-kernel, inference, training, regression tracking |
| Phase 10: Runtime Abstractions | Next | OrionModel, OrionKernel, OrionRuntime |
| Phase 11: Build & Quality | Planned | Makefile, warnings cleanup |

## Acknowledgements

Built on research from:
- [maderix/ANE](https://github.com/maderix/ANE) — ANE private API reverse-engineering (MIT)
- [ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt) — LLM training harness on ANE (MIT)
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — ANE community documentation

## License

MIT — Murai Labs
