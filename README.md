# Orion

**End-to-end on-device LLM on Apple Silicon — inference and training on the Apple Neural Engine.**

Orion runs GPT-2 inference and Stories110M training directly on the ANE using reverse-engineered private APIs. No CoreML, no Metal, no GPU. Fully offline.

## Features

- **ANE Inference**: Bucketed prefill on ANE + CPU autoregressive decode (GPT-2 124M)
- **ANE Training**: Forward/backward on ANE, dW + Adam on CPU (Stories110M, Llama2-style)
- **Weight Swapping**: Fast recompile with program cache (Mode A) + LoRA hot-swap (Mode B, stretch)
- **CLI + GUI**: `orion` CLI for training/inference + SwiftUI demo app

## Requirements

- macOS 15+ (Sequoia) on Apple Silicon
- Xcode Command Line Tools
- Python 3.10+ with `torch`, `transformers` (for weight conversion only)

## Quick Start

```bash
# Build
xcrun clang -O2 -fobjc-arc -framework Foundation -framework IOSurface \
  -framework CoreML -framework Accelerate -ldl -lobjc \
  -o orion apps/cli/main.m

# Inference
./orion infer --model gpt2_124m --prompt "Once upon a time"

# Training
./orion train --model stories110m --dataset tinystories --steps 1000
```

## Architecture

See [ORION_v2_ANE_LLM_SPEC.md](ORION_v2_ANE_LLM_SPEC.md) for the full technical specification.

## Acknowledgements

Built on research from:
- [maderix/ANE](https://github.com/maderix/ANE) — ANE private API reverse-engineering
- [ANEgpt](https://github.com/vipuldivyanshu92/ANEgpt) — Full LLM training harness on ANE
- [hollance/neural-engine](https://github.com/hollance/neural-engine) — ANE community documentation

## License

MIT
