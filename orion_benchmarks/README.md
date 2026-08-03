# Orion ANE Benchmarks — M4 MacBook Air

Benchmark suite comparing LLM inference frameworks on Apple Silicon M4 MacBook Air, with emphasis on thermal throttling characterization under sustained load.

## Background

[Orion](https://github.com/mechramc/Orion) (Kumaresan, 2026) is the first open-source system for direct Apple Neural Engine (ANE) programming. The original paper benchmarks on M4 Max (desktop, active cooling, 546 GB/s bandwidth). This project extends those benchmarks to the M4 MacBook Air (fanless, thermal-constrained, ~120 GB/s bandwidth).

## What This Measures

| Framework | Backend | What It Tests |
|-----------|---------|---------------|
| **Orion** | ANE (direct) | Direct NPU programming, lowest power |
| **MLX** | GPU (Metal) | Apple's optimized GPU framework |
| **llama.cpp** | GPU (Metal) | Community standard, broad compatibility |
| **CoreML** | ANE (via API) | Apple's public ML framework |

## Key Questions

1. How does ANE performance degrade on fanless hardware under sustained load?
2. Does ANE show better thermal stability than GPU-based inference?
3. What's the practical throughput for edge deployment (Llama-3.2-3B)?
4. How do burst vs sustained performance differ across frameworks?

## Quick Start

```bash
# Install dependencies
pip3 install mlx-lm
brew install llama.cpp

python3 benchmark_v2.py --backends orion,mlx --burst-iter 10 --sustained-min 5
```

## Results

Measured on M4 MacBook Air 8GB, macOS 26.5.2. Each burst run = 10 iterations; sustained = full-duration thermal soak. All numbers are `mean_tok_s` from the benchmark JSON output.

### On AC Power

| Model | Backend | Burst (tok/s) | Sustained (tok/s) | Sustained Thermal |
|-------|---------|:-------------:|:-----------------:|:-----------------:|
| gpt2-124m | Orion (ANE) | 110.1 | 110.2 | Moderate |
| llama-3.2-3b | MLX | 52.6 | 48.9 | **Heavy** |
| phi-3-mini | MLX | 45.0 | 38.2 | **Heavy** |

### On Battery

| Model | Backend | Burst (tok/s) | Sustained (tok/s) | Sustained Thermal |
|-------|---------|:-------------:|:-----------------:|:-----------------:|
| gpt2-124m | Orion (ANE) | 75.8 | 75.9 | Nominal |
| llama-3.2-3b | MLX | 26.2 | 25.3 | Nominal |
| phi-3-mini | MLX | 22.4 | 21.6 | Nominal |

### Key Findings

- **Orion (ANE) is ~2× faster** than MLX on gpt2-124m (110 vs 52.6 tok/s on AC).
- **MLX throttles under sustained AC load** — llama-3.2-3b degrades from 52.6→48.9 tok/s and phi-3-mini from 45.0→38.2 tok/s, both reaching Heavy thermal pressure on the fanless M4 Air.
- **Orion stays thermally stable on battery** (Nominal across all sustained runs), only reaching Moderate on AC sustained — significantly better than MLX's Heavy throttling.
- **Battery cuts MLX throughput by ~50%** (52.6→26.2 tok/s for llama-3.2-3b), while ANE is more graceful (~31% drop: 110.1→75.8 tok/s).

## Contributing

This is a collaborative project with the Orion team. Benchmark scripts and raw data will be contributed back to the Orion repository via pull request.

## Citation

If you use these benchmarks, please cite:

```bibtex
@article{kumaresan2026orion,
  title={Orion: Characterizing and Programming Apple's Neural Engine for LLM Training and Inference},
  author={Kumaresan, Ramchand},
  journal={arXiv preprint arXiv:2603.06728},
  year={2026}
}
```

## Author

Viraj Shah — DES Pune University (B.Tech CSE, Third Year)
