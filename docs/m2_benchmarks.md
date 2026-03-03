# M2 Benchmarks — ANE Prefill vs CPU

**Hardware**: Mac Studio M4 Max 64GB
**Model**: GPT-2 124M (12 layers, 768 dims, 12 heads)
**Date**: 2026-03-03

## Prefill Latency

| Backend | Prompt 4tok | ms/token | Notes |
|---------|-------------|----------|-------|
| CPU     | 13.5 ms     | 3.4 ms   | cblas_sgemm, single-threaded |
| ANE     | 1399.3 ms   | 349.8 ms | **Includes 25× MIL compilation** |

## Why ANE is Slower (For Now)

ANE prefill currently compiles 25 MIL programs on every call:
- 12 attention kernels × ~55 ms compile each
- 12 FFN kernels × ~55 ms compile each
- 1 final LayerNorm kernel

Compilation accounts for ~83% of wall time (validated in M0 upstream benchmarks).

## Expected After Program Cache (M4)

With `orion_program_cache` (T084), programs are compiled once per bucket size and reused:
- First call: ~1400 ms (compile + eval)
- Subsequent calls: ~50-100 ms (eval only, estimated)
- ANE eval time for 25 programs ≈ 25 × 0.2ms = ~5 ms (from M0 T008 measurements)

## Numerical Accuracy

| Metric | Value |
|--------|-------|
| K projection max error | 0.0025 (layer 0, pos 0) |
| Top-1 argmax match | 100% (3/3 golden vectors) |
| Top-5 overlap | 5/5 (perfect) |
| 5-token greedy match | Exact (Hello → ", I'm sorry,") |

## Decode Performance (CPU, unchanged)

| Metric | Value |
|--------|-------|
| Decode p50 | 3.5 ms/tok |
| Decode p90 | 3.7 ms/tok |
| Throughput | 268-283 tok/s |
