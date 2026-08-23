# ADR-008: Orion-Q Boundary

**Status**: Accepted  
**Date**: 2026-03-15

## Decision

This repository uses the following boundary:

- `Orion`: the model-agnostic execution core
- `Orion-Q`: Qwen-focused porting, reference kernels, and diagnostics built on top of Orion

`Orion-Q` is not a separate engine. In this PR it is an integration/verification layer, not a claim that Qwen is already wired into the user-facing `orion infer` command.

## Orion Includes

- shared compiler and runtime
- shared model registry
- shared tokenizer and weight-loading base
- CPU and ANE execution primitives

## Orion-Q Includes

- Qwen frontend and model registration
- Qwen manifest/blob conversion and validation
- Qwen CPU reference/diagnostic kernels
- Qwen ANE/hybrid prefill and parity probes
- Qwen LoRA training-preparation primitives
- Qwen diagnostics: smoke, probe, parity, and diff tests

## Orion-Q Excludes in This PR

- end-to-end Qwen dispatch from `orion infer`
- production Qwen CLI tokenizer/special-token handling
- a supported user-facing Qwen generation path
- downstream training tracks
- domain-specific assets
- generated local reports and logs
- exported checkpoints and blobs
- Silver accelerator work

## Result

When sharing Orion-Q, it should be framed as:

`Qwen porting and diagnostics scaffolding built on top of Orion`

and not as a disconnected new engine or a completed Orion Qwen CLI integration.
