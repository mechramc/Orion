# ADR-008: Orion-Q Boundary

**Status**: Accepted  
**Date**: 2026-03-15

## Decision

This repository uses the following boundary:

- `Orion`: the model-agnostic execution core
- `Orion-Q`: the Qwen-focused subset built on top of Orion

`Orion-Q` is not a separate engine. It is the part of the Orion worktree that makes Qwen-family models runnable, testable, and trainable inside Orion.

## Orion Includes

- shared compiler and runtime
- shared model registry
- shared tokenizer and weight loading base
- CPU and ANE execution primitives

## Orion-Q Includes

- Qwen frontend and model registration
- Qwen weight loading and export path
- Qwen CPU inference path
- Qwen ANE or hybrid inference path
- Qwen LoRA training primitives
- Qwen diagnostics: smoke, probe, parity, and diff tests

## Orion-Q Excludes

- downstream training tracks
- domain-specific assets
- generated local reports and logs
- exported checkpoints and blobs
- Silver accelerator work

## Result

When sharing Orion-Q, it should be framed as:

`a Qwen-focused subset built on top of Orion`

and not as a disconnected new engine.
