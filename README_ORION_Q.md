# Orion-Q

Orion-Q is Qwen-focused porting and diagnostics scaffolding built on top of Orion.

It is not a separate engine, and this PR does not yet expose Qwen through the user-facing `orion infer` command. The current CLI remains GPT-2-only. Orion-Q adds repository-local building blocks and verification harnesses for:

- Qwen model configs and blob conversion
- Qwen CPU reference and diagnostic inference paths
- Qwen ANE/hybrid prefill and parity probes
- Qwen LoRA training-preparation primitives
- Qwen diagnostics, including smoke, probe, parity, and diff tests

The boundary for this subset is defined in:

- `docs/orion_q/ADR-008-orion-q-boundary.md`

## What Orion-Q Is

Orion-Q is the part of the Orion worktree used to port and verify Qwen-family model components against Orion's compiler, runtime, CPU reference code, and ANE experiments.

In practical terms, Orion-Q includes:

- Qwen frontend and model registration
- Qwen manifest/blob conversion and validation
- Qwen-specific CPU reference kernels
- Qwen ANE/hybrid diagnostic paths
- Qwen LoRA training preparation
- Qwen diagnostics and validation tests

These pieces are useful for integration work, but they should not be read as a claim that the normal Orion CLI already provides end-to-end Qwen generation.

## What Orion-Q Is Not

Orion-Q does not currently include:

- end-to-end Qwen dispatch in `orion infer`
- a production Qwen CLI tokenizer/special-token integration
- a supported Qwen user-facing generation command
- Silver accelerator work
- user-specific training tracks
- CRPG or other domain assets
- reports, logs, or generated tokenizer experiment outputs
- exported checkpoints or model weights

## Current Status

Within the currently defined Orion-Q diagnostic scope:

- Qwen porting primitives and model registration: present
- CPU reference/diagnostic paths: present
- target hybrid parity smoke diagnostics: close-out achieved for the documented fixtures
- ANE training preparation line: complete for the documented candidate path
- user-facing Qwen CLI integration: not included in this PR

Supporting documents:

- `docs/orion_q/ORION_Q_PORT_CLOSEOUT.md`
- `docs/orion_q/ORION_Q_HYBRID_PARITY_CLOSEOUT.md`
- `docs/orion_q/ORION_Q_ANE_PREP_CLOSEOUT.md`

## Tests

`make test` includes the two self-contained Qwen frontend tests in addition to Orion's existing verification suite.

`make test-qwen` builds all 33 Qwen test executables, then runs the two self-contained frontend tests. The remaining Qwen programs are explicit probes that require converted model/tokenizer fixtures and/or ANE hardware and are not silently run without those prerequisites.

## Included Code Areas

The shared Orion-Q subset covers these groups:

- shared Orion core changes required by Qwen support
- `compiler/frontends/qwen35_*`
- `kernels/inference/qwen_*`
- `kernels/training/qwen_lora_*`
- `model/configs/qwen35_*`
- `model/convert/hf_to_blobs_qwen35.py`
- `tests/test_qwen35_*`
- `tests/test_qwen35_9b_*`

## Validation Philosophy

Orion-Q treats diagnostics as first-class integration evidence rather than throwaway experiments. Depending on prerequisites, those diagnostics include:

- self-contained frontend smoke tests
- manifest and attention-shape audits
- bridge-stage diffs
- layer diffs
- parity checks
- ANE training probes

Passing a diagnostic scope does not by itself establish a supported end-to-end Qwen CLI path.

## Recommended Share Mode

The recommended way to share Orion-Q is:

1. As an Orion-based fork or draft PR branch
2. With generated artifacts excluded
3. With the diagnostic/integration scope stated explicitly

Suggested framing:

`Orion-Q: Qwen porting and diagnostics scaffolding built on top of Orion`

## Excluded Artifacts

Do not publish these as part of Orion-Q:

- `tokenizer/data/orion_*`
- `tokenizer/data/*_tok`
- exported blobs and checkpoints
- local reports and logs
- user workflow assets

## Local Share Bundle

This workspace can generate a clean Orion-Q share bundle with:

```bash
python3 scripts/prepare_orion_q_share.py --clean
```

Default output:

```text
build/orion_q_share/Orion-Q
```

The bundle is driven by:

- `scripts/orion_q_share_manifest.txt`

## Relationship to Upstream Orion

Upstream Orion remains the execution core.

Orion-Q should be communicated as:

- Orion core
- plus Qwen-specific porting primitives
- plus Qwen-specific diagnostics and integration probes

It should not be presented as a replacement brand, a disconnected new engine, or a completed Qwen CLI integration.
