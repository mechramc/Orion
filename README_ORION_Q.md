# Orion-Q

Orion-Q is a Qwen-focused porting and diagnostics subset built on top of Orion.

It is not a separate engine. It is a curated extension of Orion that adds:

- Qwen model configs and blob conversion
- Qwen CPU inference path
- Qwen ANE or hybrid inference path
- Qwen LoRA training primitives
- Qwen diagnostics, including smoke, probe, parity, and diff tests

The boundary for this subset is defined in:

- `docs/orion_q/ADR-008-orion-q-boundary.md`

## What Orion-Q Is

Orion-Q is the part of the local Orion worktree that makes Qwen-family models runnable and verifiable inside Orion.

In practical terms, Orion-Q includes:

- Qwen frontend and model registration
- Qwen weight loading and export path
- Qwen-specific CPU and ANE execution support
- Qwen LoRA training path
- Qwen diagnostics and validation tests

## What Orion-Q Is Not

Orion-Q does not include:

- Silver accelerator work
- user-specific training tracks
- CRPG or other domain assets
- reports, logs, or generated tokenizer experiment outputs
- exported checkpoints or model weights

Those belong to downstream tracks or local runtime artifacts, not to Orion-Q itself.

## Current Status

Within the currently defined Orion-Q scope:

- Qwen porting core: complete
- binary judge diagnostics: close-out achieved
- target hybrid parity smoke scope: close-out achieved
- ANE training preparation line: complete

Supporting documents:

- `docs/orion_q/ORION_Q_PORT_CLOSEOUT.md`
- `docs/orion_q/ORION_Q_HYBRID_PARITY_CLOSEOUT.md`
- `docs/orion_q/ORION_Q_ANE_PREP_CLOSEOUT.md`

## Included Code Areas

The shared Orion-Q subset is expected to cover these groups:

- shared Orion core changes required by Qwen support
- `compiler/frontends/qwen35_*`
- `kernels/inference/qwen_*`
- `kernels/training/qwen_lora_*`
- `model/configs/qwen35_*`
- `model/convert/hf_to_blobs_qwen35.py`
- `tests/test_qwen35_*`
- `tests/test_qwen35_9b_*`

## Validation Philosophy

Orion-Q treats diagnostics as part of the product surface, not as throwaway experiments.

That means the following are first-class parts of the subset:

- smoke tests
- bridge-stage diffs
- layer diffs
- parity checks
- ANE training probes

## Recommended Share Mode

The recommended way to share Orion-Q is:

1. As an Orion-based fork or draft PR branch
2. With generated artifacts excluded
3. With a narrow, explicit scope

Suggested framing:

`Orion-Q: a Qwen-focused porting and diagnostics subset built on top of Orion`

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
- plus Qwen-specific porting
- plus Qwen-specific diagnostics

It should not be presented as a replacement brand or a disconnected new project.
