# Orion — Project Status Dashboard

> **Purpose**: Orchestrator's view of overall project health.
> Tracks milestone progress, task completion, decisions, and risks.
> This is NOT the handoff document — see `CHECKPOINT.md` for cross-session handoff.

## Current Phase
**M3 — Training (IN PROGRESS)** — CPU training ops (T057-T061), tokenizer (T062), data loader (T063), ANE training kernels (T064-T071), single training step (T072) all done. Next: GCD async dW overlap (T073+).

## Milestone Progress

### M0 — Upstream Validation
| Task | ID | Size | Status |
|------|----|------|--------|
| Clone maderix/ANE | T001 | S | **DONE** |
| Clone ANEgpt | T002 | S | **DONE** |
| Build maderix/ANE train_large | T003 | M | **DONE** |
| Run train_large 10+ steps | T004 | M | **DONE** |
| Build ANEgpt binaries | T005 | M | **DONE** |
| Build libane_bridge.dylib | T006 | S | **DONE** |
| Run ANEgpt train_large | T007 | M | **DONE** |
| Hello MIL proof-of-concept | T008 | M | **DONE** |
| Document ANE API reference | T009 | S | **DONE** |
| Download TinyStories data | T010 | S | **DONE** |
| Download Stories110M weights | T011 | S | **DONE** |

### M1 — CPU Baseline Inference
| Task | ID | Size | Status |
|------|----|------|--------|
| IOSurface tensor create | T012 | M | **DONE** |
| IOSurface read/write | T013 | S | **DONE** |
| fp16↔fp32 NEON conversion | T014 | M | **DONE** |
| orion_compile_mil | T015 | L | **DONE** |
| orion_eval | T016 | M | **DONE** |
| orion_release_program | T017 | M | **DONE** |
| ANE runtime integration test | T018 | S | **DONE** |
| MIL linear helper | T019 | M | **DONE** |
| MIL layernorm/rmsnorm | T020 | M | **DONE** |
| MIL gelu/silu | T021 | S | **DONE** |
| MIL causal attention | T022 | L | **DONE** |
| MIL program wrapper | T023 | M | **DONE** |
| BLOBFILE writer | T024 | M | **DONE** |
| GPT-2 weight converter | T025 | M | **DONE** |
| Weight conversion tests | T026 | S | **DONE** |
| Stories weight converter | T027 | M | **DONE** |
| GPT-2 BPE tokenizer | T028 | L | **DONE** |
| Tokenizer golden vectors | T029 | M | **DONE** |
| Tokenizer golden test runner | T030 | M | **DONE** |
| GPT-2 weight loading | T031 | M | **DONE** |
| Token + positional embedding | T032 | M | **DONE** |
| CPU LayerNorm | T033 | M | **DONE** |
| CPU attention | T034 | M | **DONE** |
| CPU FFN | T035 | M | **DONE** |
| CPU GPT-2 full forward | T036 | L | **DONE** |
| KV cache prefill store | T037 | M | **DONE** |
| KV cache append | T038 | M | **DONE** |
| CPU single-step decode | T039 | M | **DONE** |
| Token sampling | T040 | M | **DONE** |
| CLI arg parsing (infer) | T041 | M | **DONE** |
| Wire orion infer E2E (CPU) | T042 | L | **DONE** |
| Inference golden vectors | T043 | M | **DONE** |
| Profiler core | T044 | M | **DONE** |
| Profiler print | T045 | S | **DONE** |
| Wire profiler to CLI | T046 | S | **DONE** |

### M2 — ANE Prefill Inference
| Task | ID | Size | Status |
|------|----|------|--------|
| MIL GPT-2 attn prefill | T047 | L | **DONE** |
| MIL GPT-2 FFN prefill | T048 | L | **DONE** |
| MIL final LN + logits | T049 | M | **DONE** |
| Bucket selection | T050 | S | **DONE** |
| Prompt padding for ANE | T051 | M | **DONE** |
| ANE prefill runner | T052 | XL | **DONE** |
| Extract K,V to cache | T053 | M | **DONE** |
| Wire hybrid inference | T054 | L | **DONE** |
| ANE golden tests | T055 | M | **DONE** |
| Benchmark ANE vs CPU | T056 | S | **DONE** |

### M3 — Training (Stories110M)
| Task | ID | Size | Status |
|------|----|------|--------|
| CPU RMSNorm | T057 | M | **DONE** |
| CPU cross-entropy | T058 | M | **DONE** |
| CPU embedding | T059 | S | **DONE** |
| CPU Adam | T060 | M | **DONE** |
| CPU dW accum (cblas) | T061 | M | **DONE** |
| SentencePiece tokenizer | T062 | M | **DONE** |
| Data loader | T063 | M | **DONE** |
| MIL fwdAttn kernel | T064 | L | **DONE** |
| MIL fwdFFN kernel | T065 | L | **DONE** |
| MIL ffnBwd kernel | T066 | L | **DONE** |
| MIL sdpaBwd1 kernel | T067 | L | **DONE** |
| MIL sdpaBwd2 kernel | T068 | M | **DONE** |
| MIL qkvBwd kernel | T069 | L | **DONE** |
| MIL classifier fwd | T070 | M | **DONE** |
| MIL vocab softmax | T071 | M | **DONE** |
| Single training step | T072 | XL | **DONE** |
| GCD async dW overlap | T073 | M | Pending |
| Gradient accumulation | T074 | M | Pending |
| Checkpoint save | T075 | L | Pending |
| Checkpoint resume | T076 | L | Pending |
| exec() restart | T077 | L | Pending |
| Recompile after Adam | T078 | L | Pending |
| CLI arg parsing (train) | T079 | M | Pending |
| Wire orion train E2E | T080 | L | Pending |
| Training profiler | T081 | S | Pending |
| Training smoke test | T082 | M | **DONE** |
| Checkpoint resume test | T083 | M | Pending |

### M4 — Weight Swapping
| Task | ID | Size | Status |
|------|----|------|--------|
| Program cache | T084 | L | Pending |
| Cache eviction | T085 | M | Pending |
| weights_id abstraction | T086 | M | Pending |
| CLI bench swap args | T087 | M | Pending |
| Swap endurance test | T088 | XL | Pending |
| Wire bench swap CLI | T089 | S | Pending |

### M5 — Demo App
| Task | ID | Size | Status |
|------|----|------|--------|
| Build system (Makefile) | T090 | M | Pending |
| SwiftUI content view | T091 | M | Pending |
| ModelRunner bridge | T092 | M | Pending |
| Metrics overlay | T093 | S | Pending |
| README benchmarks | T094 | M | Pending |
| Wiring audit | T095 | S | Pending |

### M6 — LoRA Stretch
| Task | ID | Size | Status |
|------|----|------|--------|
| MIL LoRA-fused linear | T096 | L | Pending |
| LoRA adapter loading | T097 | M | Pending |
| Hot-swap demo | T098 | L | Pending |

## Task Progress
- **M0**: 11/11 complete (ALL DONE)
- **M1**: 35/35 complete (ALL DONE)
- **M2**: 10/10 complete (ALL DONE)
- **M3**: 17/27 complete
- **M4**: 0/6 complete
- **M5**: 0/6 complete
- **M6**: 0/3 complete (stretch)
- **Grand Total**: 73/98 complete (0 in progress)
- **Critical path**: T001 → T008 → T015 → T019 → T047 → T052 → T054

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-03 | Objective-C for core runtime | ANE private APIs are ObjC; minimal FFI overhead |
| 2026-03-03 | Single-file clang compilation (no Xcode project for M0-M4) | Match upstream ANEgpt build pattern; Xcode project deferred to M5 |
| 2026-03-03 | GPT-2 for inference demo, Stories110M for training | GPT-2 is widely reproducible; Stories matches ANEgpt upstream |
| 2026-03-03 | Mode A (recompile) before Mode B (LoRA) | Recompile is proven in upstream; LoRA is experimental |
| 2026-03-03 | ANEgpt checkpoint format (CkptHdr) | Compatibility with upstream; proven to work |
| 2026-03-03 | Multi-output instead of concat for training kernels | ANE compiler rejects concat along axis=1; multi-output via orion_mil_program_multi works |
| 2026-03-03 | Uniform output IOSurface sizes for multi-output ANE programs | ANE eval fails with status 0x1d if output buffers differ in size; pad all to max |

## Blockers
- **None** — M0 complete, ready for M1

## Risks
| Risk | Impact | Status |
|------|--------|--------|
| Private API breaks on macOS update | All ANE work blocked | **Validated** — works on macOS 15 / M4 Max |
| ~119 compile limit per process | Training requires exec() restart | **Validated** — exec() restart works (T004/T007) |
| SDPA ignores causal masks | Wrong attention outputs | **Confirmed** — must decompose manually (T008 doc) |
| ANE minimum tensor size | Small tensors fail at eval | **NEW** — [1,4,1,4] fails; [1,256,1,64] works |
| ANE multi-output buffer sizes | Mixed-size outputs fail eval | **SOLVED** — pad all output IOSurfaces to max size |
| GPT-2 BPE tokenizer complexity | Large task (T028) | Open — may wrap tiktoken as fallback |
| fp16 numerical drift | Golden tests may need relaxed tolerance | Open — two-tier tolerance planned |
