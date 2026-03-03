# Orion — Project Status Dashboard

> **Purpose**: Orchestrator's view of overall project health.
> Tracks milestone progress, task completion, decisions, and risks.
> This is NOT the handoff document — see `CHECKPOINT.md` for cross-session handoff.

## Current Phase
**M1 — CPU Baseline Inference (in progress)** — M0 complete. Core runtime (IOSurface tensors + ANE compile/eval/release) implemented and tested (11/11 pass). Next: MIL builder helpers (T019-T023), weight format (T024-T027), tokenizer (T028-T030).

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
| Stories weight converter | T027 | M | Pending |
| GPT-2 BPE tokenizer | T028 | L | Pending |
| Tokenizer golden vectors | T029 | M | Pending |
| Tokenizer golden test runner | T030 | M | Pending |
| GPT-2 weight loading | T031 | M | Pending |
| Token + positional embedding | T032 | M | Pending |
| CPU LayerNorm | T033 | M | Pending |
| CPU attention | T034 | M | Pending |
| CPU FFN | T035 | M | Pending |
| CPU GPT-2 full forward | T036 | L | Pending |
| KV cache prefill store | T037 | M | Pending |
| KV cache append | T038 | M | Pending |
| CPU single-step decode | T039 | M | Pending |
| Token sampling | T040 | M | Pending |
| CLI arg parsing (infer) | T041 | M | Pending |
| Wire orion infer E2E (CPU) | T042 | L | Pending |
| Inference golden vectors | T043 | M | Pending |
| Profiler core | T044 | M | Pending |
| Profiler print | T045 | S | Pending |
| Wire profiler to CLI | T046 | S | Pending |

### M2 — ANE Prefill Inference
| Task | ID | Size | Status |
|------|----|------|--------|
| MIL GPT-2 attn prefill | T047 | L | Pending |
| MIL GPT-2 FFN prefill | T048 | L | Pending |
| MIL final LN + logits | T049 | M | Pending |
| Bucket selection | T050 | S | Pending |
| Prompt padding for ANE | T051 | M | Pending |
| ANE prefill runner | T052 | XL | Pending |
| Extract K,V to cache | T053 | M | Pending |
| Wire hybrid inference | T054 | L | Pending |
| ANE golden tests | T055 | M | Pending |
| Benchmark ANE vs CPU | T056 | S | Pending |

### M3 — Training (Stories110M)
| Task | ID | Size | Status |
|------|----|------|--------|
| CPU RMSNorm | T057 | M | Pending |
| CPU cross-entropy | T058 | M | Pending |
| CPU embedding | T059 | S | Pending |
| CPU Adam | T060 | M | Pending |
| CPU dW accum (cblas) | T061 | M | Pending |
| SentencePiece tokenizer | T062 | M | Pending |
| Data loader | T063 | M | Pending |
| MIL fwdAttn kernel | T064 | L | Pending |
| MIL fwdFFN kernel | T065 | L | Pending |
| MIL ffnBwd kernel | T066 | L | Pending |
| MIL sdpaBwd1 kernel | T067 | L | Pending |
| MIL sdpaBwd2 kernel | T068 | M | Pending |
| MIL qkvBwd kernel | T069 | L | Pending |
| MIL classifier fwd | T070 | M | Pending |
| MIL vocab softmax | T071 | M | Pending |
| Single training step | T072 | XL | Pending |
| GCD async dW overlap | T073 | M | Pending |
| Gradient accumulation | T074 | M | Pending |
| Checkpoint save | T075 | L | Pending |
| Checkpoint resume | T076 | L | Pending |
| exec() restart | T077 | L | Pending |
| Recompile after Adam | T078 | L | Pending |
| CLI arg parsing (train) | T079 | M | Pending |
| Wire orion train E2E | T080 | L | Pending |
| Training profiler | T081 | S | Pending |
| Training smoke test | T082 | M | Pending |
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
- **M1**: 15/35 complete
- **M2**: 0/10 complete
- **M3**: 0/27 complete
- **M4**: 0/6 complete
- **M5**: 0/6 complete
- **M6**: 0/3 complete (stretch)
- **Grand Total**: 26/98 complete (0 in progress)
- **Critical path**: T001 → T008 → T015 → T019 → T047 → T052 → T054

## Decisions Log
| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-03 | Objective-C for core runtime | ANE private APIs are ObjC; minimal FFI overhead |
| 2026-03-03 | Single-file clang compilation (no Xcode project for M0-M4) | Match upstream ANEgpt build pattern; Xcode project deferred to M5 |
| 2026-03-03 | GPT-2 for inference demo, Stories110M for training | GPT-2 is widely reproducible; Stories matches ANEgpt upstream |
| 2026-03-03 | Mode A (recompile) before Mode B (LoRA) | Recompile is proven in upstream; LoRA is experimental |
| 2026-03-03 | ANEgpt checkpoint format (CkptHdr) | Compatibility with upstream; proven to work |

## Blockers
- **None** — M0 complete, ready for M1

## Risks
| Risk | Impact | Status |
|------|--------|--------|
| Private API breaks on macOS update | All ANE work blocked | **Validated** — works on macOS 15 / M4 Max |
| ~119 compile limit per process | Training requires exec() restart | **Validated** — exec() restart works (T004/T007) |
| SDPA ignores causal masks | Wrong attention outputs | **Confirmed** — must decompose manually (T008 doc) |
| ANE minimum tensor size | Small tensors fail at eval | **NEW** — [1,4,1,4] fails; [1,256,1,64] works |
| GPT-2 BPE tokenizer complexity | Large task (T028) | Open — may wrap tiktoken as fallback |
| fp16 numerical drift | Golden tests may need relaxed tolerance | Open — two-tier tolerance planned |
