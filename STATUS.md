# Orion — Project Status Dashboard

> **Purpose**: Orchestrator's view of overall project health.
> Tracks milestone progress, task completion, decisions, and risks.
> This is NOT the handoff document — see `CHECKPOINT.md` for cross-session handoff.

## Current Phase
**M4 — Weight Swapping (COMPLETE)** — All 6 tasks done. Program cache (store/lookup), eviction by weights_id, weights_id abstraction, CLI bench swap with 100-iter endurance test (RSS 1.41x, PASS). Next: Phase 8 (ANE Full Forward).

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
| GCD async dW overlap | T073 | M | **DONE** |
| Gradient accumulation | T074 | M | **DONE** |
| Checkpoint save | T075 | L | **DONE** |
| Checkpoint resume | T076 | L | **DONE** |
| exec() restart | T077 | L | **DONE** |
| Recompile after Adam | T078 | L | **DONE** |
| CLI arg parsing (train) | T079 | M | **DONE** |
| Wire orion train E2E | T080 | L | **DONE** |
| Training profiler | T081 | S | **DONE** |
| Training smoke test | T082 | M | **DONE** |
| Checkpoint resume test | T083 | M | **DONE** |

### M4 — Weight Swapping
| Task | ID | Size | Status |
|------|----|------|--------|
| Program cache | T084 | L | **DONE** |
| Cache eviction | T085 | M | **DONE** |
| weights_id abstraction | T086 | M | **DONE** |
| CLI bench swap args | T087 | M | **DONE** |
| Swap endurance test | T088 | XL | **DONE** |
| Wire bench swap CLI | T089 | S | **DONE** |

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

### Phase 8 — ANE Full Forward Inference (v3)
| Task | ID | Size | Status |
|------|----|------|--------|
| ANE single-token spike | T099 | M | **DONE** |
| ANE decode forward MIL | T100 | L | **DONE** |
| ANE decode step | T101 | L | **DONE** |
| Refactor infer to ANE full | T102 | L | **DONE** |
| Golden tests (ANE full) | T103 | M | **DONE** |
| Benchmark ANE full vs hybrid | T104 | M | **DONE** |

### Phase 9 — Benchmark Harness (v3)
| Task | ID | Size | Status |
|------|----|------|--------|
| bench kernels | T105 | M | Pending |
| bench inference | T106 | M | Pending |
| bench training | T107 | M | Pending |
| Benchmark regression tracking | T108 | S | Pending |

### Phase 10 — Runtime Abstractions (v3)
| Task | ID | Size | Status |
|------|----|------|--------|
| OrionModel registry | T109 | M | Pending |
| OrionKernel interface | T110 | L | Pending |
| OrionRuntime interface | T111 | L | Pending |
| Refactor inference to abstractions | T112 | M | Pending |
| Refactor training to abstractions | T113 | M | Pending |

### Phase 11 — Build & Quality (v3)
| Task | ID | Size | Status |
|------|----|------|--------|
| Makefile | T114 | M | Pending |
| -Wall -Wextra clean build | T115 | S | Pending |
| ANE constraints doc | T116 | S | Pending |

## Task Progress
- **M0**: 11/11 complete (ALL DONE)
- **M1**: 35/35 complete (ALL DONE)
- **M2**: 10/10 complete (ALL DONE)
- **M3**: 27/27 complete (ALL DONE)
- **M4**: 6/6 complete (ALL DONE)
- **M5**: 0/6 complete
- **M6**: 0/3 complete (stretch)
- **Phase 8 (ANE Full Forward)**: 6/6 complete (ALL DONE)
- **Phase 9 (Benchmarks)**: 0/4 complete
- **Phase 10 (Abstractions)**: 0/5 complete
- **Phase 11 (Build Quality)**: 0/3 complete
- **Grand Total**: 95/116 complete (0 in progress)
- **Critical paths**: Training DONE | Weight swap DONE | ANE inference v3 DONE | Benchmarks: T105→T108

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
| 2026-03-03 | v3 spec: ANE full forward inference | Replace ANE prefill + CPU decode with ANE full forward + CPU sampling; better ANE demo |
| 2026-03-03 | v3 spec: Runtime abstraction layers | OrionKernel, OrionModel, OrionRuntime; GPT-2/Stories are reference impls |
| 2026-03-03 | v3 spec: Expanded benchmark harness | bench kernels, inference, training, swap; SRAM spill detection |
| 2026-03-03 | v3 spec: Project evolution roadmap | Stage 1 (Core) → Stage 2 (Compiler/auto-tune) → Stage 3 (Platform) |
| 2026-03-03 | Program cache: store/lookup (not compile-on-miss) | Cache can't know how to compile each kernel type; callers compile on miss and store |
| 2026-03-03 | Decode uses seq=16 (not seq=1) as minimum ANE bucket | ANE requires ~49KB minimum IOSurface allocation; seq=1 tensors fail at eval even though they compile |
| 2026-03-03 | ANE multi-output surfaces ordered alphabetically by MIL name | Output surfaces must be provided in alphabetical order of their MIL variable names, not return tuple order |

## Blockers
- **None** — M4 complete, Phase 8 ready to start

## Risks
| Risk | Impact | Status |
|------|--------|--------|
| Private API breaks on macOS update | All ANE work blocked | **Validated** — works on macOS 15 / M4 Max |
| ~119 compile limit per process | Training requires exec() restart | **Validated** — exec() restart works (T004/T007) |
| SDPA ignores causal masks | Wrong attention outputs | **Confirmed** — must decompose manually (T008 doc) |
| ANE minimum tensor size | Small tensors fail at eval | **CONFIRMED** — seq=1 fails at eval; minimum IOSurface allocation ~49KB, use seq=16 as decode bucket |
| ANE multi-output buffer sizes | Mixed-size outputs fail eval | **SOLVED** — pad all output IOSurfaces to max size |
| GPT-2 BPE tokenizer complexity | Large task (T028) | **SOLVED** — T028 done |
| fp16 numerical drift | Golden tests may need relaxed tolerance | **Managed** — two-tier tolerance in golden tests |
| ANE single-token dispatch overhead | Decode-on-ANE may be slower than CPU | **RESOLVED** — T099: seq=1 eval ~0.03ms for GPT-2 FFN (768→3072→768); well under 5ms threshold |
| ANE minimum tensor size blocks decode | seq_len=1 may not work on ANE | **RESOLVED** — T099: seq=1 compiles and evals for all program types (add, conv, FFN) |
