# Orion — On-Device LLM on Apple Silicon (ANE)

## Tech Stack
- Runtime: macOS (Objective-C, Swift)
- AI: GPT-2 124M (inference), Stories110M/Llama2-style (training)
- Hardware: Apple Neural Engine via private APIs (`_ANEClient`, `_ANECompiler`, MIL)
- Infra: Mac Studio M4 Max 64GB unified memory
- Build: Xcode / clang, Python for weight conversion
- Key deps: IOSurface, Accelerate (cblas), SentencePiece

## Code Conventions
- Core runtime: Objective-C (.h/.m)
- MIL kernel generators: .milgen.h files (C/ObjC that emit MIL text)
- SwiftUI app: Swift
- Weight converters: Python
- Tensor layout: fp16 `[1, C, 1, S]` for all ANE I/O (IOSurface-backed)
- Naming: `orion_` prefix for all public C API functions
- File naming: snake_case for all source files

## Build & Run
```bash
# Build CLI (once Xcode project exists)
xcodebuild -scheme orion -configuration Release

# Inference
./orion infer --model gpt2_124m --weights base --prompt "Hello" --max_new_tokens 128

# Training
./orion train --model stories110m --dataset tinystories --steps 20000

# Weight swap benchmark
./orion bench swap --model stories110m --ckpt_a ckpt_01000 --ckpt_b ckpt_01100 --iters 100
```

## Testing
```bash
# Tokenizer golden tests (20 prompts)
./orion test tokenizer

# Inference golden tests (token-exact + top-k fallback)
./orion test infer --golden tests/test_infer_golden.json

# Training smoke (50 steps, loss decreases, checkpoint works)
./orion test train

# Program swap endurance (100 alternations, no memory leak)
./orion test swap
```

## Architecture Notes
- Weights are baked into compiled ANE programs (BLOBFILE constants)
- Weight swapping = recompile with new weights (Mode A)
- Causal attention: explicit decomposition required (ANE SDPA ignores masks)
- Compiler leaks memory: aggressive ARC release + program cache limits
- Inference: ANE prefill (bucketed) → CPU decode with KV cache
- Training: ANE forward/backward (dx) → CPU dW + Adam

## Session Tracking
- **ALWAYS update `STATUS.md` and `CHECKPOINT.md` before committing** — no exceptions
- `STATUS.md` tracks milestone progress, task completion, decisions, risks
- `CHECKPOINT.md` is the cross-session handoff document
- `TASKS.md` has the full 94-task atomic task list with dependencies
- Update task status in STATUS.md as you complete work
- Record decisions, blockers, and key findings in both files

## Do's
- Always release ANE program objects explicitly after use
- Use bucketed sequence lengths (32/64/128/256/512/1024)
- Pin macOS version for reproducible builds
- Keep ANE work chunky (full prefill/layer ops), CPU for token loops
- Test weight swap endurance (100+ swaps without restart)
- Update STATUS.md + CHECKPOINT.md before every commit

## Don'ts
- Don't pass causal masks to ANE SDPA (it ignores them)
- Don't hold references to compiled programs longer than needed
- Don't commit weight blobs (model/blobs/ is gitignored)
- Don't assume ANE fp16 matches PyTorch fp32 exactly
- Don't use interactive/streaming ANE calls (batch only)

## Milestones
- M0: Build + run upstream ANEgpt reference
- M1: CPU baseline inference (GPT-2)
- M2: ANE prefill for GPT-2
- M3: Training integration (Stories110M)
- M4: Weight swapping (Mode A)
- M5: Demo app + narrative
- M6: (stretch) Adapter-as-input (Mode B / LoRA)
