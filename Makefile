# Orion — Makefile
# Build the Orion CLI, test suite, and benchmarks.

CC       = xcrun clang
CFLAGS   = -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK -Wall -Wextra -I . -I core
FRAMEWORKS = -framework Foundation -framework IOSurface -framework Accelerate
LDFLAGS  = -ldl $(FRAMEWORKS)

BUILDDIR = build

# ---------------------------------------------------------------------------
# Source files (30 total)
# ---------------------------------------------------------------------------

CORE_SRC = \
	core/ane_runtime.m \
	core/ane_program_cache.m \
	core/mil_builder.m \
	core/iosurface_tensor.m \
	core/profiler.m \
	core/bucket.m \
	core/checkpoint.m \
	core/model_registry.m \
	core/kernel.m \
	core/runtime.m

INFERENCE_SRC = \
	kernels/inference/prefill_ane.m \
	kernels/inference/decode_ane.m \
	kernels/inference/decode_cpu.m \
	kernels/inference/kv_cache.m \
	kernels/inference/gpt2_prefill_attn.milgen.m \
	kernels/inference/gpt2_prefill_ffn.milgen.m \
	kernels/inference/gpt2_final.milgen.m \
	kernels/inference/gpt2_decode_ane.milgen.m

TRAINING_SRC = \
	kernels/training/stories_train.m \
	kernels/training/stories_train_kernels.milgen.m \
	kernels/training/classifier_softmax.milgen.m \
	kernels/training/stories_cpu_ops.m \
	kernels/training/data_loader.m

MODEL_SRC = model/weight_loader.m

TOKENIZER_SRC = \
	tokenizer/gpt2_bpe.m \
	tokenizer/sentencepiece_wrap.m

CLI_SRC = \
	apps/cli/commands/infer.m \
	apps/cli/commands/train.m \
	apps/cli/commands/bench.m

MAIN_SRC = apps/cli/main.m

# All library sources (everything except main.m)
LIB_SRC = $(CORE_SRC) $(INFERENCE_SRC) $(TRAINING_SRC) $(MODEL_SRC) $(TOKENIZER_SRC) $(CLI_SRC)

# Object files
LIB_OBJ  = $(patsubst %.m,$(BUILDDIR)/%.o,$(LIB_SRC))
MAIN_OBJ = $(patsubst %.m,$(BUILDDIR)/%.o,$(MAIN_SRC))
ALL_OBJ  = $(LIB_OBJ) $(MAIN_OBJ)

# ---------------------------------------------------------------------------
# Test binaries (17 total — excludes test_program_swap which is a stub)
# ---------------------------------------------------------------------------

TEST_NAMES = \
	test_ane_runtime \
	test_mil_builder \
	test_cpu_forward \
	test_tokenizer \
	test_decode \
	test_infer_golden \
	test_ane_prefill \
	test_cpu_training_ops \
	test_sp_tokenizer \
	test_data_loader \
	test_train_kernels \
	test_train_smoke \
	test_program_cache \
	test_decode_ane \
	test_decode_ane_step \
	test_infer_golden_ane \
	test_bench_decode

TEST_BINS = $(patsubst %,$(BUILDDIR)/tests/%,$(TEST_NAMES))

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

.PHONY: all clean test bench

all: orion

# Build the CLI binary
orion: $(ALL_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Compile .m → .o (auto-create directories)
$(BUILDDIR)/%.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# Build and run all 17 test binaries
test: $(TEST_BINS)
	@passed=0; failed=0; total=0; \
	for t in $(TEST_BINS); do \
		total=$$((total + 1)); \
		name=$$(basename $$t); \
		printf "%-30s " "$$name"; \
		if $$t > /dev/null 2>&1; then \
			printf "PASS\n"; \
			passed=$$((passed + 1)); \
		else \
			printf "FAIL\n"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "$$passed/$$total passed, $$failed failed"; \
	if [ $$failed -gt 0 ]; then exit 1; fi

# Build each test binary: compile test .m + link with all library .o files
$(BUILDDIR)/tests/%: tests/%.m $(LIB_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIB_OBJ)

# Run benchmarks
bench: orion
	./orion bench kernels --iters 10

# Clean all build artifacts
clean:
	rm -rf $(BUILDDIR)
	rm -f orion
