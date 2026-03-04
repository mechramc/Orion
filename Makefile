# Orion — Makefile
# Build the Orion CLI, test suite, and benchmarks.

CC       = xcrun clang
CFLAGS   = -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK -Wall -Wextra -I . -I core -I compiler
FRAMEWORKS = -framework Foundation -framework IOSurface -framework Accelerate
LDFLAGS  = -ldl $(FRAMEWORKS)

BUILDDIR = build

# ---------------------------------------------------------------------------
# Source files (42 total: 30 original + 12 compiler)
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

# Stage 2: Compiler sources (pure C + ObjC for codegen/adapter)
COMPILER_C_SRC = \
	compiler/graph.c \
	compiler/builder.c \
	compiler/topo.c \
	compiler/patterns.c \
	compiler/validate.c \
	compiler/pass_dce.c \
	compiler/pass_identity.c \
	compiler/pass_conv_bias.c \
	compiler/pass_cast.c \
	compiler/pass_sram.c \
	compiler/pass_uniform_outputs.c \
	compiler/pass_ane_validate.c \
	compiler/pipeline.c \
	compiler/frontends/gpt2_prefill.c \
	compiler/frontends/gpt2_decode.c \
	compiler/frontends/stories_train.c

COMPILER_M_SRC = \
	compiler/codegen.m \
	compiler/kernel_adapter.m \
	compiler/mil_diff.m

MAIN_SRC = apps/cli/main.m

# All library sources (everything except main.m)
LIB_SRC = $(CORE_SRC) $(INFERENCE_SRC) $(TRAINING_SRC) $(MODEL_SRC) $(TOKENIZER_SRC) $(CLI_SRC)

# Object files for library
LIB_OBJ  = $(patsubst %.m,$(BUILDDIR)/%.o,$(LIB_SRC))
MAIN_OBJ = $(patsubst %.m,$(BUILDDIR)/%.o,$(MAIN_SRC))
ALL_OBJ  = $(LIB_OBJ) $(MAIN_OBJ)

# Compiler object files (C and ObjC)
COMPILER_C_OBJ = $(patsubst %.c,$(BUILDDIR)/%.o,$(COMPILER_C_SRC))
COMPILER_M_OBJ = $(patsubst %.m,$(BUILDDIR)/%.o,$(COMPILER_M_SRC))
COMPILER_OBJ = $(COMPILER_C_OBJ) $(COMPILER_M_OBJ)

# ---------------------------------------------------------------------------
# Test binaries (21 total: 17 original + 4 compiler tests)
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

# Compiler tests (link with compiler objects, not full library)
COMPILER_TEST_NAMES = \
	test_graph_ir \
	test_passes \
	test_ane_passes \
	test_compiler_equiv

TEST_BINS = $(patsubst %,$(BUILDDIR)/tests/%,$(TEST_NAMES))
COMPILER_TEST_BINS = $(patsubst %,$(BUILDDIR)/tests/%,$(COMPILER_TEST_NAMES))

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

.PHONY: all clean test test-compiler bench

all: orion

# Build the CLI binary
orion: $(ALL_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Compile .m → .o (auto-create directories)
$(BUILDDIR)/%.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile .c → .o (for pure C compiler sources)
$(BUILDDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -O2 -Wall -Wextra -I . -I core -I compiler -c -o $@ $<

# Build and run all test binaries (original 17 + 4 compiler tests)
test: $(TEST_BINS) $(COMPILER_TEST_BINS)
	@passed=0; failed=0; total=0; \
	for t in $(TEST_BINS) $(COMPILER_TEST_BINS); do \
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

# Run only compiler tests
test-compiler: $(COMPILER_TEST_BINS)
	@passed=0; failed=0; total=0; \
	for t in $(COMPILER_TEST_BINS); do \
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

# Build compiler test binaries: link with compiler objects only (+ Foundation)
$(BUILDDIR)/tests/test_graph_ir: tests/test_graph_ir.m $(COMPILER_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -framework Foundation -o $@ $< $(COMPILER_OBJ)

$(BUILDDIR)/tests/test_passes: tests/test_passes.m $(COMPILER_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -framework Foundation -o $@ $< $(COMPILER_OBJ)

$(BUILDDIR)/tests/test_ane_passes: tests/test_ane_passes.m $(COMPILER_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -framework Foundation -o $@ $< $(COMPILER_OBJ)

$(BUILDDIR)/tests/test_compiler_equiv: tests/test_compiler_equiv.m $(COMPILER_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -framework Foundation -o $@ $< $(COMPILER_OBJ)

# Run benchmarks
bench: orion
	./orion bench kernels --iters 10

# Clean all build artifacts
clean:
	rm -rf $(BUILDDIR)
	rm -f orion
