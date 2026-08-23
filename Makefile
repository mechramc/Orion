# Orion — Makefile
# Build the Orion CLI, test suite, and benchmarks.

CC       = xcrun clang
CFLAGS   = -O2 -fobjc-arc -DACCELERATE_NEW_LAPACK -Wall -Wextra -I . -I core -I compiler
FRAMEWORKS = -framework Foundation -framework IOSurface -framework Accelerate
LDFLAGS  = -ldl $(FRAMEWORKS)

BUILDDIR = build

# ---------------------------------------------------------------------------
# Source files
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
	core/runtime.m \
	core/lora_adapter.m

INFERENCE_SRC = \
	kernels/inference/prefill_ane.m \
	kernels/inference/decode_ane.m \
	kernels/inference/decode_cpu.m \
	kernels/inference/kv_cache.m \
	kernels/inference/qwen_cpu_ops.m

TRAINING_SRC = \
	kernels/training/stories_train.m \
	kernels/training/stories_cpu_ops.m \
	kernels/training/qwen_lora_cpu_ops.m \
	kernels/training/qwen_lora_train.m \
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
	compiler/frontends/gpt2_final.c \
	compiler/frontends/qwen35_prefill.c \
	compiler/frontends/classifier_softmax.c \
	compiler/frontends/stories_train.c \
	compiler/frontends/lora.c

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
# Test binaries
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
	test_bench_decode \
	test_delta_compile \
	test_lora

# Compiler tests (link with compiler objects, not full library)
COMPILER_TEST_NAMES = \
	test_graph_ir \
	test_passes \
	test_ane_passes \
	test_compiler_equiv

# These Qwen frontend tests need no converted model, tokenizer assets, or ANE.
# They are therefore safe to include in the default verification path.
QWEN_SELF_CONTAINED_TEST_NAMES = \
	test_qwen35_prefill_frontend \
	test_qwen35_9b_prefill_frontend

# Remaining Qwen diagnostics compile as part of test-qwen-build but require a
# converted model/tokenizer, ANE hardware, or explicit command-line fixtures to run.
QWEN_PROBE_TEST_NAMES = \
	test_qwen35_24layer_logits_cpu_smoke \
	test_qwen35_9b_bridge_stage_diff \
	test_qwen35_9b_cpu_infer_smoke \
	test_qwen35_9b_decode_loop_cpu_smoke \
	test_qwen35_9b_decode_loop_mixed_ane_cpu_smoke \
	test_qwen35_9b_hidden_cache_dump \
	test_qwen35_9b_hybrid_layer_diff \
	test_qwen35_9b_lora_adapter_roundtrip \
	test_qwen35_9b_lora_ane_forward_probe \
	test_qwen35_9b_lora_ane_qv_probe \
	test_qwen35_9b_lora_ane_train_pairs \
	test_qwen35_9b_lora_ane_train_smoke1 \
	test_qwen35_9b_lora_reload_drift \
	test_qwen35_9b_lora_train_pairs \
	test_qwen35_9b_lora_train_sequences \
	test_qwen35_9b_lora_train_smoke1 \
	test_qwen35_9b_manifest_loader \
	test_qwen35_9b_prefill_cpu_decode_bridge \
	test_qwen35_9b_prefill_runtime_smoke \
	test_qwen35_attention_shape_audit \
	test_qwen35_cpu_forward_scaffold \
	test_qwen35_decode_loop_cpu_smoke \
	test_qwen35_full_attention_cpu_smoke \
	test_qwen35_full_attention_rope_cpu_smoke \
	test_qwen35_hybrid_dispatch_cpu_smoke \
	test_qwen35_infer_prep \
	test_qwen35_linear_attention_prep_cpu_smoke \
	test_qwen35_linear_attention_recurrent_cpu_smoke \
	test_qwen35_manifest_loader \
	test_qwen35_mlp_cpu_smoke \
	test_qwen35_tokenizer_parity_smoke

QWEN_TEST_NAMES = $(QWEN_SELF_CONTAINED_TEST_NAMES) $(QWEN_PROBE_TEST_NAMES)

TEST_BINS = $(patsubst %,$(BUILDDIR)/tests/%,$(TEST_NAMES))
COMPILER_TEST_BINS = $(patsubst %,$(BUILDDIR)/tests/%,$(COMPILER_TEST_NAMES))
QWEN_SELF_CONTAINED_TEST_BINS = $(patsubst %,$(BUILDDIR)/tests/%,$(QWEN_SELF_CONTAINED_TEST_NAMES))
QWEN_TEST_BINS = $(patsubst %,$(BUILDDIR)/tests/%,$(QWEN_TEST_NAMES))

# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

.PHONY: all clean test test-compiler test-qwen test-qwen-build bench

all: orion

# Build the CLI binary (runtime + compiler objects)
orion: $(ALL_OBJ) $(COMPILER_OBJ)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $^

# Compile .m → .o (auto-create directories)
$(BUILDDIR)/%.o: %.m
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c -o $@ $<

# Compile .c → .o (for pure C compiler sources)
$(BUILDDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -O2 -Wall -Wextra -I . -I core -I compiler -c -o $@ $<

# Build and run the default suite, including Qwen tests that have no external fixtures.
test: $(TEST_BINS) $(COMPILER_TEST_BINS) $(QWEN_SELF_CONTAINED_TEST_BINS)
	@passed=0; failed=0; total=0; \
	for t in $(TEST_BINS) $(COMPILER_TEST_BINS) $(QWEN_SELF_CONTAINED_TEST_BINS); do \
		total=$$((total + 1)); \
		name=$$(basename $$t); \
		printf "%-40s " "$$name"; \
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

# Reproduce the Orion-Q build surface: all 33 Qwen test executables must compile.
test-qwen-build: $(QWEN_TEST_BINS)
	@echo "Built 33/33 Qwen test executables."

# Build all Qwen diagnostics, then run only the two self-contained frontend tests.
# Model/tokenizer/ANE-dependent probes remain explicit opt-in binaries.
test-qwen: test-qwen-build
	@passed=0; failed=0; total=0; \
	for t in $(QWEN_SELF_CONTAINED_TEST_BINS); do \
		total=$$((total + 1)); \
		name=$$(basename $$t); \
		printf "%-40s " "$$name"; \
		if $$t > /dev/null 2>&1; then \
			printf "PASS\n"; \
			passed=$$((passed + 1)); \
		else \
			printf "FAIL\n"; \
			failed=$$((failed + 1)); \
		fi; \
	done; \
	echo ""; \
	echo "Qwen self-contained: $$passed/$$total passed, $$failed failed"; \
	if [ $$failed -gt 0 ]; then exit 1; fi

# Build each test binary: compile test .m + link with all library + compiler .o files
$(BUILDDIR)/tests/%: tests/%.m $(LIB_OBJ) $(COMPILER_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LIB_OBJ) $(COMPILER_OBJ)

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
