#import <Foundation/Foundation.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <sys/time.h>
#import "../../../core/profiler.h"
#import "../../../core/ane_runtime.h"
#import "../../../model/weight_loader.h"
#import "../../../model/configs/gpt2_124m.h"
#import "../../../tokenizer/gpt2_bpe.h"
#import "../../../kernels/inference/decode_cpu.h"
#import "../../../kernels/inference/kv_cache.h"
#import "../../../kernels/inference/prefill_ane.h"
#import "../../../kernels/inference/decode_ane.h"

// T041-T042: CLI inference command
// T054: Wire hybrid inference (ANE prefill → CPU decode)
// T102: ANE full forward (ANE prefill → ANE decode)
//
// Usage: orion infer --prompt "Hello, world" --max_tokens 64 [--ane]

static void print_infer_help(void) {
    fprintf(stderr,
        "Usage: orion infer [options]\n"
        "\n"
        "Options:\n"
        "  --prompt TEXT          Input prompt (required)\n"
        "  --max_tokens N         Maximum tokens to generate (default: 128)\n"
        "  --temperature FLOAT    Sampling temperature (0=greedy, default: 0.0)\n"
        "  --top_p FLOAT          Top-p nucleus sampling (default: 0.9)\n"
        "  --seed N               RNG seed (default: 42)\n"
        "  --ane                  Use ANE for prefill + decode (default: CPU only)\n"
        "  --ane-prefill          Use ANE for prefill only, CPU decode (v2 mode)\n"
        "  --weights PATH         Path to weight blobs directory\n"
        "                         (default: model/blobs/gpt2_124m)\n"
        "  --vocab PATH           Path to vocab.json (default: tokenizer/data/vocab.json)\n"
        "  --merges PATH          Path to merges.txt (default: tokenizer/data/merges.txt)\n"
        "  --help                 Show this help message\n"
    );
}

static double time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int orion_cmd_infer(int argc, const char* argv[]) {
    // Parse arguments
    const char* prompt = NULL;
    int max_tokens = 128;
    float temperature = 0.0f;
    float top_p = 0.9f;
    uint64_t seed = 42;
    bool use_ane = false;
    bool ane_decode = false;  // true = ANE decode (v3), false = CPU decode
    const char* weights_path = "model/blobs/gpt2_124m";
    const char* vocab_path = "tokenizer/data/vocab.json";
    const char* merges_path = "tokenizer/data/merges.txt";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--max_tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = atof(argv[++i]);
        } else if (strcmp(argv[i], "--top_p") == 0 && i + 1 < argc) {
            top_p = atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--ane") == 0) {
            use_ane = true;
            ane_decode = true;
        } else if (strcmp(argv[i], "--ane-prefill") == 0) {
            use_ane = true;
            ane_decode = false;
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--merges") == 0 && i + 1 < argc) {
            merges_path = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            print_infer_help();
            return 0;
        }
    }

    if (!prompt) {
        fprintf(stderr, "Error: --prompt is required\n\n");
        print_infer_help();
        return 1;
    }

    // Initialize ANE if requested
    if (use_ane) {
        if (!orion_ane_init()) {
            fprintf(stderr, "Warning: ANE init failed, falling back to CPU\n");
            use_ane = false;
        }
    }

    // Load tokenizer
    fprintf(stderr, "Loading tokenizer...\n");
    OrionGPT2Tokenizer* tok = orion_gpt2_tokenizer_load(vocab_path, merges_path);
    if (!tok) {
        fprintf(stderr, "Error: failed to load tokenizer\n");
        return 1;
    }

    // Load weights
    fprintf(stderr, "Loading weights from %s...\n", weights_path);
    double t0 = time_ms();
    OrionGPT2Weights* w = orion_gpt2_weights_load(weights_path);
    if (!w) {
        fprintf(stderr, "Error: failed to load weights\n");
        orion_gpt2_tokenizer_free(tok);
        return 1;
    }
    double t_load = time_ms() - t0;
    fprintf(stderr, "Weights loaded in %.1f ms\n", t_load);

    OrionModelConfig cfg = kGPT2_124M;
    cfg.n_layer = w->n_layer;
    cfg.d_model = w->d_model;
    cfg.hidden_dim = w->d_ff;
    cfg.vocab = w->vocab;
    cfg.max_seq = w->max_seq;

    // Tokenize prompt
    int prompt_tokens[1024];
    int prompt_len = orion_gpt2_encode(tok, prompt, prompt_tokens, 1024);
    if (prompt_len == 0) {
        fprintf(stderr, "Error: failed to tokenize prompt\n");
        orion_gpt2_weights_free(w);
        orion_gpt2_tokenizer_free(tok);
        return 1;
    }
    fprintf(stderr, "Prompt: \"%s\" → %d tokens\n", prompt, prompt_len);

    // Allocate
    float* logits = (float*)malloc(w->vocab * sizeof(float));
    OrionKVCache* kv = orion_kv_cache_create(&cfg);
    int gen_count = 0;

    // Start profiler
    orion_prof_begin();

    // Prefill
    fprintf(stderr, "Prefilling %d tokens (%s)...\n", prompt_len,
            use_ane ? "ANE" : "CPU");
    t0 = time_ms();

    bool prefill_ok;
    if (use_ane) {
        prefill_ok = orion_ane_prefill(w, prompt_tokens, prompt_len,
                                        &cfg, weights_path, kv, logits);
        if (!prefill_ok) {
            fprintf(stderr, "Warning: ANE prefill failed, falling back to CPU\n");
            // Reset KV cache
            orion_kv_cache_free(kv);
            kv = orion_kv_cache_create(&cfg);
            orion_gpt2_prefill_kv(w, prompt_tokens, prompt_len, kv, logits);
            prefill_ok = true;
        }
    } else {
        orion_gpt2_prefill_kv(w, prompt_tokens, prompt_len, kv, logits);
        prefill_ok = true;
    }

    double t_prefill = time_ms() - t0;
    fprintf(stderr, "Prefill: %.1f ms (%.1f ms/token)\n", t_prefill, t_prefill / prompt_len);

    // Print prompt
    printf("%s", prompt);
    fflush(stdout);

    // Decode loop
    uint64_t rng_state = seed;
    fprintf(stderr, "Decoding (%s)...\n", ane_decode ? "ANE" : "CPU");

    for (int i = 0; i < max_tokens; i++) {
        int next_token = orion_sample_token(logits, w->vocab, temperature, top_p, &rng_state);

        if (next_token == 50256) break;  // EOS
        gen_count++;

        // Print decoded token
        char* decoded = orion_gpt2_decode(tok, &next_token, 1);
        if (decoded) {
            printf("%s", decoded);
            fflush(stdout);
            free(decoded);
        }

        // Decode step (timed)
        t0 = time_ms();
        if (ane_decode) {
            bool decode_ok = orion_ane_decode_step(w, kv, next_token, weights_path, logits);
            if (!decode_ok) {
                fprintf(stderr, "Warning: ANE decode failed at step %d, falling back to CPU\n", i);
                ane_decode = false;
                orion_gpt2_decode_step(w, kv, next_token, logits);
            }
        } else {
            orion_gpt2_decode_step(w, kv, next_token, logits);
        }
        double decode_ms = time_ms() - t0;
        orion_prof_record_decode(decode_ms);
    }

    printf("\n");

    // Print profiler stats
    OrionPerfMetrics metrics = orion_prof_finish(gen_count, 0);
    metrics.prefill_ms = t_prefill;
    const char *backend = "CPU";
    if (use_ane && ane_decode) backend = "ANE full";
    else if (use_ane) backend = "ANE prefill + CPU decode";
    fprintf(stderr, "\nPrompt tokens: %d, Generated tokens: %d, Backend: %s\n",
            prompt_len, gen_count, backend);
    orion_prof_print(&metrics);

    // Cleanup
    free(logits);
    orion_kv_cache_free(kv);
    orion_gpt2_weights_free(w);
    orion_gpt2_tokenizer_free(tok);

    return 0;
}
