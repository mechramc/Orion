#import <Foundation/Foundation.h>
#import <math.h>
#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <unistd.h>
#import "../model/weight_loader.h"
#import "../tokenizer/gpt2_bpe.h"
#import "../kernels/inference/qwen_cpu_ops.h"

static double abs_sum(const float *x, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) total += fabs((double)x[i]);
    return total;
}

static NSDictionary* load_json(NSString* path) {
    NSData* data = [NSData dataWithContentsOfFile:path];
    if (!data) return nil;
    return [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
}

static int file_exists(const char *path) {
    return access(path, F_OK) == 0;
}

static float *load_exact(const char *blob_dir, int layer_idx, const char *suffix, int count) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/layer%d/%s", blob_dir, layer_idx, suffix);
    return orion_read_blob_f32_exact(path, count);
}

static int sampled_topk_logits(const char *blob_dir,
                               const char *lm_head_name,
                               const float *hidden,
                               int d_model,
                               int sample_vocab,
                               int *top_id,
                               float *top_logit,
                               int *runnerup_id,
                               float *runnerup_logit) {
    char path[2048];
    snprintf(path, sizeof(path), "%s/model/%s", blob_dir, lm_head_name);
    float *row = (float *)malloc((size_t)d_model * sizeof(float));
    if (!row) return 0;

    int best_id = -1;
    int second_id = -1;
    float best = -INFINITY;
    float second = -INFINITY;
    for (int tok = 0; tok < sample_vocab; tok++) {
        if (!orion_read_blob_row_f32(path, tok, d_model, row)) {
            free(row);
            return 0;
        }
        float dot = 0.0f;
        for (int i = 0; i < d_model; i++) dot += hidden[i] * row[i];
        if (dot > best) {
            second = best;
            second_id = best_id;
            best = dot;
            best_id = tok;
        } else if (dot > second) {
            second = dot;
            second_id = tok;
        }
    }

    free(row);
    *top_id = best_id;
    *top_logit = best;
    *runnerup_id = second_id;
    *runnerup_logit = second;
    return 1;
}

static int compute_next_token(const char *blob_dir,
                              OrionQwen35Manifest *manifest,
                              const int *token_ids,
                              int seq_len,
                              int sample_vocab,
                              int *top_id,
                              float *top_logit,
                              int *runnerup_id,
                              float *runnerup_logit,
                              double *last_hidden_abs_sum) {
    const int d_model = manifest->d_model;
    const int d_ff = manifest->d_ff;
    const int n_head = manifest->n_head;
    const int n_kv_head = manifest->n_kv_head;
    const int head_dim = manifest->head_dim;
    const int q_dim = n_head * head_dim;
    const int kv_dim = n_kv_head * head_dim;

    float *hidden = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *normed = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *mixer_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *mlp_out = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *scratch = (float *)calloc((size_t)seq_len * d_model, sizeof(float));
    float *last_hidden = (float *)calloc((size_t)d_model, sizeof(float));
    float *final_norm = NULL;
    if (!hidden || !normed || !mixer_out || !mlp_out || !scratch || !last_hidden) goto fail;

    char embed_path[2048];
    snprintf(embed_path, sizeof(embed_path), "%s/model/embed_tokens.bin", blob_dir);
    for (int s = 0; s < seq_len; s++) {
        if (!orion_read_blob_row_f32(embed_path, token_ids[s], d_model, hidden + s * d_model)) goto fail;
    }

    for (int layer_idx = 0; layer_idx < manifest->n_layer; layer_idx++) {
        float *input_ln = load_exact(blob_dir, layer_idx, "input_layernorm.bin", d_model);
        float *post_ln = load_exact(blob_dir, layer_idx, "post_attention_layernorm.bin", d_model);
        float *gate_proj = load_exact(blob_dir, layer_idx, "mlp_gate_proj.bin", d_ff * d_model);
        float *up_proj = load_exact(blob_dir, layer_idx, "mlp_up_proj.bin", d_ff * d_model);
        float *down_proj = load_exact(blob_dir, layer_idx, "mlp_down_proj.bin", d_model * d_ff);
        if (!input_ln || !post_ln || !gate_proj || !up_proj || !down_proj) {
            free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
            goto fail;
        }

        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(hidden + s * d_model, input_ln, d_model, 1e-6f, normed + s * d_model);
        }
        memset(mixer_out, 0, (size_t)seq_len * d_model * sizeof(float));

        char full_q_path[2048];
        snprintf(full_q_path, sizeof(full_q_path), "%s/layer%d/self_attn_q_proj.bin", blob_dir, layer_idx);
        if (file_exists(full_q_path)) {
            float *q_proj = load_exact(blob_dir, layer_idx, "self_attn_q_proj.bin", (q_dim * 2) * d_model);
            float *k_proj = load_exact(blob_dir, layer_idx, "self_attn_k_proj.bin", kv_dim * d_model);
            float *v_proj = load_exact(blob_dir, layer_idx, "self_attn_v_proj.bin", kv_dim * d_model);
            float *o_proj = load_exact(blob_dir, layer_idx, "self_attn_o_proj.bin", d_model * q_dim);
            float *q_norm = load_exact(blob_dir, layer_idx, "self_attn_q_norm.bin", head_dim);
            float *k_norm = load_exact(blob_dir, layer_idx, "self_attn_k_norm.bin", head_dim);
            if (!q_proj || !k_proj || !v_proj || !o_proj || !q_norm || !k_norm) {
                free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
                free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
                goto fail;
            }
            orion_qwen_cpu_full_attention_prefill_with_rope(
                normed, seq_len, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
                d_model, n_head, n_kv_head, head_dim, manifest->rope_theta, manifest->partial_rotary_factor,
                mixer_out
            );
            free(q_proj); free(k_proj); free(v_proj); free(o_proj); free(q_norm); free(k_norm);
        } else {
            char path_qkv[2048], path_out[2048], path_dt[2048], path_norm[2048], path_conv[2048];
            snprintf(path_qkv, sizeof(path_qkv), "%s/layer%d/linear_attn_in_proj_qkv.bin", blob_dir, layer_idx);
            snprintf(path_out, sizeof(path_out), "%s/layer%d/linear_attn_out_proj.bin", blob_dir, layer_idx);
            snprintf(path_dt, sizeof(path_dt), "%s/layer%d/linear_attn_dt_bias.bin", blob_dir, layer_idx);
            snprintf(path_norm, sizeof(path_norm), "%s/layer%d/linear_attn_norm.bin", blob_dir, layer_idx);
            snprintf(path_conv, sizeof(path_conv), "%s/layer%d/linear_attn_conv1d.bin", blob_dir, layer_idx);

            int qkv_rows = orion_blob_element_count(path_qkv) / d_model;
            int value_dim = orion_blob_element_count(path_out) / d_model;
            int num_v_heads = orion_blob_element_count(path_dt);
            int head_v_dim = orion_blob_element_count(path_norm);
            int key_dim = (qkv_rows - value_dim) / 2;
            int num_k_heads = num_v_heads;
            int head_k_dim = key_dim / num_k_heads;
            int conv_kernel = orion_blob_element_count(path_conv) / qkv_rows;

            float *in_proj_qkv = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_qkv.bin", qkv_rows * d_model);
            float *in_proj_z = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_z.bin", value_dim * d_model);
            float *in_proj_a = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_a.bin", num_v_heads * d_model);
            float *in_proj_b = load_exact(blob_dir, layer_idx, "linear_attn_in_proj_b.bin", num_v_heads * d_model);
            float *conv1d = load_exact(blob_dir, layer_idx, "linear_attn_conv1d.bin", qkv_rows * conv_kernel);
            float *dt_bias = load_exact(blob_dir, layer_idx, "linear_attn_dt_bias.bin", num_v_heads);
            float *a_log = load_exact(blob_dir, layer_idx, "linear_attn_a_log.bin", num_v_heads);
            float *norm_weight = load_exact(blob_dir, layer_idx, "linear_attn_norm.bin", head_v_dim);
            float *out_proj = load_exact(blob_dir, layer_idx, "linear_attn_out_proj.bin", d_model * value_dim);
            if (!in_proj_qkv || !in_proj_z || !in_proj_a || !in_proj_b || !conv1d ||
                !dt_bias || !a_log || !norm_weight || !out_proj) {
                free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
                free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
                free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
                goto fail;
            }
            orion_qwen_cpu_linear_attention_recurrent_prefill(
                normed, seq_len, in_proj_qkv, in_proj_z, in_proj_a, in_proj_b,
                conv1d, dt_bias, a_log, norm_weight, out_proj,
                d_model, num_k_heads, num_v_heads, head_k_dim, head_v_dim, conv_kernel,
                mixer_out
            );
            free(in_proj_qkv); free(in_proj_z); free(in_proj_a); free(in_proj_b);
            free(conv1d); free(dt_bias); free(a_log); free(norm_weight); free(out_proj);
        }

        for (int i = 0; i < seq_len * d_model; i++) hidden[i] += mixer_out[i];

        for (int s = 0; s < seq_len; s++) {
            orion_qwen_cpu_rmsnorm(hidden + s * d_model, post_ln, d_model, 1e-6f, scratch + s * d_model);
            orion_qwen_cpu_swiglu_ffn(scratch + s * d_model, gate_proj, up_proj, down_proj, d_model, d_ff, mlp_out + s * d_model);
        }
        for (int i = 0; i < seq_len * d_model; i++) hidden[i] += mlp_out[i];

        free(input_ln); free(post_ln); free(gate_proj); free(up_proj); free(down_proj);
    }

    char final_norm_path[2048];
    snprintf(final_norm_path, sizeof(final_norm_path), "%s/model/final_norm.bin", blob_dir);
    final_norm = orion_read_blob_f32_exact(final_norm_path, d_model);
    if (!final_norm) goto fail;
    orion_qwen_cpu_rmsnorm(hidden + (seq_len - 1) * d_model, final_norm, d_model, 1e-6f, last_hidden);

    const char *lm_head_name = manifest->tie_word_embeddings ? "embed_tokens.bin" : "lm_head.bin";
    if (!sampled_topk_logits(blob_dir, lm_head_name, last_hidden, d_model, sample_vocab,
                             top_id, top_logit, runnerup_id, runnerup_logit)) goto fail;
    *last_hidden_abs_sum = abs_sum(last_hidden, d_model);

    free(hidden); free(normed); free(mixer_out); free(mlp_out); free(scratch); free(last_hidden); free(final_norm);
    return 1;

fail:
    free(hidden); free(normed); free(mixer_out); free(mlp_out); free(scratch); free(last_hidden); free(final_norm);
    return 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 3) {
            fprintf(stderr, "usage: %s <blob_dir> <tokenizer_dir>\n", argv[0]);
            return 2;
        }

        const char *blob_dir = argv[1];
        NSString* tokDir = [NSString stringWithUTF8String:argv[2]];
        NSDictionary* meta = load_json([tokDir stringByAppendingPathComponent:@"meta.json"]);
        if (!meta) {
            fprintf(stderr, "FAIL: missing tokenizer meta.json\n");
            return 1;
        }
        NSString* regex = meta[@"regex_pattern"];
        NSString* vocabPath = [tokDir stringByAppendingPathComponent:@"vocab.json"];
        NSString* mergesPath = [tokDir stringByAppendingPathComponent:@"merges.txt"];
        OrionGPT2Tokenizer* tok = orion_gpt2_tokenizer_load_with_regex(vocabPath.UTF8String, mergesPath.UTF8String, regex.UTF8String);
        if (!tok) {
            fprintf(stderr, "FAIL: tokenizer load failed\n");
            return 1;
        }

        OrionQwen35Manifest* manifest = orion_qwen35_manifest_load(blob_dir);
        if (!manifest) {
            fprintf(stderr, "FAIL: manifest loader returned NULL\n");
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        const char *prompt = getenv("PROMPT");
        if (!prompt || !prompt[0]) prompt = "YES";
        int tokens[64] = {0};
        int prompt_len = orion_gpt2_encode(tok, prompt, tokens, 64);
        if (prompt_len <= 0) {
            fprintf(stderr, "FAIL: prompt encode failed\n");
            orion_qwen35_manifest_free(manifest);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        const int sample_vocab = 4096;
        int gen_ids[2] = {0};
        float gen_logits[2] = {0.0f, 0.0f};
        int gen_runnerup_ids[2] = {0};
        float gen_runnerup_logits[2] = {0.0f, 0.0f};
        float gen_top2_margins[2] = {0.0f, 0.0f};
        double final_hidden_abs = 0.0;
        int seq[128] = {0};
        memcpy(seq, tokens, (size_t)prompt_len * sizeof(int));
        int seq_len = prompt_len;

        for (int step = 0; step < 2; step++) {
            int next_id = -1;
            float next_logit = -INFINITY;
            int runnerup_id = -1;
            float runnerup_logit = -INFINITY;
            if (!compute_next_token(blob_dir, manifest, seq, seq_len, sample_vocab,
                                    &next_id, &next_logit, &runnerup_id, &runnerup_logit, &final_hidden_abs)) {
                fprintf(stderr, "FAIL: top1 compute failed at step %d\n", step);
                orion_qwen35_manifest_free(manifest);
                orion_gpt2_tokenizer_free(tok);
                return 1;
            }
            gen_ids[step] = next_id;
            gen_logits[step] = next_logit;
            gen_runnerup_ids[step] = runnerup_id;
            gen_runnerup_logits[step] = runnerup_logit;
            gen_top2_margins[step] = next_logit - runnerup_logit;
            seq[seq_len++] = next_id;
        }

        char* decoded_prompt = orion_gpt2_decode(tok, tokens, prompt_len);
        char* decoded_generated = orion_gpt2_decode(tok, gen_ids, 2);
        char* decoded_full = orion_gpt2_decode(tok, seq, seq_len);
        if (!decoded_prompt || !decoded_generated || !decoded_full) {
            fprintf(stderr, "FAIL: decode returned NULL\n");
            free(decoded_prompt); free(decoded_generated); free(decoded_full);
            orion_qwen35_manifest_free(manifest);
            orion_gpt2_tokenizer_free(tok);
            return 1;
        }

        printf("PASS: qwen35 9b decode loop cpu smoke\n");
        printf("  prompt=%s\n", prompt);
        printf("  prompt_len=%d\n", prompt_len);
        printf("  gen_len=%d\n", 2);
        printf("  sample_vocab=%d\n", sample_vocab);
        printf("  gen_token_0=%d\n", gen_ids[0]);
        printf("  gen_token_1=%d\n", gen_ids[1]);
        printf("  gen_logit_0=%.6f\n", gen_logits[0]);
        printf("  gen_logit_1=%.6f\n", gen_logits[1]);
        printf("  gen_runnerup_token_0=%d\n", gen_runnerup_ids[0]);
        printf("  gen_runnerup_token_1=%d\n", gen_runnerup_ids[1]);
        printf("  gen_runnerup_logit_0=%.6f\n", gen_runnerup_logits[0]);
        printf("  gen_runnerup_logit_1=%.6f\n", gen_runnerup_logits[1]);
        printf("  gen_top2_margin_0=%.6f\n", gen_top2_margins[0]);
        printf("  gen_top2_margin_1=%.6f\n", gen_top2_margins[1]);
        printf("  final_hidden_abs_sum=%.6f\n", final_hidden_abs);
        printf("  decoded_prompt=%s\n", decoded_prompt);
        printf("  decoded_generated=%s\n", decoded_generated);
        printf("  decoded_full=%s\n", decoded_full);
        printf("  next_blocker=%s\n", "9b ANE prefill integration");

        free(decoded_prompt);
        free(decoded_generated);
        free(decoded_full);
        orion_qwen35_manifest_free(manifest);
        orion_gpt2_tokenizer_free(tok);
        return 0;
    }
}
