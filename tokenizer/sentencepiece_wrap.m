#import "sentencepiece_wrap.h"
#import <Foundation/Foundation.h>
#import <stdlib.h>
#import <string.h>
#import <math.h>

// T062: SentencePiece tokenizer using Karpathy's tokenizer.bin format.
// Implements BPE encoding/decoding for Llama2 32k vocab.

struct OrionSPTokenizer {
    char** vocab;       // vocab[i] = token string
    float* scores;      // scores[i] = BPE merge score
    int vocab_size;
    int max_token_len;
};

OrionSPTokenizer* orion_sp_tokenizer_load(const char* path, int vocab_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    OrionSPTokenizer* tok = calloc(1, sizeof(OrionSPTokenizer));
    tok->vocab_size = vocab_size;
    tok->vocab = calloc(vocab_size, sizeof(char*));
    tok->scores = calloc(vocab_size, sizeof(float));

    // Read max token length
    if (fread(&tok->max_token_len, sizeof(int), 1, f) != 1) {
        fclose(f);
        orion_sp_tokenizer_free(tok);
        return NULL;
    }

    // Read vocab entries: score(float) + len(int) + chars
    for (int i = 0; i < vocab_size; i++) {
        float score;
        int len;
        if (fread(&score, sizeof(float), 1, f) != 1) break;
        if (fread(&len, sizeof(int), 1, f) != 1) break;
        tok->scores[i] = score;
        tok->vocab[i] = calloc(len + 1, 1);
        if (fread(tok->vocab[i], 1, len, f) != (size_t)len) break;
        tok->vocab[i][len] = '\0';
    }

    fclose(f);
    return tok;
}

// Find token id for a string (linear scan — fine for encode)
static int sp_lookup(OrionSPTokenizer* tok, const char* str) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i] && strcmp(tok->vocab[i], str) == 0) {
            return i;
        }
    }
    return -1;
}

// SentencePiece ▁ character (U+2581, 3 bytes: E2 96 81)
static const char SP_SPACE[] = "\xe2\x96\x81";
static const int SP_SPACE_LEN = 3;

int orion_sp_encode(OrionSPTokenizer* tok, const char* text,
                    int* tokens, int max_tokens) {
    if (!tok || !text || !tokens) return 0;

    int text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    // Step 0: Preprocess — SentencePiece prepends ▁ and replaces spaces with ▁
    // "Once upon a time" → "▁Once▁upon▁a▁time"
    int prep_cap = text_len * 3 + SP_SPACE_LEN + 1;
    char* prep = calloc(prep_cap, 1);
    int prep_len = 0;

    // Prepend ▁
    memcpy(prep, SP_SPACE, SP_SPACE_LEN);
    prep_len = SP_SPACE_LEN;

    for (int i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            memcpy(prep + prep_len, SP_SPACE, SP_SPACE_LEN);
            prep_len += SP_SPACE_LEN;
        } else {
            prep[prep_len++] = text[i];
        }
    }
    prep[prep_len] = '\0';

    // Step 1: Initialize with per-character (or per-byte) tokens
    int n_tokens = 0;
    for (int i = 0; i < prep_len && n_tokens < max_tokens; ) {
        // Try multi-byte UTF-8 character first
        int char_len = 1;
        unsigned char c = (unsigned char)prep[i];
        if (c >= 0xF0) char_len = 4;
        else if (c >= 0xE0) char_len = 3;
        else if (c >= 0xC0) char_len = 2;

        // Look up the full UTF-8 character
        char buf[5] = {0};
        int actual_len = (i + char_len <= prep_len) ? char_len : 1;
        memcpy(buf, prep + i, actual_len);
        buf[actual_len] = '\0';

        int id = sp_lookup(tok, buf);
        if (id != -1) {
            tokens[n_tokens++] = id;
            i += actual_len;
        } else {
            // Fall back to byte token <0xNN>
            char hex[7];
            snprintf(hex, sizeof(hex), "<0x%02X>", (unsigned char)prep[i]);
            id = sp_lookup(tok, hex);
            if (id != -1) {
                tokens[n_tokens++] = id;
            }
            i++;
        }
    }
    free(prep);

    // Step 2: BPE merge loop — greedily merge the highest-scoring adjacent pair
    char* merge_buf = calloc(tok->max_token_len * 2 + 1, 1);

    while (n_tokens >= 2) {
        float best_score = -1e10f;
        int best_idx = -1;
        int best_id = -1;

        // Find best merge
        for (int i = 0; i < n_tokens - 1; i++) {
            // Concatenate tokens[i] + tokens[i+1]
            const char* a = tok->vocab[tokens[i]];
            const char* b = tok->vocab[tokens[i+1]];
            int la = (int)strlen(a);
            int lb = (int)strlen(b);
            if (la + lb > tok->max_token_len * 2) continue;
            memcpy(merge_buf, a, la);
            memcpy(merge_buf + la, b, lb);
            merge_buf[la + lb] = '\0';

            int id = sp_lookup(tok, merge_buf);
            if (id != -1 && tok->scores[id] > best_score) {
                best_score = tok->scores[id];
                best_idx = i;
                best_id = id;
            }
        }

        if (best_idx == -1) break; // No more merges possible

        // Apply merge: replace tokens[best_idx] with merged token, shift rest left
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < n_tokens - 1; i++) {
            tokens[i] = tokens[i + 1];
        }
        n_tokens--;
    }

    free(merge_buf);
    return n_tokens;
}

char* orion_sp_decode(OrionSPTokenizer* tok, const int* tokens, int n_tokens) {
    if (!tok || !tokens || n_tokens <= 0) {
        char* empty = calloc(1, 1);
        return empty;
    }

    // Calculate total length
    int total_len = 0;
    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->vocab_size && tok->vocab[tokens[i]]) {
            const char* piece = tok->vocab[tokens[i]];
            // Handle byte tokens <0xNN>
            if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
                total_len += 1; // Single byte
            } else {
                total_len += strlen(piece);
            }
        }
    }

    char* result = calloc(total_len + 1, 1);
    int pos = 0;

    for (int i = 0; i < n_tokens; i++) {
        if (tokens[i] >= 0 && tokens[i] < tok->vocab_size && tok->vocab[tokens[i]]) {
            const char* piece = tok->vocab[tokens[i]];
            if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x') {
                // Decode hex byte token
                unsigned int byte_val;
                sscanf(piece, "<0x%02X>", &byte_val);
                result[pos++] = (char)byte_val;
            } else {
                int len = (int)strlen(piece);
                memcpy(result + pos, piece, len);
                pos += len;
            }
        }
    }
    result[pos] = '\0';

    // SentencePiece uses ▁ (U+2581, 3 bytes: E2 96 81) for leading space.
    // Replace all occurrences with a regular space.
    // Count output length after replacement
    int out_len = 0;
    for (int i = 0; i < pos; ) {
        if (i + 2 < pos &&
            (unsigned char)result[i] == 0xE2 &&
            (unsigned char)result[i+1] == 0x96 &&
            (unsigned char)result[i+2] == 0x81) {
            out_len++;
            i += 3;
        } else {
            out_len++;
            i++;
        }
    }

    char* cleaned = calloc(out_len + 1, 1);
    int j = 0;
    for (int i = 0; i < pos; ) {
        if (i + 2 < pos &&
            (unsigned char)result[i] == 0xE2 &&
            (unsigned char)result[i+1] == 0x96 &&
            (unsigned char)result[i+2] == 0x81) {
            cleaned[j++] = ' ';
            i += 3;
        } else {
            cleaned[j++] = result[i++];
        }
    }
    cleaned[j] = '\0';

    free(result);
    return cleaned;
}

int orion_sp_vocab_size(OrionSPTokenizer* tok) {
    return tok ? tok->vocab_size : 0;
}

void orion_sp_tokenizer_free(OrionSPTokenizer* tok) {
    if (!tok) return;
    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }
    free(tok->scores);
    free(tok);
}
