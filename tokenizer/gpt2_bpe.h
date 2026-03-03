#ifndef ORION_GPT2_BPE_H
#define ORION_GPT2_BPE_H

#import <Foundation/Foundation.h>

/// GPT-2 BPE tokenizer.
/// Vocabulary: 50257 tokens.

typedef struct OrionGPT2Tokenizer OrionGPT2Tokenizer;

/// Load GPT-2 tokenizer from vocab + merges files.
/// If paths are NULL, attempts to load from default HuggingFace cache.
OrionGPT2Tokenizer* orion_gpt2_tokenizer_load(const char* vocab_path,
                                                const char* merges_path);

/// Encode text to token ids.
/// @param tok      Tokenizer handle
/// @param text     Input text (UTF-8)
/// @param tokens   Output token id buffer (caller-allocated)
/// @param max_tokens  Maximum number of tokens to produce
/// @return Number of tokens produced
int orion_gpt2_encode(OrionGPT2Tokenizer* tok, const char* text,
                      int* tokens, int max_tokens);

/// Decode token ids to text.
/// @param tok      Tokenizer handle
/// @param tokens   Input token ids
/// @param n_tokens Number of tokens
/// @return Decoded text (caller must free with free())
char* orion_gpt2_decode(OrionGPT2Tokenizer* tok, const int* tokens, int n_tokens);

/// Free tokenizer.
void orion_gpt2_tokenizer_free(OrionGPT2Tokenizer* tok);

#endif // ORION_GPT2_BPE_H
