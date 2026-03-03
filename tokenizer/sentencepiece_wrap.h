#ifndef ORION_SENTENCEPIECE_WRAP_H
#define ORION_SENTENCEPIECE_WRAP_H

/// SentencePiece wrapper for Llama2/Stories vocabulary (32000 tokens).
/// Reads Karpathy's tokenizer.bin format (from llama2.c).
/// Used for interactive inference with Stories model.
/// Training uses pretokenized data (T063) instead.

typedef struct OrionSPTokenizer OrionSPTokenizer;

/// Load tokenizer from Karpathy's tokenizer.bin format.
/// Format: max_token_len(int) + vocab_size*(score(float) + len(int) + chars)
OrionSPTokenizer* orion_sp_tokenizer_load(const char* path, int vocab_size);

/// Encode text to token ids. Returns number of tokens written.
/// Uses greedy BPE merge (matching SentencePiece behavior for inference).
int orion_sp_encode(OrionSPTokenizer* tok, const char* text,
                    int* tokens, int max_tokens);

/// Decode token ids to text. Returns newly allocated string (caller must free).
char* orion_sp_decode(OrionSPTokenizer* tok, const int* tokens, int n_tokens);

/// Get vocab size.
int orion_sp_vocab_size(OrionSPTokenizer* tok);

/// Free tokenizer.
void orion_sp_tokenizer_free(OrionSPTokenizer* tok);

#endif // ORION_SENTENCEPIECE_WRAP_H
