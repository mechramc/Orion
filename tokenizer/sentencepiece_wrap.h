#ifndef ORION_SENTENCEPIECE_WRAP_H
#define ORION_SENTENCEPIECE_WRAP_H

/// SentencePiece wrapper for Llama2/Stories vocabulary (32000 tokens).
/// Used for training on TinyStories dataset.

typedef struct OrionSPTokenizer OrionSPTokenizer;

/// Load SentencePiece model file.
OrionSPTokenizer* orion_sp_tokenizer_load(const char* model_path);

/// Encode text to token ids.
int orion_sp_encode(OrionSPTokenizer* tok, const char* text,
                    int* tokens, int max_tokens);

/// Decode token ids to text.
char* orion_sp_decode(OrionSPTokenizer* tok, const int* tokens, int n_tokens);

/// Free tokenizer.
void orion_sp_tokenizer_free(OrionSPTokenizer* tok);

#endif // ORION_SENTENCEPIECE_WRAP_H
