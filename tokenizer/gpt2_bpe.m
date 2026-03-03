#import "gpt2_bpe.h"

// TODO(M1): Implement GPT-2 BPE tokenizer
// Options:
//   1. Port tiktoken's BPE algorithm to C/ObjC
//   2. Wrap tiktoken via Python subprocess
//   3. Implement from vocab.json + merges.txt directly

struct OrionGPT2Tokenizer {
    // TODO: BPE merge table, vocab mapping, byte encoder
    int vocab_size;
};

OrionGPT2Tokenizer* orion_gpt2_tokenizer_load(const char* vocab_path,
                                                const char* merges_path) {
    // TODO(M1): Load vocab.json + merges.txt
    return NULL;
}

int orion_gpt2_encode(OrionGPT2Tokenizer* tok, const char* text,
                      int* tokens, int max_tokens) {
    // TODO(M1): BPE encode
    return 0;
}

char* orion_gpt2_decode(OrionGPT2Tokenizer* tok, const int* tokens, int n_tokens) {
    // TODO(M1): Decode token ids back to text
    return NULL;
}

void orion_gpt2_tokenizer_free(OrionGPT2Tokenizer* tok) {
    free(tok);
}
