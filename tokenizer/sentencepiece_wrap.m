#import "sentencepiece_wrap.h"
#import <stdlib.h>

// TODO(M3): Implement SentencePiece wrapper
// ANEgpt uses pretokenized data (download_data.sh → tinystories_data00.bin)
// so this is only needed for interactive inference with Stories model

struct OrionSPTokenizer {
    // TODO: SentencePiece model handle
    int vocab_size;
};

OrionSPTokenizer* orion_sp_tokenizer_load(const char* model_path) {
    // TODO(M3): Load .model file
    return NULL;
}

int orion_sp_encode(OrionSPTokenizer* tok, const char* text,
                    int* tokens, int max_tokens) {
    // TODO(M3): Encode
    return 0;
}

char* orion_sp_decode(OrionSPTokenizer* tok, const int* tokens, int n_tokens) {
    // TODO(M3): Decode
    return NULL;
}

void orion_sp_tokenizer_free(OrionSPTokenizer* tok) {
    free(tok);
}
