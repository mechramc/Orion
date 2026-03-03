#import <Foundation/Foundation.h>
#import "tokenizer/sentencepiece_wrap.h"
#import <string.h>

static int g_pass = 0, g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        NSLog(@"FAIL: %s — %@", __func__, msg); \
        g_fail++; return; \
    } \
} while(0)

#define PASS() do { NSLog(@"PASS: %s", __func__); g_pass++; } while(0)

static const char* TOK_PATH = "tokenizer/data/tokenizer_32k.bin";

static void test_load(void) {
    OrionSPTokenizer* tok = orion_sp_tokenizer_load(TOK_PATH, 32000);
    ASSERT(tok != NULL, @"should load tokenizer");
    ASSERT(orion_sp_vocab_size(tok) == 32000, @"vocab size should be 32000");
    orion_sp_tokenizer_free(tok);
    PASS();
}

static void test_encode_basic(void) {
    OrionSPTokenizer* tok = orion_sp_tokenizer_load(TOK_PATH, 32000);
    ASSERT(tok != NULL, @"should load");

    int tokens[256];
    // "Once upon a time there was a little girl" → [9038, 2501, 263, 931, 727, 471, 263, 2217, 7826]
    int n = orion_sp_encode(tok, "Once upon a time there was a little girl", tokens, 256);
    ASSERT(n > 0, @"should produce tokens");

    // Compare with Python reference
    int expected[] = {9038, 2501, 263, 931, 727, 471, 263, 2217, 7826};
    int expected_len = 9;

    NSLog(@"  Encoded %d tokens: ", n);
    for (int i = 0; i < n; i++) NSLog(@"    [%d] = %d", i, tokens[i]);

    NSString* lenMsg = [NSString stringWithFormat:@"expected %d tokens, got %d", expected_len, n];
    ASSERT(n == expected_len, lenMsg);
    for (int i = 0; i < expected_len; i++) {
        NSString* tokMsg = [NSString stringWithFormat:@"token[%d]: expected %d, got %d", i, expected[i], tokens[i]];
        ASSERT(tokens[i] == expected[i], tokMsg);
    }

    orion_sp_tokenizer_free(tok);
    PASS();
}

static void test_decode_basic(void) {
    OrionSPTokenizer* tok = orion_sp_tokenizer_load(TOK_PATH, 32000);
    ASSERT(tok != NULL, @"should load");

    int tokens[] = {9038, 2501, 263, 931, 727, 471, 263, 2217, 7826};
    char* text = orion_sp_decode(tok, tokens, 9);
    ASSERT(text != NULL, @"decode should return text");

    // SentencePiece prepends ▁ to words → decode replaces with space
    // First token may have leading space
    NSLog(@"  Decoded: \"%s\"", text);
    ASSERT(strstr(text, "Once upon a time") != NULL, @"should contain original text");

    free(text);
    orion_sp_tokenizer_free(tok);
    PASS();
}

static void test_roundtrip(void) {
    OrionSPTokenizer* tok = orion_sp_tokenizer_load(TOK_PATH, 32000);
    ASSERT(tok != NULL, @"should load");

    const char* original = "Hello world";
    int tokens[256];
    int n = orion_sp_encode(tok, original, tokens, 256);
    ASSERT(n > 0, @"should encode");

    char* decoded = orion_sp_decode(tok, tokens, n);
    ASSERT(decoded != NULL, @"should decode");

    // SentencePiece may add leading space; trim for comparison
    const char* trimmed = decoded;
    while (*trimmed == ' ') trimmed++;

    NSLog(@"  Original: \"%s\", Decoded: \"%s\"", original, decoded);
    NSString* rtMsg = [NSString stringWithFormat:@"roundtrip mismatch: \"%s\" vs \"%s\"", original, trimmed];
    ASSERT(strcmp(trimmed, original) == 0, rtMsg);

    free(decoded);
    orion_sp_tokenizer_free(tok);
    PASS();
}

static void test_empty_input(void) {
    OrionSPTokenizer* tok = orion_sp_tokenizer_load(TOK_PATH, 32000);
    ASSERT(tok != NULL, @"should load");

    int tokens[256];
    int n = orion_sp_encode(tok, "", tokens, 256);
    ASSERT(n == 0, @"empty input should produce 0 tokens");

    orion_sp_tokenizer_free(tok);
    PASS();
}

static void test_single_char(void) {
    OrionSPTokenizer* tok = orion_sp_tokenizer_load(TOK_PATH, 32000);
    ASSERT(tok != NULL, @"should load");

    int tokens[256];
    int n = orion_sp_encode(tok, "a", tokens, 256);
    ASSERT(n == 1, @"single char should produce 1 token");

    char* decoded = orion_sp_decode(tok, tokens, n);
    ASSERT(decoded != NULL, @"should decode");
    // 'a' may decode to 'a' or ' a' depending on SentencePiece handling
    ASSERT(strstr(decoded, "a") != NULL, @"decoded should contain 'a'");

    free(decoded);
    orion_sp_tokenizer_free(tok);
    PASS();
}

static void test_load_invalid_path(void) {
    OrionSPTokenizer* tok = orion_sp_tokenizer_load("/nonexistent/path.bin", 32000);
    ASSERT(tok == NULL, @"should return NULL for invalid path");
    PASS();
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        NSLog(@"=== T062: SentencePiece Tokenizer Tests ===");
        test_load();
        test_encode_basic();
        test_decode_basic();
        test_roundtrip();
        test_empty_input();
        test_single_char();
        test_load_invalid_path();

        NSLog(@"\n=== Results: %d passed, %d failed ===", g_pass, g_fail);
        return g_fail > 0 ? 1 : 0;
    }
}
