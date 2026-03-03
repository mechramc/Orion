#import <Foundation/Foundation.h>
#import <stdio.h>
#import <string.h>
#import "../tokenizer/gpt2_bpe.h"

// T030: GPT-2 BPE tokenizer golden test runner.
// Loads 20 golden vectors from tiktoken and verifies encode/decode.

static int tests_passed = 0;
static int tests_failed = 0;

static void test_encode_decode(OrionGPT2Tokenizer* tok,
                                const char* text,
                                const int* expected_tokens, int expected_count) {
    int tokens[1024];
    int count = orion_gpt2_encode(tok, text, tokens, 1024);

    // Check encode
    bool encode_ok = (count == expected_count);
    if (encode_ok) {
        for (int i = 0; i < count; i++) {
            if (tokens[i] != expected_tokens[i]) {
                encode_ok = false;
                printf("FAIL encode \"%s\": token[%d]=%d, expected %d\n",
                       text, i, tokens[i], expected_tokens[i]);
                break;
            }
        }
    } else {
        printf("FAIL encode \"%s\": count=%d, expected %d\n", text, count, expected_count);
    }

    // Check decode
    char* decoded = orion_gpt2_decode(tok, expected_tokens, expected_count);
    bool decode_ok = decoded && strcmp(decoded, text) == 0;
    if (!decode_ok) {
        printf("FAIL decode \"%s\": got \"%s\"\n", text, decoded ? decoded : "(null)");
    }
    if (decoded) free(decoded);

    if (encode_ok && decode_ok) {
        printf("PASS: \"%s\" → %d tokens\n", text, count);
        tests_passed++;
    } else {
        tests_failed++;
    }
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        printf("=== Orion GPT-2 Tokenizer Tests ===\n\n");

        OrionGPT2Tokenizer* tok = orion_gpt2_tokenizer_load(
            "tokenizer/data/vocab.json",
            "tokenizer/data/merges.txt");

        if (!tok) {
            printf("FATAL: Failed to load tokenizer\n");
            return 1;
        }

        // Load golden vectors from JSON
        NSData* data = [NSData dataWithContentsOfFile:@"tests/tokenizer_golden.json"];
        if (!data) {
            printf("FATAL: Cannot read tests/tokenizer_golden.json\n");
            return 1;
        }

        NSArray* golden = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
        if (!golden) {
            printf("FATAL: Cannot parse golden JSON\n");
            return 1;
        }

        for (NSDictionary* tc in golden) {
            NSString* text = tc[@"text"];
            NSArray<NSNumber*>* expected = tc[@"tokens"];

            int expected_tokens[1024];
            int expected_count = (int)expected.count;
            for (int i = 0; i < expected_count; i++) {
                expected_tokens[i] = expected[i].intValue;
            }

            test_encode_decode(tok, text.UTF8String, expected_tokens, expected_count);
        }

        printf("\n=== Results: %d passed, %d failed ===\n",
               tests_passed, tests_failed);

        orion_gpt2_tokenizer_free(tok);
        return tests_failed > 0 ? 1 : 0;
    }
}
