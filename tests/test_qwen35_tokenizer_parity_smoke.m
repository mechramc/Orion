#import <Foundation/Foundation.h>
#import "../tokenizer/gpt2_bpe.h"

static NSDictionary* load_json(NSString* path) {
    NSData* data = [NSData dataWithContentsOfFile:path];
    if (!data) return nil;
    return [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <tokenizer_dir>\n", argv[0]);
            return 2;
        }

        NSString* tokDir = [NSString stringWithUTF8String:argv[1]];
        NSString* vocabPath = [tokDir stringByAppendingPathComponent:@"vocab.json"];
        NSString* mergesPath = [tokDir stringByAppendingPathComponent:@"merges.txt"];
        NSString* metaPath = [tokDir stringByAppendingPathComponent:@"meta.json"];
        NSString* refsPath = [tokDir stringByAppendingPathComponent:@"parity_refs.json"];

        NSDictionary* meta = load_json(metaPath);
        NSDictionary* refs = load_json(refsPath);
        if (!meta || !refs) {
            fprintf(stderr, "FAIL: missing meta/parity refs\n");
            return 1;
        }

        NSString* regex = meta[@"regex_pattern"];
        OrionGPT2Tokenizer* tok = orion_gpt2_tokenizer_load_with_regex(
            vocabPath.UTF8String,
            mergesPath.UTF8String,
            regex.UTF8String
        );
        if (!tok) {
            fprintf(stderr, "FAIL: tokenizer load failed\n");
            return 1;
        }

        NSArray* cases = refs[@"cases"];
        int encoded_ok = 0;
        int decoded_ok = 0;

        for (NSDictionary* c in cases) {
            NSString* text = c[@"text"];
            NSArray* ids = c[@"ids"];
            NSString* decoded_ref = c[@"decoded"];

            int out[512] = {0};
            int n = orion_gpt2_encode(tok, text.UTF8String, out, 512);
            if (n != (int)ids.count) {
                fprintf(stderr, "FAIL: encode length mismatch for %s (%d vs %lu)\n",
                        text.UTF8String, n, (unsigned long)ids.count);
                orion_gpt2_tokenizer_free(tok);
                return 1;
            }
            for (int i = 0; i < n; i++) {
                if (out[i] != [ids[i] intValue]) {
                    fprintf(stderr, "FAIL: token mismatch for %s at %d (%d vs %d)\n",
                            text.UTF8String, i, out[i], [ids[i] intValue]);
                    orion_gpt2_tokenizer_free(tok);
                    return 1;
                }
            }
            encoded_ok++;

            char* decoded = orion_gpt2_decode(tok, out, n);
            if (!decoded) {
                fprintf(stderr, "FAIL: decode returned NULL\n");
                orion_gpt2_tokenizer_free(tok);
                return 1;
            }
            NSString* decoded_ns = [NSString stringWithUTF8String:decoded];
            free(decoded);
            if (![decoded_ns isEqualToString:decoded_ref]) {
                fprintf(stderr, "FAIL: decode mismatch for %s\n", text.UTF8String);
                fprintf(stderr, "  got=%s\n", decoded_ns.UTF8String);
                fprintf(stderr, "  ref=%s\n", decoded_ref.UTF8String);
                orion_gpt2_tokenizer_free(tok);
                return 1;
            }
            decoded_ok++;
        }

        printf("PASS: qwen35 tokenizer parity smoke\n");
        printf("  tokenizer_dir=%s\n", tokDir.UTF8String);
        printf("  cases=%lu\n", (unsigned long)cases.count);
        printf("  encoded_ok=%d\n", encoded_ok);
        printf("  decoded_ok=%d\n", decoded_ok);

        orion_gpt2_tokenizer_free(tok);
        return 0;
    }
}
