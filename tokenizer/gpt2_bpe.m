#import "gpt2_bpe.h"
#import <stdlib.h>
#import <string.h>

// T028: GPT-2 BPE Tokenizer
//
// Algorithm:
// 1. Pre-tokenize using GPT-2 regex pattern (split into "words")
// 2. For each word: map bytes → Unicode chars, then iteratively apply BPE merges
// 3. Look up merged tokens in vocabulary → token ids
//
// Data files: vocab.json (50257 entries) + merges.txt (50000 merge rules)

#pragma mark - Byte-to-Unicode Mapping

// GPT-2's byte_encoder: maps each byte 0-255 to a Unicode character.
// Printable ASCII (33-126, 161-172, 174-255) maps to itself.
// Other bytes map to 256+ Unicode code points to avoid control chars.
static void build_byte_to_unicode(unichar byte_to_unicode[256]) {
    int n = 0;
    // First pass: printable ranges map to themselves
    for (int b = 0; b < 256; b++) {
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            byte_to_unicode[b] = (unichar)b;
        } else {
            byte_to_unicode[b] = 0; // mark for second pass
        }
    }
    // Second pass: non-printable bytes map to 256+
    n = 0;
    for (int b = 0; b < 256; b++) {
        if (byte_to_unicode[b] == 0) {
            byte_to_unicode[b] = (unichar)(256 + n);
            n++;
        }
    }
}

// Inverse: Unicode char → byte value
static void build_unicode_to_byte(const unichar byte_to_unicode[256], uint8_t unicode_to_byte[512]) {
    memset(unicode_to_byte, 0, 512);
    for (int b = 0; b < 256; b++) {
        unicode_to_byte[byte_to_unicode[b]] = (uint8_t)b;
    }
}

#pragma mark - Data Structures

typedef struct {
    NSString* left;
    NSString* right;
    int rank;  // Lower = higher priority
} BPEMerge;

struct OrionGPT2Tokenizer {
    int vocab_size;
    // ObjC objects stored as void* with CFBridgingRetain (ARC can't manage C struct fields)
    void* encoder;     // NSMutableDictionary<NSString*, NSNumber*>*
    void* decoder;     // NSMutableArray<NSString*>*
    void* bpe_ranks;   // NSMutableDictionary<NSString*, NSNumber*>*
    int num_merges;
    unichar byte_to_unicode[256];
    uint8_t unicode_to_byte[512];
    void* bpe_cache;   // NSMutableDictionary<NSString*, NSString*>*
    void* pretokenize_pattern; // NSString*
};

// Accessors for type safety
#define TOK_ENCODER(tok)   ((__bridge NSMutableDictionary<NSString*, NSNumber*>*)(tok)->encoder)
#define TOK_DECODER(tok)   ((__bridge NSMutableArray<NSString*>*)(tok)->decoder)
#define TOK_BPE_RANKS(tok) ((__bridge NSMutableDictionary<NSString*, NSNumber*>*)(tok)->bpe_ranks)
#define TOK_BPE_CACHE(tok) ((__bridge NSMutableDictionary<NSString*, NSString*>*)(tok)->bpe_cache)
#define TOK_PRETOKENIZE_PATTERN(tok) ((__bridge NSString*)(tok)->pretokenize_pattern)

#pragma mark - JSON Parsing (vocab.json)

static bool load_vocab(OrionGPT2Tokenizer* tok, const char* path) {
    NSData* data = [NSData dataWithContentsOfFile:
                    [NSString stringWithUTF8String:path]];
    if (!data) {
        fprintf(stderr, "tokenizer: cannot read %s\n", path);
        return false;
    }

    NSError* err = nil;
    NSDictionary* vocab = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
    if (!vocab || err) {
        fprintf(stderr, "tokenizer: JSON parse error: %s\n", err.localizedDescription.UTF8String);
        return false;
    }

    NSMutableDictionary* enc = [NSMutableDictionary dictionaryWithCapacity:vocab.count];
    int max_id = -1;
    for (NSString* key in vocab) {
        NSNumber* val = vocab[key];
        if (val.intValue > max_id) max_id = val.intValue;
    }
    NSMutableArray* dec = [NSMutableArray arrayWithCapacity:max_id + 1];

    // Pre-fill decoder with empty strings
    for (int i = 0; i <= max_id; i++) {
        [dec addObject:@""];
    }

    for (NSString* key in vocab) {
        NSNumber* val = vocab[key];
        enc[key] = val;
        int idx = val.intValue;
        if (idx < (int)dec.count) {
            dec[idx] = key;
        }
    }

    tok->encoder = (void*)CFBridgingRetain(enc);
    tok->decoder = (void*)CFBridgingRetain(dec);
    tok->vocab_size = max_id + 1;
    return true;
}

#pragma mark - Merges Parsing (merges.txt)

static bool load_merges(OrionGPT2Tokenizer* tok, const char* path) {
    NSString* content = [NSString stringWithContentsOfFile:
                         [NSString stringWithUTF8String:path]
                         encoding:NSUTF8StringEncoding error:nil];
    if (!content) {
        fprintf(stderr, "tokenizer: cannot read %s\n", path);
        return false;
    }

    NSArray<NSString*>* lines = [content componentsSeparatedByString:@"\n"];
    NSMutableDictionary* ranks = [NSMutableDictionary dictionaryWithCapacity:lines.count];

    int rank = 0;
    for (NSUInteger i = 0; i < lines.count; i++) {
        NSString* line = lines[i];
        // Skip header line "#version: ..." and empty lines
        if (line.length == 0 || [line hasPrefix:@"#"]) continue;

        // Each line is "left right"
        NSRange space = [line rangeOfString:@" "];
        if (space.location == NSNotFound) continue;

        ranks[line] = @(rank);
        rank++;
    }

    tok->bpe_ranks = (void*)CFBridgingRetain(ranks);
    tok->num_merges = rank;
    return true;
}

#pragma mark - Pre-tokenization (GPT-2 regex)

// GPT-2 pre-tokenization pattern:
// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// We implement this with NSRegularExpression.
static NSArray<NSString*>* pre_tokenize(OrionGPT2Tokenizer* tok, NSString* text) {
    NSRegularExpression* regex =
        [NSRegularExpression regularExpressionWithPattern:TOK_PRETOKENIZE_PATTERN(tok)
                                                 options:0 error:nil];
    if (!regex) return @[];

    NSMutableArray<NSString*>* tokens = [NSMutableArray array];
    NSArray<NSTextCheckingResult*>* matches =
        [regex matchesInString:text options:0
                         range:NSMakeRange(0, text.length)];

    for (NSTextCheckingResult* match in matches) {
        [tokens addObject:[text substringWithRange:match.range]];
    }

    return tokens;
}

#pragma mark - BPE Algorithm

// Convert a UTF-8 word to its byte-encoded Unicode representation.
static NSString* bytes_to_unicode_str(const char* utf8, int len,
                                       const unichar byte_to_unicode[256]) {
    NSMutableString* result = [NSMutableString stringWithCapacity:len];
    for (int i = 0; i < len; i++) {
        uint8_t b = (uint8_t)utf8[i];
        [result appendFormat:@"%C", byte_to_unicode[b]];
    }
    return result;
}

// Get the set of bigram pairs from a list of BPE tokens.
static NSArray<NSString*>* get_pairs(NSArray<NSString*>* word) {
    if (word.count < 2) return @[];
    NSMutableArray<NSString*>* pairs = [NSMutableArray arrayWithCapacity:word.count - 1];
    for (NSUInteger i = 0; i < word.count - 1; i++) {
        NSString* pair = [NSString stringWithFormat:@"%@ %@", word[i], word[i+1]];
        [pairs addObject:pair];
    }
    return pairs;
}

// Run BPE on a single word (already byte-encoded as Unicode chars).
static NSString* bpe(OrionGPT2Tokenizer* tok, NSString* token) {
    // Check cache
    NSString* cached = TOK_BPE_CACHE(tok)[token];
    if (cached) return cached;

    // Split into individual characters
    NSMutableArray<NSString*>* word = [NSMutableArray arrayWithCapacity:token.length];
    for (NSUInteger i = 0; i < token.length; i++) {
        [word addObject:[token substringWithRange:NSMakeRange(i, 1)]];
    }

    if (word.count <= 1) {
        TOK_BPE_CACHE(tok)[token] = token;
        return token;
    }

    while (true) {
        // Find the pair with lowest rank (highest priority)
        NSArray<NSString*>* pairs = get_pairs(word);
        if (pairs.count == 0) break;

        NSString* best_pair = nil;
        int best_rank = INT_MAX;
        for (NSString* pair in pairs) {
            NSNumber* rank = TOK_BPE_RANKS(tok)[pair];
            if (rank && rank.intValue < best_rank) {
                best_rank = rank.intValue;
                best_pair = pair;
            }
        }

        if (!best_pair) break;  // No more merges possible

        // Split best_pair into left and right
        NSRange space = [best_pair rangeOfString:@" "];
        NSString* left = [best_pair substringToIndex:space.location];
        NSString* right = [best_pair substringFromIndex:space.location + 1];

        // Merge all occurrences of (left, right) in word
        NSMutableArray<NSString*>* new_word = [NSMutableArray arrayWithCapacity:word.count];
        NSUInteger i = 0;
        while (i < word.count) {
            if (i < word.count - 1 &&
                [word[i] isEqualToString:left] &&
                [word[i+1] isEqualToString:right]) {
                [new_word addObject:[left stringByAppendingString:right]];
                i += 2;
            } else {
                [new_word addObject:word[i]];
                i += 1;
            }
        }

        word = new_word;
        if (word.count == 1) break;
    }

    NSString* result = [word componentsJoinedByString:@" "];
    TOK_BPE_CACHE(tok)[token] = result;
    return result;
}

#pragma mark - Public API

OrionGPT2Tokenizer* orion_gpt2_tokenizer_load_with_regex(const char* vocab_path,
                                                         const char* merges_path,
                                                         const char* regex_pattern) {
    OrionGPT2Tokenizer* tok = calloc(1, sizeof(OrionGPT2Tokenizer));

    build_byte_to_unicode(tok->byte_to_unicode);
    build_unicode_to_byte(tok->byte_to_unicode, tok->unicode_to_byte);

    if (!load_vocab(tok, vocab_path)) {
        free(tok);
        return NULL;
    }

    if (!load_merges(tok, merges_path)) {
        free(tok);
        return NULL;
    }

    tok->bpe_cache = (void*)CFBridgingRetain([NSMutableDictionary dictionaryWithCapacity:10000]);
    NSString* pattern = regex_pattern
        ? [NSString stringWithUTF8String:regex_pattern]
        : @"'s|'t|'re|'ve|'m|'ll|'d"
          @"| ?\\p{L}+"
          @"| ?\\p{N}+"
          @"| ?[^\\s\\p{L}\\p{N}]+"
          @"|\\s+(?!\\S)"
          @"|\\s+";
    tok->pretokenize_pattern = (void*)CFBridgingRetain(pattern);

    fprintf(stderr, "tokenizer: loaded %d vocab, %d merges\n",
            tok->vocab_size, tok->num_merges);
    return tok;
}

OrionGPT2Tokenizer* orion_gpt2_tokenizer_load(const char* vocab_path,
                                              const char* merges_path) {
    return orion_gpt2_tokenizer_load_with_regex(vocab_path, merges_path, NULL);
}

int orion_gpt2_encode(OrionGPT2Tokenizer* tok, const char* text,
                      int* tokens, int max_tokens) {
    {
        NSString* nsText = [NSString stringWithUTF8String:text];

        // Honor exact added-token/special-token matches before pre-tokenization.
        NSNumber* direct_id = TOK_ENCODER(tok)[nsText];
        if (direct_id && max_tokens > 0) {
            tokens[0] = direct_id.intValue;
            return 1;
        }

        NSArray<NSString*>* pre_tokens = pre_tokenize(tok, nsText);

        int count = 0;
        for (NSString* chunk in pre_tokens) {
            const char* utf8 = chunk.UTF8String;
            int len = (int)strlen(utf8);
            NSString* encoded = bytes_to_unicode_str(utf8, len, tok->byte_to_unicode);

            NSString* bpe_result = bpe(tok, encoded);

            // Split by space and look up each token
            NSArray<NSString*>* bpe_tokens = [bpe_result componentsSeparatedByString:@" "];
            for (NSString* bt in bpe_tokens) {
                NSNumber* id_num = TOK_ENCODER(tok)[bt];
                if (id_num && count < max_tokens) {
                    tokens[count++] = id_num.intValue;
                }
            }
        }

        return count;
    }
}

char* orion_gpt2_decode(OrionGPT2Tokenizer* tok, const int* tokens, int n_tokens) {
    {
        NSMutableString* text = [NSMutableString string];

        for (int i = 0; i < n_tokens; i++) {
            if (tokens[i] >= 0 && tokens[i] < tok->vocab_size) {
                [text appendString:TOK_DECODER(tok)[tokens[i]]];
            }
        }

        // Convert Unicode chars back to bytes
        NSMutableData* bytes = [NSMutableData dataWithCapacity:text.length];
        for (NSUInteger i = 0; i < text.length; i++) {
            unichar ch = [text characterAtIndex:i];
            if (ch < 512) {
                uint8_t b = tok->unicode_to_byte[ch];
                [bytes appendBytes:&b length:1];
            }
        }

        // Return as C string (caller frees)
        char* result = malloc(bytes.length + 1);
        memcpy(result, bytes.bytes, bytes.length);
        result[bytes.length] = '\0';
        return result;
    }
}

void orion_gpt2_tokenizer_free(OrionGPT2Tokenizer* tok) {
    if (!tok) return;
    if (tok->encoder)   CFRelease(tok->encoder);
    if (tok->decoder)   CFRelease(tok->decoder);
    if (tok->bpe_ranks) CFRelease(tok->bpe_ranks);
    if (tok->bpe_cache) CFRelease(tok->bpe_cache);
    if (tok->pretokenize_pattern) CFRelease(tok->pretokenize_pattern);
    free(tok);
}
