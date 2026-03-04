// compiler/mil_diff.m — T138: MIL text comparison

#import "mil_diff.h"

// Normalize a line: trim whitespace, collapse multiple spaces
static NSString* normalize_line(NSString* line) {
    NSString* trimmed = [line stringByTrimmingCharactersInSet:
                         [NSCharacterSet whitespaceCharacterSet]];
    // Collapse multiple spaces
    NSRegularExpression* regex = [NSRegularExpression regularExpressionWithPattern:@"\\s+"
                                                                          options:0 error:nil];
    return [regex stringByReplacingMatchesInString:trimmed
                                           options:0
                                             range:NSMakeRange(0, trimmed.length)
                                      withTemplate:@" "];
}

bool orion_mil_diff(NSString* a, NSString* b, char* diff_buf, int buf_size) {
    if (!a || !b) {
        if (diff_buf && buf_size > 0)
            snprintf(diff_buf, buf_size, "one or both MIL strings are nil");
        return false;
    }

    NSArray<NSString*>* lines_a = [a componentsSeparatedByString:@"\n"];
    NSArray<NSString*>* lines_b = [b componentsSeparatedByString:@"\n"];

    // Normalize and filter empty lines
    NSMutableArray<NSString*>* norm_a = [NSMutableArray array];
    NSMutableArray<NSString*>* norm_b = [NSMutableArray array];

    for (NSString* line in lines_a) {
        NSString* n = normalize_line(line);
        if (n.length > 0) [norm_a addObject:n];
    }
    for (NSString* line in lines_b) {
        NSString* n = normalize_line(line);
        if (n.length > 0) [norm_b addObject:n];
    }

    if (norm_a.count != norm_b.count) {
        if (diff_buf && buf_size > 0)
            snprintf(diff_buf, buf_size, "line count differs: %lu vs %lu",
                    (unsigned long)norm_a.count, (unsigned long)norm_b.count);
        return false;
    }

    for (NSUInteger i = 0; i < norm_a.count; i++) {
        if (![norm_a[i] isEqualToString:norm_b[i]]) {
            if (diff_buf && buf_size > 0)
                snprintf(diff_buf, buf_size, "line %lu differs:\n  A: %s\n  B: %s",
                        (unsigned long)(i + 1),
                        norm_a[i].UTF8String, norm_b[i].UTF8String);
            return false;
        }
    }

    return true;
}

bool orion_mil_structural_equiv(NSString* a, NSString* b, char* diff_buf, int buf_size) {
    if (!a || !b) {
        if (diff_buf && buf_size > 0)
            snprintf(diff_buf, buf_size, "one or both MIL strings are nil");
        return false;
    }

    // Count ops by scanning for known MIL op names
    NSArray<NSString*>* ops = @[@"conv(", @"add(", @"sub(", @"mul(", @"matmul(",
                                @"reshape(", @"transpose(", @"cast(", @"relu(",
                                @"tanh(", @"sigmoid(", @"softmax(", @"exp(",
                                @"pow(", @"reduce_sum(", @"reduce_mean(", @"reduce_max(",
                                @"neg(", @"sqrt(", @"rsqrt(", @"identity(",
                                @"slice_by_index("];

    for (NSString* op in ops) {
        NSUInteger count_a = 0, count_b = 0;
        NSRange search_a = NSMakeRange(0, a.length);
        NSRange search_b = NSMakeRange(0, b.length);

        while (search_a.location < a.length) {
            NSRange found = [a rangeOfString:op options:0 range:search_a];
            if (found.location == NSNotFound) break;
            count_a++;
            search_a.location = found.location + found.length;
            search_a.length = a.length - search_a.location;
        }

        while (search_b.location < b.length) {
            NSRange found = [b rangeOfString:op options:0 range:search_b];
            if (found.location == NSNotFound) break;
            count_b++;
            search_b.location = found.location + found.length;
            search_b.length = b.length - search_b.location;
        }

        if (count_a != count_b) {
            if (diff_buf && buf_size > 0)
                snprintf(diff_buf, buf_size, "op count differs for '%s': %lu vs %lu",
                        op.UTF8String, (unsigned long)count_a, (unsigned long)count_b);
            return false;
        }
    }

    // Check same number of outputs (count "} -> (" patterns)
    // Check same BLOBFILE references
    NSUInteger blob_a = [[a componentsSeparatedByString:@"BLOBFILE"] count] - 1;
    NSUInteger blob_b = [[b componentsSeparatedByString:@"BLOBFILE"] count] - 1;
    if (blob_a != blob_b) {
        if (diff_buf && buf_size > 0)
            snprintf(diff_buf, buf_size, "BLOBFILE count differs: %lu vs %lu",
                    (unsigned long)blob_a, (unsigned long)blob_b);
        return false;
    }

    return true;
}
