#import <Foundation/Foundation.h>
#import <stdio.h>

// TODO(M1): Implement inference command
// Wiring: tokenizer → ANE prefill (M2) or CPU forward (M1) → CPU decode → sampling

int orion_cmd_infer(int argc, const char* argv[]) {
    fprintf(stderr, "orion infer: not yet implemented\n");
    return 1;
}
