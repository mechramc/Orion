#import <Foundation/Foundation.h>
#import <stdio.h>

// TODO(M3): Implement training command
// Wiring: data loader → forward (ANE) → loss (CPU) → backward (ANE) → dW (CPU) → Adam → checkpoint

int orion_cmd_train(int argc, const char* argv[]) {
    fprintf(stderr, "orion train: not yet implemented\n");
    return 1;
}
