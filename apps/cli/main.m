#import <Foundation/Foundation.h>
#import <stdio.h>
#import <string.h>

// Forward declarations for command handlers
extern int orion_cmd_infer(int argc, const char* argv[]);
extern int orion_cmd_train(int argc, const char* argv[]);
extern int orion_cmd_bench(int argc, const char* argv[]);

static void print_usage(void) {
    fprintf(stderr,
        "Orion — On-Device LLM on Apple Silicon (ANE)\n"
        "\n"
        "Usage:\n"
        "  orion infer   [options]   Run inference\n"
        "  orion train   [options]   Train a model\n"
        "  orion bench   [options]   Run benchmarks\n"
        "\n"
        "Run 'orion <command> --help' for command-specific options.\n"
    );
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            print_usage();
            return 1;
        }

        const char* cmd = argv[1];

        if (strcmp(cmd, "infer") == 0) {
            return orion_cmd_infer(argc - 1, argv + 1);
        } else if (strcmp(cmd, "train") == 0) {
            return orion_cmd_train(argc - 1, argv + 1);
        } else if (strcmp(cmd, "bench") == 0) {
            return orion_cmd_bench(argc - 1, argv + 1);
        } else if (strcmp(cmd, "--help") == 0 || strcmp(cmd, "-h") == 0) {
            print_usage();
            return 0;
        } else {
            fprintf(stderr, "Unknown command: %s\n\n", cmd);
            print_usage();
            return 1;
        }
    }
}
