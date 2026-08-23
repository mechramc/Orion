#import <Foundation/Foundation.h>
#import <stdio.h>
#import "../model/weight_loader.h"

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <blob_dir>\n", argv[0]);
            return 2;
        }

        OrionQwen35Manifest *manifest = orion_qwen35_manifest_load(argv[1]);
        if (!manifest) {
            fprintf(stderr, "FAIL: qwen35 9b manifest loader returned NULL\n");
            return 1;
        }

        printf("PASS: qwen35 9b manifest loader\n");
        printf("  blob_dir=%s\n", manifest->blob_dir);
        printf("  n_layer=%d\n", manifest->n_layer);
        printf("  d_model=%d\n", manifest->d_model);
        printf("  d_ff=%d\n", manifest->d_ff);
        printf("  n_head=%d\n", manifest->n_head);
        printf("  n_kv_head=%d\n", manifest->n_kv_head);
        printf("  head_dim=%d\n", manifest->head_dim);
        printf("  vocab=%d\n", manifest->vocab);
        printf("  max_seq=%d\n", manifest->max_seq);
        printf("  tie_word_embeddings=%d\n", manifest->tie_word_embeddings);
        printf("  n_entries=%d\n", manifest->n_entries);
        printf("  n_alias_entries=%d\n", manifest->n_alias_entries);
        printf("  n_linear_layers=%d\n", manifest->n_linear_layers);
        printf("  n_full_layers=%d\n", manifest->n_full_layers);

        if (manifest->n_layer != 32 || manifest->d_model != 4096 || manifest->d_ff != 12288) {
            fprintf(stderr, "FAIL: unexpected 9b core dimensions\n");
            orion_qwen35_manifest_free(manifest);
            return 1;
        }
        if (manifest->n_head != 16 || manifest->n_kv_head != 4 || manifest->head_dim != 256) {
            fprintf(stderr, "FAIL: unexpected 9b head geometry\n");
            orion_qwen35_manifest_free(manifest);
            return 1;
        }
        if (manifest->vocab != 248320 || manifest->max_seq != 262144) {
            fprintf(stderr, "FAIL: unexpected 9b vocab/max_seq\n");
            orion_qwen35_manifest_free(manifest);
            return 1;
        }
        if (manifest->n_linear_layers != 24 || manifest->n_full_layers != 8) {
            fprintf(stderr, "FAIL: unexpected 9b hybrid topology\n");
            orion_qwen35_manifest_free(manifest);
            return 1;
        }
        if (manifest->n_entries < 400) {
            fprintf(stderr, "FAIL: too few exported entries for 9b\n");
            orion_qwen35_manifest_free(manifest);
            return 1;
        }

        orion_qwen35_manifest_free(manifest);
        return 0;
    }
}
