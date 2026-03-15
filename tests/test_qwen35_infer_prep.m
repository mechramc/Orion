#import <Foundation/Foundation.h>
#import <stdio.h>
#import "../core/model_registry.h"
#import "../model/weight_loader.h"

static int require_match(const char *label, int lhs, int rhs) {
    if (lhs != rhs) {
        fprintf(stderr, "FAIL: %s mismatch (registry=%d manifest=%d)\n", label, lhs, rhs);
        return 0;
    }
    return 1;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 2) {
            fprintf(stderr, "usage: %s <blob_dir>\n", argv[0]);
            return 2;
        }

        const OrionModelSpec *spec = orion_model_lookup("qwen35_08b");
        if (!spec) {
            fprintf(stderr, "FAIL: registry lookup returned NULL\n");
            return 1;
        }

        OrionQwen35Manifest *manifest = orion_qwen35_manifest_load(argv[1]);
        if (!manifest) {
            fprintf(stderr, "FAIL: manifest loader returned NULL\n");
            return 1;
        }

        printf("PASS: qwen35 cpu infer prep\n");
        printf("  model_name=%s\n", spec->name);
        printf("  blob_dir=%s\n", manifest->blob_dir);
        printf("  manifest_path=%s\n", manifest->manifest_path);
        printf("  registry_n_layer=%d\n", spec->config.n_layer);
        printf("  manifest_n_layer=%d\n", manifest->n_layer);
        printf("  registry_d_model=%d\n", spec->config.d_model);
        printf("  manifest_d_model=%d\n", manifest->d_model);
        printf("  registry_d_ff=%d\n", spec->config.hidden_dim);
        printf("  manifest_d_ff=%d\n", manifest->d_ff);
        printf("  registry_n_head=%d\n", spec->config.n_head);
        printf("  manifest_n_head=%d\n", manifest->n_head);
        printf("  registry_n_kv_head=%d\n", spec->config.n_kv_head);
        printf("  manifest_n_kv_head=%d\n", manifest->n_kv_head);
        printf("  registry_head_dim=%d\n", spec->config.head_dim);
        printf("  manifest_head_dim=%d\n", manifest->head_dim);
        printf("  registry_vocab=%d\n", spec->config.vocab);
        printf("  manifest_vocab=%d\n", manifest->vocab);
        printf("  registry_max_seq=%d\n", spec->config.max_seq);
        printf("  manifest_max_seq=%d\n", manifest->max_seq);
        printf("  manifest_n_linear_layers=%d\n", manifest->n_linear_layers);
        printf("  manifest_n_full_layers=%d\n", manifest->n_full_layers);

        int ok = 1;
        ok &= require_match("n_layer", spec->config.n_layer, manifest->n_layer);
        ok &= require_match("d_model", spec->config.d_model, manifest->d_model);
        ok &= require_match("d_ff", spec->config.hidden_dim, manifest->d_ff);
        ok &= require_match("n_head", spec->config.n_head, manifest->n_head);
        ok &= require_match("n_kv_head", spec->config.n_kv_head, manifest->n_kv_head);
        ok &= require_match("head_dim", spec->config.head_dim, manifest->head_dim);
        ok &= require_match("vocab", spec->config.vocab, manifest->vocab);
        ok &= require_match("max_seq", spec->config.max_seq, manifest->max_seq);

        if (manifest->n_linear_layers != 18 || manifest->n_full_layers != 6) {
            fprintf(stderr, "FAIL: unexpected qwen35 hybrid layer topology (%d linear / %d full)\n",
                    manifest->n_linear_layers, manifest->n_full_layers);
            ok = 0;
        }

        orion_qwen35_manifest_free(manifest);
        return ok ? 0 : 1;
    }
}
