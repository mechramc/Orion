// compiler/frontends/classifier_softmax.c — T142: Classifier + softmax frontends

#include "classifier_softmax.h"
#include "../builder.h"
#include <stddef.h>

OrionGraph* orion_frontend_classifier_fwd(int dim, int vocab) {
    // Match milgen: hardcoded seq=256 (Stories110M training seq_len)
    int seq = 256;
    OrionGraph* g = orion_graph_create();

    // Input: fp16 [1, dim, 1, seq]
    int in_shape[4] = {1, dim, 1, seq};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, in_shape);

    // Linear projection using embedding weights: [vocab, dim, 1, 1] conv
    int cls = orion_gb_linear(g, x, "cls", dim, vocab, seq,
                               "@model_path/embed.bin", NULL);

    // Cast to fp32
    int out_shape[4] = {1, vocab, 1, seq};
    int output = orion_gb_cast(g, cls, ORION_DTYPE_FP32, "output", out_shape);

    orion_gb_output(g, output, "output");
    return g;
}

OrionGraph* orion_frontend_vocab_softmax(int vocab, int seq_len) {
    OrionGraph* g = orion_graph_create();

    // Input: fp16 [1, vocab, 1, seq_len]
    int in_shape[4] = {1, vocab, 1, seq_len};
    int x = orion_gb_input(g, "x", ORION_DTYPE_FP16, in_shape);

    // Softmax along axis=1 (channel/vocab dimension)
    int sm = orion_gb_softmax(g, x, 1, "sm");

    // Cast to fp32
    int out_shape[4] = {1, vocab, 1, seq_len};
    int output = orion_gb_cast(g, sm, ORION_DTYPE_FP32, "output", out_shape);

    orion_gb_output(g, output, "output");
    return g;
}
