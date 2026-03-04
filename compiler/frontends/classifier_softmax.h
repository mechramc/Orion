// compiler/frontends/classifier_softmax.h — T142: Classifier + softmax frontends
#ifndef ORION_FRONTEND_CLASSIFIER_SOFTMAX_H
#define ORION_FRONTEND_CLASSIFIER_SOFTMAX_H

#include "../graph.h"
#include "../model_config.h"

// Build classifier forward graph: embed^T @ hidden → logits.
// Equivalent to orion_milgen_classifier_fwd.
// Input:  fp16 [1, dim, 1, seq]
// Output: fp32 [1, vocab, 1, seq]
OrionGraph* orion_frontend_classifier_fwd(int dim, int vocab);

// Build vocab softmax graph.
// Equivalent to orion_milgen_vocab_softmax.
// Input:  fp16 [1, vocab, 1, seq_len]
// Output: fp32 [1, vocab, 1, seq_len]
OrionGraph* orion_frontend_vocab_softmax(int vocab, int seq_len);

#endif // ORION_FRONTEND_CLASSIFIER_SOFTMAX_H
