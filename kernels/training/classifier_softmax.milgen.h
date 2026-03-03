#ifndef ORION_CLASSIFIER_SOFTMAX_H
#define ORION_CLASSIFIER_SOFTMAX_H

#import "../../core/mil_builder.h"
#import "../../core/ane_runtime.h"

// MIL generators for classifier and softmax on ANE.
// Optional offloads from CPU (ANEgpt train_large_ane variant):
//   - Classifier forward (embed @ x): 10.2x speedup
//   - Vocab softmax (32000 classes): 33.8x speedup
//
// Note: Classifier backward is rejected by ANE (32000-input-channel conv)
// and must remain on CPU.

// TODO(M3): Implement
NSString* orion_milgen_classifier_fwd(int dim, int vocab);
NSString* orion_milgen_vocab_softmax(int vocab, int seq_len);

#endif // ORION_CLASSIFIER_SOFTMAX_H
