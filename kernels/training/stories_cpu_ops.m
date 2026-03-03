#import "stories_cpu_ops.h"
#import <math.h>

// TODO(M3): Implement all CPU ops with vDSP/cblas vectorization

void orion_cpu_rmsnorm(float* out, const float* x, const float* weight,
                       int dim, float eps) {
    // TODO: RMSNorm: out = weight * x / sqrt(mean(x^2) + eps)
}

float orion_cpu_cross_entropy(float* dlogits, const float* logits,
                              const int* targets, int vocab, int seq_len) {
    // TODO: NLL loss + gradient via softmax → gather → log
    return 0.0f;
}

void orion_cpu_embedding(float* out, const float* embed_table,
                         const int* tokens, int dim, int seq_len) {
    // TODO: Simple table lookup — out[i] = embed_table[tokens[i]]
}

void orion_cpu_adam_step(float* param, const float* grad,
                         float* m, float* v,
                         int size, float lr, float beta1, float beta2,
                         float eps, int t) {
    // TODO: Adam with bias correction
}

void orion_cpu_dw_accum(float* dw, const float* x, const float* dy,
                        int m, int n, int k) {
    // TODO: dW += x^T @ dy via cblas_sgemm
    // cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
    //             n, k, m, 1.0f, x, n, dy, k, 1.0f, dw, k);
}
