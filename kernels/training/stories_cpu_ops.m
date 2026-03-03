#import "stories_cpu_ops.h"
#import <math.h>

// T057: RMSNorm forward — out = weight * x / sqrt(mean(x^2) + eps)
void orion_cpu_rmsnorm(float* out, const float* x, const float* weight,
                       int dim, float eps) {
    // Compute sum of squares using vDSP
    float ss = 0.0f;
    vDSP_dotpr(x, 1, x, 1, &ss, dim);
    ss = 1.0f / sqrtf(ss / (float)dim + eps);

    // Scale by weight: out[i] = weight[i] * x[i] * ss
    vDSP_vmul(x, 1, weight, 1, out, 1, dim);
    vDSP_vsmul(out, 1, &ss, out, 1, dim);
}

// T058: Cross-entropy loss + softmax gradient
// For each position: softmax over vocab, then NLL for target token
// Returns average loss; writes gradient into dlogits
float orion_cpu_cross_entropy(float* dlogits, const float* logits,
                              const int* targets, int vocab, int seq_len) {
    float total_loss = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        const float* logits_t = logits + t * vocab;
        float* dlogits_t = dlogits + t * vocab;
        int target = targets[t];

        // Find max for numerical stability
        float max_val = logits_t[0];
        for (int i = 1; i < vocab; i++) {
            if (logits_t[i] > max_val) max_val = logits_t[i];
        }

        // Compute softmax: exp(x - max) / sum(exp(x - max))
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab; i++) {
            dlogits_t[i] = expf(logits_t[i] - max_val);
            sum_exp += dlogits_t[i];
        }

        float inv_sum = 1.0f / sum_exp;
        for (int i = 0; i < vocab; i++) {
            dlogits_t[i] *= inv_sum;
        }

        // NLL loss: -log(softmax[target])
        total_loss += -logf(dlogits_t[target] + 1e-10f);

        // Gradient of cross-entropy w.r.t. logits = softmax - one_hot(target)
        // dlogits already has softmax probabilities, subtract 1 at target
        dlogits_t[target] -= 1.0f;

        // Average gradient over sequence length
        float scale = 1.0f / (float)seq_len;
        vDSP_vsmul(dlogits_t, 1, &scale, dlogits_t, 1, vocab);
    }

    return total_loss / (float)seq_len;
}

// T059: Token embedding lookup — out[i] = embed_table[tokens[i]]
void orion_cpu_embedding(float* out, const float* embed_table,
                         const int* tokens, int dim, int seq_len) {
    for (int i = 0; i < seq_len; i++) {
        memcpy(out + i * dim, embed_table + tokens[i] * dim, dim * sizeof(float));
    }
}

// T060: Adam optimizer with bias correction
void orion_cpu_adam_step(float* param, const float* grad,
                         float* m, float* v,
                         int size, float lr, float beta1, float beta2,
                         float eps, int t) {
    // Bias correction factors
    float bc1 = 1.0f - powf(beta1, (float)t);
    float bc2 = 1.0f - powf(beta2, (float)t);
    float lr_t = lr * sqrtf(bc2) / bc1;

    for (int i = 0; i < size; i++) {
        float g = grad[i];

        // Update biased first moment: m = beta1*m + (1-beta1)*g
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;

        // Update biased second moment: v = beta2*v + (1-beta2)*g^2
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        // Parameter update: param -= lr_t * m / (sqrt(v) + eps)
        param[i] -= lr_t * m[i] / (sqrtf(v[i]) + eps);
    }
}

// T061: dW gradient accumulation via cblas_sgemm
// dW += x^T @ dy  where x is [m, n], dy is [m, k], dW is [n, k]
void orion_cpu_dw_accum(float* dw, const float* x, const float* dy,
                        int m, int n, int k) {
    // x^T @ dy: x is [m, n] transposed to [n, m], dy is [m, k]
    // Result is [n, k], accumulated into dw (beta=1.0)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                n, k, m,
                1.0f, x, n, dy, k,
                1.0f, dw, k);
}
