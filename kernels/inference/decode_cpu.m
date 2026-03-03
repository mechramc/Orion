#import "decode_cpu.h"
#import <Accelerate/Accelerate.h>
#import <math.h>

// TODO(M1): Implement CPU decode

void orion_decode_cpu_step_impl(
    const OrionModelConfig* cfg,
    OrionKVCache* kv,
    const void* weights,
    int token,
    float* logits
) {
    // TODO(M1): Single-token transformer forward pass on CPU
    // 1. Embed token
    // 2. For each layer:
    //    a. LayerNorm (GPT-2) or RMSNorm (Llama)
    //    b. Q,K,V projections for this position
    //    c. Append K,V to cache
    //    d. Attention: Q @ cached_K^T / sqrt(d) → softmax → @ cached_V
    //    e. Output projection
    //    f. Residual
    //    g. FFN (LayerNorm → Linear → GELU → Linear → Residual)
    // 3. Final LayerNorm
    // 4. Logits = hidden @ embedding^T
}

int orion_sample_token(const float* logits, int vocab_size,
                       float temperature, float top_p, uint64_t* rng_state) {
    // TODO(M1): Temperature scaling → softmax → top-p nucleus sampling
    // Simple argmax fallback for temp=0
    if (temperature <= 0.0f) {
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    }
    // TODO: Full top-p sampling
    return 0;
}
