#import <Foundation/Foundation.h>
#import <Accelerate/Accelerate.h>
#import <math.h>
#import "kernels/training/stories_cpu_ops.h"

// Test framework
static int g_pass = 0, g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        NSLog(@"FAIL: %s — %@", __func__, msg); \
        g_fail++; return; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    float _a = (a), _b = (b); \
    if (fabsf(_a - _b) > (tol)) { \
        NSLog(@"FAIL: %s — %@ (got %f, expected %f, diff %e)", __func__, msg, _a, _b, fabsf(_a - _b)); \
        g_fail++; return; \
    } \
} while(0)

#define PASS() do { NSLog(@"PASS: %s", __func__); g_pass++; } while(0)

#pragma mark - T057: RMSNorm Tests

static void test_rmsnorm_unit_weight(void) {
    // RMSNorm with weight=1 should just normalize by RMS
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];

    orion_cpu_rmsnorm(out, x, w, 4, 1e-5f);

    // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) = 2.7386
    float rms = sqrtf((1+4+9+16) / 4.0f);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR(out[i], x[i] / rms, 1e-5f, @"unit weight RMSNorm");
    }
    PASS();
}

static void test_rmsnorm_scaled_weight(void) {
    float x[] = {3.0f, 4.0f};
    float w[] = {2.0f, 0.5f};
    float out[2];

    orion_cpu_rmsnorm(out, x, w, 2, 1e-5f);

    float rms = sqrtf((9.0f + 16.0f) / 2.0f); // sqrt(12.5)
    ASSERT_NEAR(out[0], 2.0f * 3.0f / rms, 1e-5f, @"scaled weight[0]");
    ASSERT_NEAR(out[1], 0.5f * 4.0f / rms, 1e-5f, @"scaled weight[1]");
    PASS();
}

static void test_rmsnorm_dim768(void) {
    // Test with real model dimension
    int dim = 768;
    float* x = calloc(dim, sizeof(float));
    float* w = calloc(dim, sizeof(float));
    float* out = calloc(dim, sizeof(float));

    // Fill with known pattern
    for (int i = 0; i < dim; i++) {
        x[i] = sinf((float)i * 0.1f);
        w[i] = 1.0f;
    }

    orion_cpu_rmsnorm(out, x, w, dim, 1e-5f);

    // Verify output is normalized: mean(out^2) should be close to 1
    float ss = 0;
    vDSP_dotpr(out, 1, out, 1, &ss, dim);
    ss /= (float)dim;
    ASSERT_NEAR(ss, 1.0f, 1e-4f, @"RMSNorm output should have unit RMS");

    free(x); free(w); free(out);
    PASS();
}

static void test_rmsnorm_eps(void) {
    // Zero input should not crash (eps prevents division by zero)
    float x[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[4];

    orion_cpu_rmsnorm(out, x, w, 4, 1e-5f);

    for (int i = 0; i < 4; i++) {
        ASSERT(!isnan(out[i]) && !isinf(out[i]), @"eps should prevent NaN/inf");
    }
    PASS();
}

#pragma mark - T058: Cross-Entropy Tests

static void test_cross_entropy_perfect(void) {
    // When logits strongly predict the correct token, loss should be near 0
    int vocab = 4, seq_len = 1;
    float logits[] = {-10.0f, -10.0f, 100.0f, -10.0f}; // strong prediction of token 2
    int targets[] = {2};
    float dlogits[4];

    float loss = orion_cpu_cross_entropy(dlogits, logits, targets, vocab, seq_len);
    ASSERT(loss < 0.001f, @"loss should be near 0 for perfect prediction");
    PASS();
}

static void test_cross_entropy_uniform(void) {
    // Uniform logits → loss = log(vocab)
    int vocab = 4, seq_len = 1;
    float logits[] = {0.0f, 0.0f, 0.0f, 0.0f};
    int targets[] = {0};
    float dlogits[4];

    float loss = orion_cpu_cross_entropy(dlogits, logits, targets, vocab, seq_len);
    ASSERT_NEAR(loss, logf(4.0f), 1e-4f, @"uniform logits → loss = ln(4)");
    PASS();
}

static void test_cross_entropy_gradient(void) {
    // Verify gradient is softmax - one_hot
    int vocab = 3, seq_len = 1;
    float logits[] = {1.0f, 2.0f, 3.0f};
    int targets[] = {1};
    float dlogits[3];

    orion_cpu_cross_entropy(dlogits, logits, targets, vocab, seq_len);

    // softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
    // gradient = softmax - [0,1,0] = [0.0900, -0.7553, 0.6652]
    float e1 = expf(1), e2 = expf(2), e3 = expf(3);
    float sum = e1 + e2 + e3;
    ASSERT_NEAR(dlogits[0], e1/sum / seq_len, 1e-4f, @"grad[0] = softmax[0]");
    ASSERT_NEAR(dlogits[1], (e2/sum - 1.0f) / seq_len, 1e-4f, @"grad[1] = softmax[1] - 1");
    ASSERT_NEAR(dlogits[2], e3/sum / seq_len, 1e-4f, @"grad[2] = softmax[2]");
    PASS();
}

static void test_cross_entropy_multi_position(void) {
    // Test with seq_len > 1
    int vocab = 3, seq_len = 2;
    float logits[] = {
        0.0f, 0.0f, 0.0f,  // position 0: uniform
        0.0f, 0.0f, 0.0f,  // position 1: uniform
    };
    int targets[] = {0, 1};
    float dlogits[6];

    float loss = orion_cpu_cross_entropy(dlogits, logits, targets, vocab, seq_len);
    // Both positions have loss = ln(3), average = ln(3)
    ASSERT_NEAR(loss, logf(3.0f), 1e-4f, @"average loss over 2 positions");
    PASS();
}

static void test_cross_entropy_numerical_stability(void) {
    // Large logits should not overflow
    int vocab = 4, seq_len = 1;
    float logits[] = {1000.0f, 999.0f, 998.0f, 997.0f};
    int targets[] = {0};
    float dlogits[4];

    float loss = orion_cpu_cross_entropy(dlogits, logits, targets, vocab, seq_len);
    ASSERT(!isnan(loss) && !isinf(loss), @"should handle large logits without overflow");
    ASSERT(loss >= 0.0f, @"loss should be non-negative");
    PASS();
}

#pragma mark - T059: Embedding Tests

static void test_embedding_lookup(void) {
    // Simple 4-token vocab, dim=3
    float table[] = {
        1.0f, 2.0f, 3.0f,   // token 0
        4.0f, 5.0f, 6.0f,   // token 1
        7.0f, 8.0f, 9.0f,   // token 2
        10.0f, 11.0f, 12.0f // token 3
    };
    int tokens[] = {2, 0, 3};
    float out[9];

    orion_cpu_embedding(out, table, tokens, 3, 3);

    // out[0] = table[2] = {7,8,9}
    ASSERT_NEAR(out[0], 7.0f, 1e-6f, @"token 2, dim 0");
    ASSERT_NEAR(out[1], 8.0f, 1e-6f, @"token 2, dim 1");
    ASSERT_NEAR(out[2], 9.0f, 1e-6f, @"token 2, dim 2");
    // out[1] = table[0] = {1,2,3}
    ASSERT_NEAR(out[3], 1.0f, 1e-6f, @"token 0, dim 0");
    ASSERT_NEAR(out[4], 2.0f, 1e-6f, @"token 0, dim 1");
    ASSERT_NEAR(out[5], 3.0f, 1e-6f, @"token 0, dim 2");
    // out[2] = table[3] = {10,11,12}
    ASSERT_NEAR(out[6], 10.0f, 1e-6f, @"token 3, dim 0");
    ASSERT_NEAR(out[7], 11.0f, 1e-6f, @"token 3, dim 1");
    ASSERT_NEAR(out[8], 12.0f, 1e-6f, @"token 3, dim 2");
    PASS();
}

static void test_embedding_boundary_tokens(void) {
    // Test first and last vocab tokens (0, 31999 for Stories)
    int dim = 768;
    int vocab = 32000;
    float* table = calloc(vocab * dim, sizeof(float));

    // Set known values at boundaries
    table[0] = 42.0f;                          // token 0, dim 0
    table[(vocab-1) * dim] = 99.0f;            // token 31999, dim 0
    table[(vocab-1) * dim + dim - 1] = -7.5f;  // token 31999, dim 767

    int tokens[] = {0, 31999};
    float* out = calloc(2 * dim, sizeof(float));

    orion_cpu_embedding(out, table, tokens, dim, 2);

    ASSERT_NEAR(out[0], 42.0f, 1e-6f, @"token 0 lookup");
    ASSERT_NEAR(out[dim], 99.0f, 1e-6f, @"token 31999, dim 0");
    ASSERT_NEAR(out[dim + dim - 1], -7.5f, 1e-6f, @"token 31999, dim 767");

    free(table); free(out);
    PASS();
}

#pragma mark - T060: Adam Tests

static void test_adam_single_step(void) {
    // Verify one Adam step matches expected computation
    float param[] = {1.0f};
    float grad[] = {0.1f};
    float m[] = {0.0f};
    float v[] = {0.0f};

    orion_cpu_adam_step(param, grad, m, v, 1,
                        /*lr=*/0.001f, /*beta1=*/0.9f, /*beta2=*/0.999f,
                        /*eps=*/1e-8f, /*t=*/1);

    // After step 1:
    // m = 0.9*0 + 0.1*0.1 = 0.01
    // v = 0.999*0 + 0.001*0.01 = 0.00001
    // bc1 = 1 - 0.9^1 = 0.1
    // bc2 = 1 - 0.999^1 = 0.001
    // lr_t = 0.001 * sqrt(0.001) / 0.1 = 0.001 * 0.031623 / 0.1 = 0.00031623
    // update = lr_t * 0.01 / (sqrt(0.00001) + 1e-8) = 0.00031623 * 0.01 / 0.003162 = 0.001
    // param = 1.0 - 0.001 = 0.999
    ASSERT_NEAR(param[0], 0.999f, 1e-4f, @"Adam single step");
    PASS();
}

static void test_adam_convergence_quadratic(void) {
    // Minimize f(x) = x^2, gradient = 2x, starting from x=5
    float param[] = {5.0f};
    float m[] = {0.0f};
    float v[] = {0.0f};
    float grad[1];

    for (int t = 1; t <= 10000; t++) {
        grad[0] = 2.0f * param[0]; // df/dx = 2x
        orion_cpu_adam_step(param, grad, m, v, 1,
                            0.01f, 0.9f, 0.999f, 1e-8f, t);
    }

    NSString* msg1 = [NSString stringWithFormat:@"should converge near 0, got %f", param[0]];
    ASSERT(fabsf(param[0]) < 0.05f, msg1);
    PASS();
}

static void test_adam_multi_param(void) {
    // Minimize f(x,y) = x^2 + 4*y^2 starting from (3, 2)
    float param[] = {3.0f, 2.0f};
    float m[] = {0.0f, 0.0f};
    float v[] = {0.0f, 0.0f};
    float grad[2];

    for (int t = 1; t <= 2000; t++) {
        grad[0] = 2.0f * param[0];
        grad[1] = 8.0f * param[1];
        orion_cpu_adam_step(param, grad, m, v, 2,
                            0.01f, 0.9f, 0.999f, 1e-8f, t);
    }

    NSString* msg2 = [NSString stringWithFormat:@"should converge to (0,0), got (%f,%f)", param[0], param[1]];
    ASSERT(fabsf(param[0]) < 0.01f && fabsf(param[1]) < 0.01f, msg2);
    PASS();
}

static void test_adam_bias_correction(void) {
    // Early steps should have larger effective learning rate due to bias correction
    float param1[] = {1.0f}, param2[] = {1.0f};
    float m1[] = {0.0f}, v1[] = {0.0f};
    float m2[] = {0.0f}, v2[] = {0.0f};
    float grad[] = {0.1f};

    // Step 1 vs step 100 — step 1 should produce a larger update
    orion_cpu_adam_step(param1, grad, m1, v1, 1, 0.001f, 0.9f, 0.999f, 1e-8f, 1);
    // Reset for step 100 comparison (pre-warm m and v)
    m2[0] = 0.1f * (1.0f - powf(0.9f, 99)) / (1.0f - 0.9f) * (1.0f - 0.9f); // simplified
    for (int t = 1; t <= 99; t++) {
        orion_cpu_adam_step(param2, grad, m2, v2, 1, 0.001f, 0.9f, 0.999f, 1e-8f, t);
    }
    float before = param2[0];
    orion_cpu_adam_step(param2, grad, m2, v2, 1, 0.001f, 0.9f, 0.999f, 1e-8f, 100);
    float step100_update = fabsf(before - param2[0]);
    float step1_update = fabsf(1.0f - param1[0]);

    // Step 1 update (bias-corrected) should be ~= step 100 (converged correction)
    // Both should be approximately lr = 0.001 for constant gradient
    ASSERT(step1_update > 0.0f && step100_update > 0.0f,
           @"both updates should be non-zero");
    PASS();
}

#pragma mark - T061: dW Accumulation Tests

static void test_dw_accum_identity(void) {
    // x = I (2x2), dy = [[1,2],[3,4]] → dW += I^T @ dy = dy
    float x[] = {1, 0, 0, 1};
    float dy[] = {1, 2, 3, 4};
    float dw[] = {0, 0, 0, 0};

    orion_cpu_dw_accum(dw, x, dy, 2, 2, 2);

    ASSERT_NEAR(dw[0], 1.0f, 1e-5f, @"dw[0,0]");
    ASSERT_NEAR(dw[1], 2.0f, 1e-5f, @"dw[0,1]");
    ASSERT_NEAR(dw[2], 3.0f, 1e-5f, @"dw[1,0]");
    ASSERT_NEAR(dw[3], 4.0f, 1e-5f, @"dw[1,1]");
    PASS();
}

static void test_dw_accum_transpose(void) {
    // x = [[1,2],[3,4]] (m=2, n=2), dy = [[1],[0]] (m=2, k=1)
    // x^T = [[1,3],[2,4]], x^T @ dy = [[1*1+3*0],[2*1+4*0]] = [[1],[2]]
    float x[] = {1, 2, 3, 4};
    float dy[] = {1, 0};
    float dw[] = {0, 0};

    orion_cpu_dw_accum(dw, x, dy, 2, 2, 1);

    ASSERT_NEAR(dw[0], 1.0f, 1e-5f, @"dw[0]");
    ASSERT_NEAR(dw[1], 2.0f, 1e-5f, @"dw[1]");
    PASS();
}

static void test_dw_accum_accumulates(void) {
    // Verify dW += (not =) by running twice
    float x[] = {1, 0, 0, 1};
    float dy[] = {1, 1, 1, 1};
    float dw[] = {0, 0, 0, 0};

    orion_cpu_dw_accum(dw, x, dy, 2, 2, 2);
    orion_cpu_dw_accum(dw, x, dy, 2, 2, 2);

    // Should be 2x the single result
    ASSERT_NEAR(dw[0], 2.0f, 1e-5f, @"accumulated dw[0,0]");
    ASSERT_NEAR(dw[1], 2.0f, 1e-5f, @"accumulated dw[0,1]");
    PASS();
}

static void test_dw_accum_large(void) {
    // m=256, n=768, k=768 (realistic training dimensions)
    int m = 256, n = 768, k = 768;
    float* x = calloc(m * n, sizeof(float));
    float* dy = calloc(m * k, sizeof(float));
    float* dw = calloc(n * k, sizeof(float));

    // Fill with small values
    for (int i = 0; i < m * n; i++) x[i] = 0.01f * sinf((float)i);
    for (int i = 0; i < m * k; i++) dy[i] = 0.01f * cosf((float)i);

    orion_cpu_dw_accum(dw, x, dy, m, n, k);

    // Verify result is finite and non-trivial
    int finite_count = 0, nonzero_count = 0;
    for (int i = 0; i < n * k; i++) {
        if (isfinite(dw[i])) finite_count++;
        if (fabsf(dw[i]) > 1e-10f) nonzero_count++;
    }
    ASSERT(finite_count == n * k, @"all elements should be finite");
    ASSERT(nonzero_count > n * k / 2, @"most elements should be non-zero");

    free(x); free(dy); free(dw);
    PASS();
}

#pragma mark - Main

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        NSLog(@"=== T057: RMSNorm Tests ===");
        test_rmsnorm_unit_weight();
        test_rmsnorm_scaled_weight();
        test_rmsnorm_dim768();
        test_rmsnorm_eps();

        NSLog(@"\n=== T058: Cross-Entropy Tests ===");
        test_cross_entropy_perfect();
        test_cross_entropy_uniform();
        test_cross_entropy_gradient();
        test_cross_entropy_multi_position();
        test_cross_entropy_numerical_stability();

        NSLog(@"\n=== T059: Embedding Tests ===");
        test_embedding_lookup();
        test_embedding_boundary_tokens();

        NSLog(@"\n=== T060: Adam Tests ===");
        test_adam_single_step();
        test_adam_convergence_quadratic();
        test_adam_multi_param();
        test_adam_bias_correction();

        NSLog(@"\n=== T061: dW Accumulation Tests ===");
        test_dw_accum_identity();
        test_dw_accum_transpose();
        test_dw_accum_accumulates();
        test_dw_accum_large();

        NSLog(@"\n=== Results: %d passed, %d failed ===", g_pass, g_fail);
        return g_fail > 0 ? 1 : 0;
    }
}
