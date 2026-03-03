#import <Foundation/Foundation.h>
#import "kernels/training/data_loader.h"
#import <string.h>

static int g_pass = 0, g_fail = 0;

#define ASSERT(cond, msg) do { \
    if (!(cond)) { \
        NSLog(@"FAIL: %s — %@", __func__, msg); \
        g_fail++; return; \
    } \
} while(0)

#define PASS() do { NSLog(@"PASS: %s", __func__); g_pass++; } while(0)

static const char* TEST_DATA = "data/tinystories_test.bin";

static void test_open(void) {
    OrionDataLoader* dl = orion_data_loader_open(TEST_DATA, 64);
    ASSERT(dl != NULL, @"should open test data");
    ASSERT(dl->n_tokens == 1010, @"should have 1010 tokens");
    orion_data_loader_close(dl);
    PASS();
}

static void test_open_invalid(void) {
    OrionDataLoader* dl = orion_data_loader_open("/nonexistent/path.bin", 64);
    ASSERT(dl == NULL, @"should return NULL for invalid path");
    PASS();
}

static void test_next_batch(void) {
    OrionDataLoader* dl = orion_data_loader_open(TEST_DATA, 64);
    ASSERT(dl != NULL, @"should open");

    int input[64], target[64];
    bool ok = orion_data_loader_next(dl, input, target);
    ASSERT(ok, @"first batch should succeed");

    // First token is BOS=1, targets shifted by 1
    ASSERT(input[0] == 1, @"first input token should be BOS=1");
    ASSERT(target[0] == input[1], @"target[0] should equal input[1]");

    // Verify target = input shifted by 1
    for (int i = 0; i < 63; i++) {
        ASSERT(target[i] == input[i+1], @"target should be input shifted by 1");
    }

    orion_data_loader_close(dl);
    PASS();
}

static void test_multiple_batches(void) {
    OrionDataLoader* dl = orion_data_loader_open(TEST_DATA, 64);
    ASSERT(dl != NULL, @"should open");

    int input[64], target[64];
    int batch_count = 0;
    while (orion_data_loader_next(dl, input, target)) {
        batch_count++;
    }

    // 1010 tokens, seq_len=64, need seq_len+1=65 per batch
    // (1010-1)/64 = 15 full batches
    ASSERT(batch_count == 15, @"should have 15 batches");

    orion_data_loader_close(dl);
    PASS();
}

static void test_reset(void) {
    OrionDataLoader* dl = orion_data_loader_open(TEST_DATA, 64);
    ASSERT(dl != NULL, @"should open");

    int input1[64], target1[64];
    orion_data_loader_next(dl, input1, target1);

    // Read a few more batches
    int dummy_in[64], dummy_tgt[64];
    orion_data_loader_next(dl, dummy_in, dummy_tgt);
    orion_data_loader_next(dl, dummy_in, dummy_tgt);

    // Reset
    orion_data_loader_reset(dl);

    int input2[64], target2[64];
    orion_data_loader_next(dl, input2, target2);

    // First batch after reset should match first batch
    ASSERT(memcmp(input1, input2, 64 * sizeof(int)) == 0, @"reset should restart from beginning");

    orion_data_loader_close(dl);
    PASS();
}

static void test_no_oob(void) {
    // seq_len very large should still work (returns false, no crash)
    OrionDataLoader* dl = orion_data_loader_open(TEST_DATA, 2000);
    ASSERT(dl == NULL, @"should fail for seq_len > n_tokens");
    PASS();
}

static void test_num_samples(void) {
    OrionDataLoader* dl = orion_data_loader_open(TEST_DATA, 64);
    ASSERT(dl != NULL, @"should open");

    int64_t n = orion_data_loader_num_samples(dl);
    // (1010 - 1) / 64 = 15
    ASSERT(n == 15, @"should report 15 samples");

    orion_data_loader_close(dl);
    PASS();
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        NSLog(@"=== T063: Data Loader Tests ===");
        test_open();
        test_open_invalid();
        test_next_batch();
        test_multiple_batches();
        test_reset();
        test_no_oob();
        test_num_samples();

        NSLog(@"\n=== Results: %d passed, %d failed ===", g_pass, g_fail);
        return g_fail > 0 ? 1 : 0;
    }
}
