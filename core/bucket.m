#import "bucket.h"

// T050: Bucket selection

int orion_select_bucket(int seq_len, const int* buckets, int num_buckets) {
    if (seq_len <= 0 || !buckets || num_buckets <= 0) return -1;
    for (int i = 0; i < num_buckets; i++) {
        if (buckets[i] >= seq_len) return buckets[i];
    }
    return -1;  // exceeds largest bucket
}
