#ifndef ORION_BUCKET_H
#define ORION_BUCKET_H

// T050: Bucket selection for ANE prefill.
// ANE programs are compiled for fixed sequence lengths.
// Given a prompt length, select the smallest bucket that fits.

/// Select the smallest bucket >= seq_len.
/// @param seq_len     Actual sequence length
/// @param buckets     Sorted array of available bucket sizes
/// @param num_buckets Number of buckets
/// @return Bucket size, or -1 if seq_len exceeds largest bucket.
int orion_select_bucket(int seq_len, const int* buckets, int num_buckets);

#endif // ORION_BUCKET_H
