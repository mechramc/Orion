#ifndef ORION_DATA_LOADER_H
#define ORION_DATA_LOADER_H

#import <stdbool.h>
#import <stdint.h>
#import <stddef.h>

/// T063: Pretokenized data loader for TinyStories.
/// Reads Karpathy's binary format: sequence of uint16 token ids.
/// Produces (input, target) pairs where target = input shifted by 1.

typedef struct {
    uint16_t* data;     // Memory-mapped token data
    int64_t n_tokens;   // Total tokens in file
    int seq_len;        // Sequence length per sample
    int64_t pos;        // Current read position
    int fd;             // File descriptor for mmap
    size_t file_size;   // File size in bytes
} OrionDataLoader;

/// Open pretokenized data file. Returns NULL on failure.
OrionDataLoader* orion_data_loader_open(const char* path, int seq_len);

/// Get next batch of (input, target) pairs.
/// input: [seq_len] token ids
/// target: [seq_len] token ids (shifted by 1)
/// Returns false when data is exhausted (wraps around).
bool orion_data_loader_next(OrionDataLoader* dl, int* input, int* target);

/// Reset to beginning of data.
void orion_data_loader_reset(OrionDataLoader* dl);

/// Get number of available samples (non-overlapping).
int64_t orion_data_loader_num_samples(OrionDataLoader* dl);

/// Close and free.
void orion_data_loader_close(OrionDataLoader* dl);

#endif // ORION_DATA_LOADER_H
