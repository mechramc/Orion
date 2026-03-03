#import "data_loader.h"
#import <Foundation/Foundation.h>
#import <sys/mman.h>
#import <sys/stat.h>
#import <fcntl.h>
#import <unistd.h>

// T063: Pretokenized data loader.
// Karpathy's format: raw sequence of uint16 token ids (BOS-separated documents).

OrionDataLoader* orion_data_loader_open(const char* path, int seq_len) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;

    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return NULL;
    }

    size_t file_size = (size_t)st.st_size;
    if (file_size < (size_t)(seq_len + 1) * sizeof(uint16_t)) {
        close(fd);
        return NULL;
    }

    void* mapped = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
        close(fd);
        return NULL;
    }

    OrionDataLoader* dl = calloc(1, sizeof(OrionDataLoader));
    dl->data = (uint16_t*)mapped;
    dl->n_tokens = (int64_t)(file_size / sizeof(uint16_t));
    dl->seq_len = seq_len;
    dl->pos = 0;
    dl->fd = fd;
    dl->file_size = file_size;

    return dl;
}

bool orion_data_loader_next(OrionDataLoader* dl, int* input, int* target) {
    if (!dl) return false;

    // Need seq_len + 1 tokens (input + 1 shifted target)
    if (dl->pos + dl->seq_len + 1 > dl->n_tokens) {
        // Wrap around
        dl->pos = 0;
        return false;
    }

    // Copy uint16 → int32
    for (int i = 0; i < dl->seq_len; i++) {
        input[i]  = (int)dl->data[dl->pos + i];
        target[i] = (int)dl->data[dl->pos + i + 1];
    }

    dl->pos += dl->seq_len;
    return true;
}

void orion_data_loader_reset(OrionDataLoader* dl) {
    if (dl) dl->pos = 0;
}

int64_t orion_data_loader_num_samples(OrionDataLoader* dl) {
    if (!dl || dl->seq_len <= 0) return 0;
    return (dl->n_tokens - 1) / dl->seq_len;
}

void orion_data_loader_close(OrionDataLoader* dl) {
    if (!dl) return;
    if (dl->data) {
        munmap((void*)dl->data, dl->file_size);
    }
    if (dl->fd >= 0) {
        close(dl->fd);
    }
    free(dl);
}
