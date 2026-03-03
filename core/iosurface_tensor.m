#import "iosurface_tensor.h"

// ANE tensor layout: fp16 [1, C, 1, S]
// IOSurface provides zero-copy shared memory between CPU and ANE

IOSurfaceRef orion_tensor_create(int channels, int seq_len) {
    // TODO(M0): Create IOSurface with correct properties for ANE
    // - Pixel format: fp16
    // - Layout: [1, channels, 1, seq_len]
    // - Alignment requirements for ANE DMA
    return NULL;
}

void orion_tensor_write(IOSurfaceRef surface, const void* data, size_t size) {
    if (!surface || !data) return;
    IOSurfaceLock(surface, 0, NULL);
    void* base = IOSurfaceGetBaseAddress(surface);
    memcpy(base, data, size);
    IOSurfaceUnlock(surface, 0, NULL);
}

void orion_tensor_read(IOSurfaceRef surface, void* data, size_t size) {
    if (!surface || !data) return;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    void* base = IOSurfaceGetBaseAddress(surface);
    memcpy(data, base, size);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

void orion_tensor_write_f32(IOSurfaceRef surface, const float* data, int count) {
    // TODO(M1): Convert f32 → fp16 and write
    // Use vImageConvert_PlanarFtoPlanar16F or manual conversion
}

void orion_tensor_read_f32(IOSurfaceRef surface, float* data, int count) {
    // TODO(M1): Read fp16 and convert → f32
}

void orion_tensor_release(IOSurfaceRef surface) {
    if (surface) {
        CFRelease(surface);
    }
}
