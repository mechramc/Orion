#import "iosurface_tensor.h"
#import <arm_neon.h>

// ANE tensor layout: fp16 [1, C, 1, S]
// IOSurface provides zero-copy shared memory between CPU and ANE.
// IOSurface uses a flat byte buffer; the 4D shape is conceptual (MIL assigns meaning).

#pragma mark - T012: Create

IOSurfaceRef orion_tensor_create(int channels, int seq_len) {
    if (channels <= 0 || seq_len <= 0) return NULL;

    // ANE uses fp16 (2 bytes per element). Total elements = C * S.
    size_t bytes = (size_t)channels * seq_len * sizeof(_Float16);

    // IOSurface properties: flat 1D byte buffer (width=bytes, height=1).
    // The 4D [1, C, 1, S] layout is defined by the MIL program, not the IOSurface.
    NSDictionary *props = @{
        (id)kIOSurfaceWidth:           @(bytes),
        (id)kIOSurfaceHeight:          @1,
        (id)kIOSurfaceBytesPerElement:  @1,
        (id)kIOSurfaceBytesPerRow:      @(bytes),
        (id)kIOSurfaceAllocSize:        @(bytes),
        (id)kIOSurfacePixelFormat:      @0
    };

    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

IOSurfaceRef orion_tensor_create_f32(int channels, int seq_len) {
    if (channels <= 0 || seq_len <= 0) return NULL;

    // fp32: 4 bytes per element
    size_t bytes = (size_t)channels * seq_len * sizeof(float);

    NSDictionary *props = @{
        (id)kIOSurfaceWidth:           @(bytes),
        (id)kIOSurfaceHeight:          @1,
        (id)kIOSurfaceBytesPerElement:  @1,
        (id)kIOSurfaceBytesPerRow:      @(bytes),
        (id)kIOSurfaceAllocSize:        @(bytes),
        (id)kIOSurfacePixelFormat:      @0
    };

    return IOSurfaceCreate((__bridge CFDictionaryRef)props);
}

#pragma mark - T013: Raw fp16 Read/Write

void orion_tensor_write(IOSurfaceRef surface, const void* data, size_t size) {
    if (!surface || !data || size == 0) return;
    IOSurfaceLock(surface, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(surface), data, size);
    IOSurfaceUnlock(surface, 0, NULL);
}

void orion_tensor_read(IOSurfaceRef surface, void* data, size_t size) {
    if (!surface || !data || size == 0) return;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(surface), size);
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

#pragma mark - T014: NEON fp16↔fp32 Conversion

void orion_tensor_write_f32(IOSurfaceRef surface, const float* data, int count) {
    if (!surface || !data || count <= 0) return;
    IOSurfaceLock(surface, 0, NULL);
    _Float16 *dst = (_Float16 *)IOSurfaceGetBaseAddress(surface);

    // NEON-accelerated f32 → f16 conversion (8 elements at a time)
    int i = 0;
    for (; i + 7 < count; i += 8) {
        float16x8_t h = vcombine_f16(
            vcvt_f16_f32(vld1q_f32(data + i)),
            vcvt_f16_f32(vld1q_f32(data + i + 4)));
        vst1q_f16((__fp16 *)(dst + i), h);
    }
    // Scalar tail
    for (; i < count; i++) {
        dst[i] = (_Float16)data[i];
    }

    IOSurfaceUnlock(surface, 0, NULL);
}

void orion_tensor_read_f32(IOSurfaceRef surface, float* data, int count) {
    if (!surface || !data || count <= 0) return;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *src = (const _Float16 *)IOSurfaceGetBaseAddress(surface);

    // NEON-accelerated f16 → f32 conversion (8 elements at a time)
    int i = 0;
    for (; i + 7 < count; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16 *)(src + i));
        vst1q_f32(data + i,     vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(data + i + 4, vcvt_f32_f16(vget_high_f16(h)));
    }
    // Scalar tail
    for (; i < count; i++) {
        data[i] = (float)src[i];
    }

    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

void orion_tensor_read_f32_direct(IOSurfaceRef surface, float* data, int count) {
    if (!surface || !data || count <= 0) return;
    IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(surface), (size_t)count * sizeof(float));
    IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
}

#pragma mark - Release

void orion_tensor_release(IOSurfaceRef surface) {
    if (surface) {
        CFRelease(surface);
    }
}
