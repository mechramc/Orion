#ifndef ORION_IOSURFACE_TENSOR_H
#define ORION_IOSURFACE_TENSOR_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

/// Create an IOSurface-backed fp16 tensor in ANE layout [1, C, 1, S].
/// @param channels Number of channels (C dimension)
/// @param seq_len  Sequence length (S dimension)
/// @return IOSurface reference, or NULL on failure. Caller must CFRelease.
IOSurfaceRef orion_tensor_create(int channels, int seq_len);

/// Copy fp16 data into an IOSurface tensor.
/// @param surface Target IOSurface
/// @param data    Source fp16 data (row-major)
/// @param size    Size in bytes
void orion_tensor_write(IOSurfaceRef surface, const void* data, size_t size);

/// Copy fp16 data from an IOSurface tensor.
/// @param surface Source IOSurface
/// @param data    Destination buffer
/// @param size    Size in bytes
void orion_tensor_read(IOSurfaceRef surface, void* data, size_t size);

/// Convert float32 array to fp16 and write to IOSurface.
void orion_tensor_write_f32(IOSurfaceRef surface, const float* data, int count);

/// Read IOSurface fp16 data and convert to float32.
void orion_tensor_read_f32(IOSurfaceRef surface, float* data, int count);

/// Release an IOSurface tensor.
void orion_tensor_release(IOSurfaceRef surface);

#endif // ORION_IOSURFACE_TENSOR_H
