#ifndef ORION_IOSURFACE_TENSOR_H
#define ORION_IOSURFACE_TENSOR_H

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

/// Create an IOSurface-backed fp16 tensor in ANE layout [1, C, 1, S].
/// @param channels Number of channels (C dimension)
/// @param seq_len  Sequence length (S dimension)
/// @return IOSurface reference, or NULL on failure. Caller must CFRelease.
IOSurfaceRef orion_tensor_create(int channels, int seq_len);

/// Create an IOSurface-backed fp32 tensor in ANE layout [1, C, 1, S].
/// Used for training kernel outputs that cast to fp32 for gradient precision.
IOSurfaceRef orion_tensor_create_f32(int channels, int seq_len);

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

/// Read fp32 data directly from an IOSurface (no conversion).
/// Used with orion_tensor_create_f32 output surfaces.
void orion_tensor_read_f32_direct(IOSurfaceRef surface, float* data, int count);

/// Copy fp16 data from one IOSurface into another at a channel offset.
/// Used to assemble backward kernel inputs from forward tap outputs.
/// dst must be pre-allocated with enough space.
/// @param dst Destination IOSurface
/// @param dst_ch_offset Channel offset in destination (element offset = dst_ch_offset * seq_len)
/// @param src Source IOSurface (full copy)
/// @param channels Number of channels in source
/// @param seq_len Sequence length
void orion_tensor_copy_into(IOSurfaceRef dst, int dst_ch_offset,
                             IOSurfaceRef src, int channels, int seq_len);

/// Write fp32 data as fp16 into an IOSurface at a channel offset.
/// @param dst Destination IOSurface
/// @param dst_ch_offset Channel offset in destination
/// @param data Source fp32 data
/// @param channels Number of channels to write
/// @param seq_len Sequence length
void orion_tensor_write_f32_at(IOSurfaceRef dst, int dst_ch_offset,
                                const float* data, int channels, int seq_len);

/// Release an IOSurface tensor.
void orion_tensor_release(IOSurfaceRef surface);

#endif // ORION_IOSURFACE_TENSOR_H
