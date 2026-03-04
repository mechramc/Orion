// compiler/kernel_adapter.h — T137: OrionKernel adapter for compiler frontends
// ObjC (bridges graph frontends to OrionKernel interface).

#ifndef ORION_KERNEL_ADAPTER_H
#define ORION_KERNEL_ADAPTER_H

#import "core/kernel.h"
#include "graph.h"

// Frontend function type: builds a graph from model config (3-arg: layer, bucket, cfg).
typedef OrionGraph* (*OrionFrontendFn)(int layer_idx, int bucket, const OrionModelConfig* cfg);

// Frontend function type for kernels that don't use bucket (2-arg: layer, cfg).
typedef OrionGraph* (*OrionFrontend2Fn)(int layer_idx, const OrionModelConfig* cfg);

// Direct API: generate MIL from a 3-arg frontend function.
// Builds graph -> validates -> optimizes -> codegen.
NSString* orion_kernel_adapter_generate_mil(OrionFrontendFn frontend,
                                             int layer_idx, int bucket,
                                             const OrionModelConfig* cfg);

// Direct API: generate MIL from a 2-arg frontend function (no bucket).
NSString* orion_kernel_adapter_generate_mil_2arg(OrionFrontend2Fn frontend,
                                                  int layer_idx,
                                                  const OrionModelConfig* cfg);

// Create an OrionKernel from a compiler frontend.
// The generate_mil implementation: call frontend -> optimize -> codegen -> return MIL string.
// This is a drop-in replacement for hand-written milgen functions.
OrionKernel orion_kernel_from_frontend(const char* name,
                                        OrionFrontendFn frontend,
                                        OrionWDictFn wdict,
                                        int n_inputs, int n_outputs);

#endif // ORION_KERNEL_ADAPTER_H
