// compiler/kernel_adapter.h — T137: OrionKernel adapter for compiler frontends
// ObjC (bridges graph frontends to OrionKernel interface).

#ifndef ORION_KERNEL_ADAPTER_H
#define ORION_KERNEL_ADAPTER_H

#import "core/kernel.h"
#include "graph.h"

// Frontend function type: builds a graph from model config.
typedef OrionGraph* (*OrionFrontendFn)(int layer_idx, int bucket, const OrionModelConfig* cfg);

// Create an OrionKernel from a compiler frontend.
// The generate_mil implementation: call frontend -> optimize -> codegen -> return MIL string.
// This is a drop-in replacement for hand-written milgen functions.
OrionKernel orion_kernel_from_frontend(const char* name,
                                        OrionFrontendFn frontend,
                                        OrionWDictFn wdict,
                                        int n_inputs, int n_outputs);

#endif // ORION_KERNEL_ADAPTER_H
