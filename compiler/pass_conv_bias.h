// compiler/pass_conv_bias.h — T126: Conv+bias fusion pass
#ifndef ORION_PASS_CONV_BIAS_H
#define ORION_PASS_CONV_BIAS_H

#include "graph.h"

// Pattern: conv1x1(x, w, no_bias) -> add(result, bias) => conv1x1(x, w, bias)
// ANE handles bias inside conv natively — saves a dispatch.
// Returns true if any fusions occurred.
bool orion_pass_conv_bias(OrionGraph* graph);

#endif // ORION_PASS_CONV_BIAS_H
