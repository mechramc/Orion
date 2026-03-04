// compiler/pass_uniform_outputs.h — T131: Output buffer uniformity pass
#ifndef ORION_PASS_UNIFORM_OUTPUTS_H
#define ORION_PASS_UNIFORM_OUTPUTS_H

#include "graph.h"

// For multi-output programs: check if all outputs have the same channel count.
// ANE constraint #2: all output IOSurfaces must be the same allocation size.
// Returns true if outputs need padding (caller should pad IOSurfaces).
// Does NOT transform the graph — just reports the required uniform size.
bool orion_pass_uniform_outputs(OrionGraph* graph);

// Get the maximum channel count across all outputs.
int orion_outputs_max_channels(const OrionGraph* graph);

#endif // ORION_PASS_UNIFORM_OUTPUTS_H
