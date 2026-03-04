// compiler/pass_cast.h — T127: Cast hoisting/elimination pass
#ifndef ORION_PASS_CAST_H
#define ORION_PASS_CAST_H

#include "graph.h"

// Eliminate redundant casts:
// - fp32->fp16 followed by fp16->fp32 → eliminate both
// - fp16->fp32 followed by fp32->fp16 → eliminate both
// ANE computes in fp16 internally, so internal casts are waste.
// Casts at graph boundaries (inputs/outputs) are preserved.
// Returns true if any casts were eliminated.
bool orion_pass_cast(OrionGraph* graph);

#endif // ORION_PASS_CAST_H
