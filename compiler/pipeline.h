// compiler/pipeline.h — T128: Optimization pass pipeline
#ifndef ORION_PIPELINE_H
#define ORION_PIPELINE_H

#include "graph.h"

// Run all optimization passes in fixed order until fixpoint.
// Order: identity -> cast -> conv_bias -> DCE
// Iterates until no pass reports changes.
void orion_pipeline_optimize(OrionGraph* graph);

// Run only ANE-specific passes (validation + constraints).
// Call after optimize.
void orion_pipeline_ane_passes(OrionGraph* graph);

#endif // ORION_PIPELINE_H
