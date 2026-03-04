// compiler/pipeline.c — T128: Optimization pass pipeline

#include "pipeline.h"
#include "pass_identity.h"
#include "pass_cast.h"
#include "pass_conv_bias.h"
#include "pass_dce.h"
#include "pass_sram.h"
#include "pass_uniform_outputs.h"
#include "pass_ane_validate.h"

#define MAX_ITERATIONS 20

void orion_pipeline_optimize(OrionGraph* graph) {
    if (!graph) return;

    for (int iter = 0; iter < MAX_ITERATIONS; iter++) {
        bool changed = false;
        changed |= orion_pass_identity(graph);
        changed |= orion_pass_cast(graph);
        changed |= orion_pass_conv_bias(graph);
        changed |= orion_pass_dce(graph);
        if (!changed) break;
    }
}

void orion_pipeline_ane_passes(OrionGraph* graph) {
    if (!graph) return;
    orion_pass_sram(graph);
    orion_pass_uniform_outputs(graph);
    // ANE constraint validation is separate — caller checks result
}
