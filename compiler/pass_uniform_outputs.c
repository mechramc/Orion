// compiler/pass_uniform_outputs.c — T131: Output buffer uniformity

#include "pass_uniform_outputs.h"
#include <stdio.h>

int orion_outputs_max_channels(const OrionGraph* graph) {
    if (!graph) return 0;
    int max_ch = 0;
    for (int i = 0; i < graph->n_outputs; i++) {
        int idx = graph->outputs[i].node_idx;
        int ch = graph->nodes[idx].shape[1]; // [1, C, 1, S]
        if (ch > max_ch) max_ch = ch;
    }
    return max_ch;
}

bool orion_pass_uniform_outputs(OrionGraph* graph) {
    if (!graph || graph->n_outputs <= 1) return false;

    int max_ch = orion_outputs_max_channels(graph);
    bool needs_padding = false;

    for (int i = 0; i < graph->n_outputs; i++) {
        int idx = graph->outputs[i].node_idx;
        int ch = graph->nodes[idx].shape[1];
        if (ch != max_ch) {
            fprintf(stderr, "[UNIFORM OUTPUT] output '%s': %d channels, max is %d — pad IOSurface to %d\n",
                    graph->outputs[i].name, ch, max_ch, max_ch);
            needs_padding = true;
        }
    }

    return needs_padding;
}
