// compiler/pass_dce.c — T124: Dead code elimination

#include "pass_dce.h"
#include <string.h>

static void mark_live(const OrionGraph* g, int idx, bool* live) {
    if (idx < 0 || idx >= g->n_nodes) return;
    if (live[idx]) return;
    if (!g->nodes[idx].is_live) return;

    live[idx] = true;
    const OrionNode* n = &g->nodes[idx];
    for (int i = 0; i < n->n_inputs; i++) {
        mark_live(g, n->inputs[i], live);
    }
}

bool orion_pass_dce(OrionGraph* graph) {
    if (!graph) return false;

    bool live[ORION_MAX_NODES];
    memset(live, 0, sizeof(live));

    // Mark from outputs
    for (int i = 0; i < graph->n_outputs; i++) {
        mark_live(graph, graph->outputs[i].node_idx, live);
    }

    // Kill unmarked live nodes
    bool changed = false;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (graph->nodes[i].is_live && !live[i]) {
            graph->nodes[i].is_live = false;
            changed = true;
        }
    }
    return changed;
}
