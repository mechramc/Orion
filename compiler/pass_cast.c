// compiler/pass_cast.c — T127: Cast hoisting/elimination

#include "pass_cast.h"

// Replace all references to old_idx with new_idx
static void rewrite_uses(OrionGraph* g, int old_idx, int new_idx) {
    for (int i = 0; i < g->n_nodes; i++) {
        if (!g->nodes[i].is_live) continue;
        for (int j = 0; j < g->nodes[i].n_inputs; j++) {
            if (g->nodes[i].inputs[j] == old_idx) {
                g->nodes[i].inputs[j] = new_idx;
            }
        }
    }
    for (int i = 0; i < g->n_outputs; i++) {
        if (g->outputs[i].node_idx == old_idx) {
            g->outputs[i].node_idx = new_idx;
        }
    }
}

bool orion_pass_cast(OrionGraph* graph) {
    if (!graph) return false;
    bool changed = false;

    for (int i = 0; i < graph->n_nodes; i++) {
        OrionNode* n = &graph->nodes[i];
        if (!n->is_live || n->op != ORION_OP_CAST) continue;
        if (n->n_inputs < 1) continue;

        int inp_idx = n->inputs[0];
        OrionNode* inp = &graph->nodes[inp_idx];
        if (!inp->is_live || inp->op != ORION_OP_CAST) continue;

        // We have: inp=cast(X, A->B), n=cast(inp, B->C)
        // If A == C, the pair is a round-trip → eliminate both
        OrionDtype dtype_a = graph->nodes[inp->inputs[0]].dtype;
        OrionDtype dtype_c = n->attrs.cast_dtype;

        if (dtype_a == dtype_c) {
            // Round-trip cast: bypass both to the original input
            int original = inp->inputs[0];
            rewrite_uses(graph, i, original);
            n->is_live = false;
            // Don't kill inp yet — it might have other users
            changed = true;
        }
    }

    return changed;
}
