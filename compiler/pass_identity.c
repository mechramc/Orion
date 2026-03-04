// compiler/pass_identity.c — T125: Identity elimination

#include "pass_identity.h"
#include <string.h>

// Replace all references to old_idx with new_idx in the graph
static void rewrite_uses(OrionGraph* g, int old_idx, int new_idx) {
    for (int i = 0; i < g->n_nodes; i++) {
        if (!g->nodes[i].is_live) continue;
        OrionNode* n = &g->nodes[i];
        for (int j = 0; j < n->n_inputs; j++) {
            if (n->inputs[j] == old_idx) {
                n->inputs[j] = new_idx;
            }
        }
    }
    // Also rewrite output references
    for (int i = 0; i < g->n_outputs; i++) {
        if (g->outputs[i].node_idx == old_idx) {
            g->outputs[i].node_idx = new_idx;
        }
    }
}

static bool is_same_shape(const int a[4], const int b[4]) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

bool orion_pass_identity(OrionGraph* graph) {
    if (!graph) return false;
    bool changed = false;

    for (int i = 0; i < graph->n_nodes; i++) {
        OrionNode* n = &graph->nodes[i];
        if (!n->is_live) continue;

        bool eliminate = false;
        int bypass_input = -1;

        switch (n->op) {
            case ORION_OP_IDENTITY:
                eliminate = true;
                bypass_input = n->inputs[0];
                break;

            case ORION_OP_CAST:
                // Cast to same dtype → identity
                if (n->n_inputs >= 1 && n->attrs.cast_dtype == graph->nodes[n->inputs[0]].dtype) {
                    eliminate = true;
                    bypass_input = n->inputs[0];
                }
                break;

            case ORION_OP_RESHAPE:
                // Reshape to same shape → identity
                if (n->n_inputs >= 2) {
                    int inp = n->inputs[1]; // input tensor
                    if (is_same_shape(n->shape, graph->nodes[inp].shape)) {
                        eliminate = true;
                        bypass_input = inp;
                    }
                }
                break;

            case ORION_OP_TRANSPOSE: {
                // Identity permutation [0,1,2,3]
                if (n->attrs.perm[0] == 0 && n->attrs.perm[1] == 1 &&
                    n->attrs.perm[2] == 2 && n->attrs.perm[3] == 3) {
                    eliminate = true;
                    bypass_input = n->inputs[1]; // input tensor
                }
                break;
            }

            default:
                break;
        }

        if (eliminate && bypass_input >= 0) {
            rewrite_uses(graph, i, bypass_input);
            n->is_live = false;
            changed = true;
        }
    }

    return changed;
}
