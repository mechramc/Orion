// compiler/pass_conv_bias.c — T126: Conv+bias fusion

#include "pass_conv_bias.h"
#include <string.h>

// Check if a node is a bias-shaped constant: [1, C, 1, 1]
static bool is_bias_const(const OrionGraph* g, int idx) {
    const OrionNode* n = &g->nodes[idx];
    if (!n->is_live) return false;
    if (n->op != ORION_OP_CONST) return false;
    if (n->shape[0] != 1 || n->shape[2] != 1 || n->shape[3] != 1) return false;
    return n->attrs.blob_path[0] != '\0'; // Must be a weight reference
}

// Count how many live nodes use idx as an input
static int count_uses(const OrionGraph* g, int idx) {
    int count = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        if (!g->nodes[i].is_live) continue;
        for (int j = 0; j < g->nodes[i].n_inputs; j++) {
            if (g->nodes[i].inputs[j] == idx) count++;
        }
    }
    return count;
}

bool orion_pass_conv_bias(OrionGraph* graph) {
    if (!graph) return false;
    bool changed = false;

    for (int i = 0; i < graph->n_nodes; i++) {
        OrionNode* add_node = &graph->nodes[i];
        if (!add_node->is_live || add_node->op != ORION_OP_ADD) continue;
        if (add_node->n_inputs != 2) continue;

        // Pattern: add(conv_result, bias) or add(bias, conv_result)
        int conv_idx = -1, bias_idx = -1;

        for (int side = 0; side < 2; side++) {
            int a = add_node->inputs[side];
            int b = add_node->inputs[1 - side];
            if (a < 0 || a >= graph->n_nodes) continue;
            if (b < 0 || b >= graph->n_nodes) continue;

            OrionNode* maybe_conv = &graph->nodes[a];
            if (maybe_conv->is_live && maybe_conv->op == ORION_OP_CONV1X1 &&
                maybe_conv->attrs.bias_input < 0 && // No existing bias
                is_bias_const(graph, b) &&
                count_uses(graph, a) == 1) { // Conv only used by this add
                conv_idx = a;
                bias_idx = b;
                break;
            }
        }

        if (conv_idx < 0) continue;

        // Fuse: add bias to conv
        OrionNode* conv = &graph->nodes[conv_idx];
        conv->attrs.bias_input = bias_idx;
        conv->inputs[conv->n_inputs] = bias_idx;
        conv->n_inputs++;

        // Copy the add's name to conv (so output references work)
        // Actually, rewrite uses of the add node to point to conv
        for (int j = 0; j < graph->n_nodes; j++) {
            if (!graph->nodes[j].is_live) continue;
            for (int k = 0; k < graph->nodes[j].n_inputs; k++) {
                if (graph->nodes[j].inputs[k] == i) {
                    graph->nodes[j].inputs[k] = conv_idx;
                }
            }
        }
        for (int j = 0; j < graph->n_outputs; j++) {
            if (graph->outputs[j].node_idx == i) {
                graph->outputs[j].node_idx = conv_idx;
            }
        }

        // Rename conv to match add's output name
        strncpy(conv->name, add_node->name, ORION_MAX_NAME - 1);

        add_node->is_live = false;
        changed = true;
    }

    return changed;
}
