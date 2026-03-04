// compiler/pass_sram.c — T130: SRAM budget estimation

#include "pass_sram.h"
#include <stdio.h>

static int dtype_bytes(OrionDtype dtype) {
    switch (dtype) {
        case ORION_DTYPE_FP16: return 2;
        case ORION_DTYPE_FP32: return 4;
        case ORION_DTYPE_INT32: return 4;
        default: return 1;
    }
}

int64_t orion_node_tensor_bytes(const OrionNode* node) {
    if (!node) return 0;
    int64_t elements = 1;
    for (int i = 0; i < 4; i++) {
        if (node->shape[i] > 0) elements *= node->shape[i];
    }
    return elements * dtype_bytes(node->dtype);
}

bool orion_pass_sram(OrionGraph* graph) {
    if (!graph) return false;
    bool has_spill_risk = false;

    for (int i = 0; i < graph->n_nodes; i++) {
        OrionNode* n = &graph->nodes[i];
        if (!n->is_live) continue;
        if (n->op == ORION_OP_CONST || n->op == ORION_OP_INPUT) continue;

        int64_t bytes = orion_node_tensor_bytes(n);
        if (bytes > ORION_SRAM_BUDGET_BYTES) {
            fprintf(stderr, "[SRAM WARNING] node '%s': %lld bytes (%.1f MB) > 32 MB budget\n",
                    n->name, (long long)bytes, (double)bytes / (1024.0 * 1024.0));
            has_spill_risk = true;
        }
    }

    return has_spill_risk;
}
