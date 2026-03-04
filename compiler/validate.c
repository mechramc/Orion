// compiler/validate.c — T121: Graph validation

#include "validate.h"
#include "topo.h"
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>

static OrionValidationResult ok_result(void) {
    OrionValidationResult r;
    r.valid = true;
    r.error_node = -1;
    r.message[0] = '\0';
    return r;
}

static OrionValidationResult err(int node, const char* fmt, ...) {
    OrionValidationResult r;
    r.valid = false;
    r.error_node = node;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(r.message, sizeof(r.message), fmt, ap);
    va_end(ap);
    return r;
}

OrionValidationResult orion_graph_validate(const OrionGraph* graph) {
    if (!graph) return err(-1, "NULL graph");
    if (graph->n_nodes == 0) return err(-1, "empty graph");

    // Check: at least one output
    if (graph->n_outputs == 0)
        return err(-1, "no outputs marked");

    // Check each node
    for (int i = 0; i < graph->n_nodes; i++) {
        const OrionNode* n = &graph->nodes[i];
        if (!n->is_live) continue;

        // Name required
        if (n->name[0] == '\0')
            return err(i, "node %d has no name", i);

        // No banned ops
        if (n->op == ORION_OP_CONCAT_BANNED)
            return err(i, "node '%s': concat op is banned on ANE (use multi-output instead)", n->name);

        // Input references valid
        for (int j = 0; j < n->n_inputs; j++) {
            int inp = n->inputs[j];
            if (inp < 0 || inp >= graph->n_nodes)
                return err(i, "node '%s': input[%d]=%d out of range [0,%d)",
                          n->name, j, inp, graph->n_nodes);
            if (!graph->nodes[inp].is_live)
                return err(i, "node '%s': input[%d] references dead node %d",
                          n->name, j, inp);
        }
    }

    // Check output nodes exist and are live
    for (int i = 0; i < graph->n_outputs; i++) {
        int idx = graph->outputs[i].node_idx;
        if (idx < 0 || idx >= graph->n_nodes)
            return err(-1, "output '%s' references invalid node %d",
                      graph->outputs[i].name, idx);
        if (!graph->nodes[idx].is_live)
            return err(idx, "output '%s' references dead node", graph->outputs[i].name);
    }

    // Cycle detection via topological sort
    int count = 0;
    int* order = orion_topo_sort(graph, &count);
    if (!order)
        return err(-1, "graph contains a cycle");
    free(order);

    return ok_result();
}
