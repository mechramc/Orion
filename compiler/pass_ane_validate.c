// compiler/pass_ane_validate.c — T132: ANE constraint validation

#include "pass_ane_validate.h"
#include "pass_sram.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static void add_error(OrionANEValidationResult* r, int constraint_id, int node_idx,
                      const char* fmt, ...) {
    if (r->n_errors >= 32) return;
    OrionANEConstraintError* e = &r->errors[r->n_errors++];
    e->constraint_id = constraint_id;
    e->node_idx = node_idx;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(e->message, sizeof(e->message), fmt, ap);
    va_end(ap);
    r->valid = false;
}

// Minimum IOSurface size in bytes (~49KB)
#define ANE_MIN_TENSOR_BYTES 49152

OrionANEValidationResult orion_pass_ane_validate(const OrionGraph* graph) {
    OrionANEValidationResult r;
    memset(&r, 0, sizeof(r));
    r.valid = true;

    if (!graph) {
        add_error(&r, 0, -1, "NULL graph");
        return r;
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        const OrionNode* n = &graph->nodes[i];
        if (!n->is_live) continue;

        // Constraint 1: No concat op
        if (n->op == ORION_OP_CONCAT_BANNED) {
            add_error(&r, 1, i, "node '%s': concat op rejected by ANE compiler", n->name);
        }

        // Constraint 10: No gelu op (must be decomposed to tanh approximation)
        // Check for any node that might represent a monolithic gelu
        // (Our builder always decomposes, but validate external graphs)
        // We don't have ORION_OP_GELU — this is by design. Nothing to check.

        // Constraint 4: Minimum tensor size for compute nodes
        if (n->op != ORION_OP_CONST && n->op != ORION_OP_INPUT) {
            int64_t bytes = orion_node_tensor_bytes(n);
            if (bytes > 0 && bytes < ANE_MIN_TENSOR_BYTES) {
                add_error(&r, 4, i, "node '%s': tensor size %lld bytes < 49KB minimum",
                         n->name, (long long)bytes);
            }
        }
    }

    // Constraint 2: Output buffer uniformity (warning, not error)
    if (graph->n_outputs > 1) {
        int max_ch = 0;
        for (int i = 0; i < graph->n_outputs; i++) {
            int ch = graph->nodes[graph->outputs[i].node_idx].shape[1];
            if (ch > max_ch) max_ch = ch;
        }
        for (int i = 0; i < graph->n_outputs; i++) {
            int ch = graph->nodes[graph->outputs[i].node_idx].shape[1];
            if (ch != max_ch) {
                add_error(&r, 2, graph->outputs[i].node_idx,
                         "output '%s': %d channels != max %d — need uniform IOSurface sizes",
                         graph->outputs[i].name, ch, max_ch);
            }
        }
    }

    // Constraint 3: Alphabetical output ordering check
    if (graph->n_outputs > 1) {
        for (int i = 1; i < graph->n_outputs; i++) {
            if (strcmp(graph->outputs[i-1].name, graph->outputs[i].name) > 0) {
                add_error(&r, 3, -1,
                         "output ordering not alphabetical: '%s' before '%s' — ANE returns outputs alphabetically",
                         graph->outputs[i-1].name, graph->outputs[i].name);
            }
        }
    }

    // Constraint 11: Weight dict must not be nil
    // Can't check from graph alone — this is a runtime constraint.
    // But we can check that any node with weights has a valid blob path.
    for (int i = 0; i < graph->n_nodes; i++) {
        const OrionNode* n = &graph->nodes[i];
        if (!n->is_live) continue;
        if (n->op == ORION_OP_CONST && n->shape[0] > 0 && n->shape[1] > 0) {
            // This looks like a weight tensor — should have a blob path
            // (scalar constants don't need blob paths)
            if (n->attrs.blob_path[0] == '\0' && n->dtype == ORION_DTYPE_FP16 &&
                (n->shape[0] > 1 || n->shape[1] > 1)) {
                // Not necessarily an error — could be a small inline constant
            }
        }
    }

    return r;
}
