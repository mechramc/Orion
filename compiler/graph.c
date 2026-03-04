// compiler/graph.c — T117: OrionGraph data structures

#include "graph.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

OrionGraph* orion_graph_create(void) {
    OrionGraph* g = (OrionGraph*)calloc(1, sizeof(OrionGraph));
    return g;
}

int orion_graph_add_node(OrionGraph* g, const OrionNode* node) {
    if (!g || !node || g->n_nodes >= ORION_MAX_NODES) return -1;
    int idx = g->n_nodes;
    g->nodes[idx] = *node;
    g->nodes[idx].is_live = true;
    g->n_nodes++;
    return idx;
}

void orion_graph_free(OrionGraph* g) {
    free(g);
}

static const char* s_op_names[] = {
    [ORION_OP_INPUT]         = "input",
    [ORION_OP_CONST]         = "const",
    [ORION_OP_CONV1X1]       = "conv",
    [ORION_OP_ADD]           = "add",
    [ORION_OP_SUB]           = "sub",
    [ORION_OP_MUL]           = "mul",
    [ORION_OP_MATMUL]        = "matmul",
    [ORION_OP_RESHAPE]       = "reshape",
    [ORION_OP_TRANSPOSE]     = "transpose",
    [ORION_OP_CAST]          = "cast",
    [ORION_OP_RELU]          = "relu",
    [ORION_OP_TANH]          = "tanh",
    [ORION_OP_SIGMOID]       = "sigmoid",
    [ORION_OP_SOFTMAX]       = "softmax",
    [ORION_OP_EXP]           = "exp",
    [ORION_OP_POW]           = "pow",
    [ORION_OP_REDUCE_SUM]    = "reduce_sum",
    [ORION_OP_REDUCE_MEAN]   = "reduce_mean",
    [ORION_OP_REDUCE_MAX]    = "reduce_max",
    [ORION_OP_NEG]           = "neg",
    [ORION_OP_SQRT]          = "sqrt",
    [ORION_OP_RSQRT]         = "rsqrt",
    [ORION_OP_CONCAT_BANNED] = "concat_BANNED",
    [ORION_OP_SPLIT]         = "split",
    [ORION_OP_PAD]           = "pad",
    [ORION_OP_SLICE]         = "slice_by_index",
    [ORION_OP_IDENTITY]      = "identity",
};

const char* orion_op_name(OrionOp op) {
    if (op < 0 || op >= ORION_OP_COUNT) return "unknown";
    return s_op_names[op];
}

static const char* s_dtype_names[] = {
    [ORION_DTYPE_FP16]   = "fp16",
    [ORION_DTYPE_FP32]   = "fp32",
    [ORION_DTYPE_INT32]  = "int32",
    [ORION_DTYPE_BOOL]   = "bool",
    [ORION_DTYPE_STRING] = "string",
};

const char* orion_dtype_name(OrionDtype dtype) {
    if (dtype < 0 || dtype > ORION_DTYPE_STRING) return "unknown";
    return s_dtype_names[dtype];
}

const char* orion_dtype_mil_name(OrionDtype dtype) {
    return orion_dtype_name(dtype); // Same for MIL
}

void orion_graph_dump(const OrionGraph* g) {
    if (!g) return;
    printf("OrionGraph: %d nodes, %d inputs, %d outputs\n",
           g->n_nodes, g->n_inputs, g->n_outputs);
    for (int i = 0; i < g->n_nodes; i++) {
        const OrionNode* n = &g->nodes[i];
        if (!n->is_live) continue;
        printf("  [%3d] %-16s %-8s [%d,%d,%d,%d] \"%s\"",
               i, orion_op_name(n->op), orion_dtype_name(n->dtype),
               n->shape[0], n->shape[1], n->shape[2], n->shape[3], n->name);
        if (n->n_inputs > 0) {
            printf("  inputs=[");
            for (int j = 0; j < n->n_inputs; j++) {
                if (j > 0) printf(",");
                printf("%d", n->inputs[j]);
            }
            printf("]");
        }
        if (n->is_output) printf(" [OUTPUT]");
        printf("\n");
    }
}
