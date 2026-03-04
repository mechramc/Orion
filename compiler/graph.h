// compiler/graph.h — T117: OrionGraph data structures
// Single-level graph IR for ANE MIL compilation.
// Pure C — no ObjC runtime dependency.

#ifndef ORION_GRAPH_H
#define ORION_GRAPH_H

#include <stdint.h>
#include <stdbool.h>

// Maximum limits
#define ORION_MAX_INPUTS   8
#define ORION_MAX_OUTPUTS  8
#define ORION_MAX_NODES    4096
#define ORION_MAX_NAME     64
#define ORION_MAX_GRAPH_IO 16

// Operations — map 1:1 to MIL ops
typedef enum {
    ORION_OP_INPUT = 0,     // Graph input placeholder
    ORION_OP_CONST,         // Constant / weight reference
    ORION_OP_CONV1X1,       // 1x1 convolution (linear layer)
    ORION_OP_ADD,
    ORION_OP_SUB,
    ORION_OP_MUL,
    ORION_OP_MATMUL,
    ORION_OP_RESHAPE,
    ORION_OP_TRANSPOSE,
    ORION_OP_CAST,
    ORION_OP_RELU,
    ORION_OP_TANH,
    ORION_OP_SIGMOID,
    ORION_OP_SOFTMAX,
    ORION_OP_EXP,
    ORION_OP_POW,
    ORION_OP_REDUCE_SUM,
    ORION_OP_REDUCE_MEAN,
    ORION_OP_REDUCE_MAX,
    ORION_OP_NEG,
    ORION_OP_SQRT,
    ORION_OP_RSQRT,
    ORION_OP_CONCAT_BANNED, // Marked banned — validation rejects this
    ORION_OP_SPLIT,
    ORION_OP_PAD,
    ORION_OP_SLICE,
    ORION_OP_IDENTITY,
    ORION_OP_COUNT
} OrionOp;

// Data types
typedef enum {
    ORION_DTYPE_FP16 = 0,
    ORION_DTYPE_FP32,
    ORION_DTYPE_INT32,
    ORION_DTYPE_BOOL,
    ORION_DTYPE_STRING, // For cast dtype argument
} OrionDtype;

// Node attributes (union of all op-specific attributes)
typedef struct {
    // For conv: groups, strides, etc. (always 1x1 for ANE linear)
    // For reshape: target shape is in shape[]
    // For transpose: perm[4]
    int perm[4];
    // For reduce ops / softmax
    int axis;
    bool keep_dims;
    // For matmul
    bool transpose_x;
    bool transpose_y;
    // For const: weight blob path + scalar value
    char blob_path[256];
    uint64_t blob_offset;
    float scalar_val;
    // For cast: target dtype
    OrionDtype cast_dtype;
    // For conv: bias input index (-1 = no bias)
    int bias_input;
    // For slice_by_index
    int slice_begin[4];
    int slice_end[4];
} OrionAttrs;

// A single node in the graph
typedef struct {
    OrionOp op;
    char name[ORION_MAX_NAME];
    OrionDtype dtype;
    int shape[4];           // [batch, channels, height, width] — ANE layout [1, C, 1, S]
    int inputs[ORION_MAX_INPUTS];  // Indices into graph->nodes[]
    int n_inputs;
    OrionAttrs attrs;
    bool is_output;         // Marked as a graph output
    bool is_live;           // Used by DCE pass
} OrionNode;

// Named I/O for the graph
typedef struct {
    char name[ORION_MAX_NAME];
    int node_idx;           // Index into nodes[]
} OrionGraphIO;

// The graph
typedef struct {
    OrionNode nodes[ORION_MAX_NODES];
    int n_nodes;
    OrionGraphIO inputs[ORION_MAX_GRAPH_IO];
    int n_inputs;
    OrionGraphIO outputs[ORION_MAX_GRAPH_IO];
    int n_outputs;
} OrionGraph;

// Lifecycle
OrionGraph* orion_graph_create(void);
int orion_graph_add_node(OrionGraph* g, const OrionNode* node);
void orion_graph_free(OrionGraph* g);

// Utility
const char* orion_op_name(OrionOp op);
const char* orion_dtype_name(OrionDtype dtype);
const char* orion_dtype_mil_name(OrionDtype dtype);
void orion_graph_dump(const OrionGraph* g); // Debug print

#endif // ORION_GRAPH_H
