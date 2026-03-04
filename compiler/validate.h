// compiler/validate.h — T121: Graph validation
// Pure C.

#ifndef ORION_VALIDATE_H
#define ORION_VALIDATE_H

#include "graph.h"

typedef struct {
    bool valid;
    int error_node;          // Node index that caused the error (-1 if N/A)
    char message[256];       // Human-readable error message
} OrionValidationResult;

// Validate graph structure:
// - No cycles (DAG check)
// - All input references valid
// - No banned ops (concat)
// - All nodes have names
// - At least one output marked
OrionValidationResult orion_graph_validate(const OrionGraph* graph);

#endif // ORION_VALIDATE_H
