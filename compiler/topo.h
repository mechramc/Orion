// compiler/topo.h — T123: Topological sort
// Pure C.

#ifndef ORION_TOPO_H
#define ORION_TOPO_H

#include "graph.h"

// Returns node indices in valid topological execution order (Kahn's algorithm).
// Caller must free() the returned array.
// Sets *out_count to the number of nodes in the order.
// Returns NULL if the graph has a cycle.
int* orion_topo_sort(const OrionGraph* graph, int* out_count);

#endif // ORION_TOPO_H
