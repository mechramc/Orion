// compiler/pass_sram.h — T130: SRAM budget estimation pass
#ifndef ORION_PASS_SRAM_H
#define ORION_PASS_SRAM_H

#include "graph.h"

// Default SRAM budget (32MB for M4 ANE)
#define ORION_SRAM_BUDGET_BYTES (32 * 1024 * 1024)

// Estimate per-node intermediate tensor sizes.
// Annotates nodes where working set exceeds SRAM budget.
// Warning-only pass — doesn't transform the graph.
// Returns true if any nodes exceed the budget (spill risk).
bool orion_pass_sram(OrionGraph* graph);

// Get estimated tensor size in bytes for a node's output.
int64_t orion_node_tensor_bytes(const OrionNode* node);

#endif // ORION_PASS_SRAM_H
