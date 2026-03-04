// compiler/pass_dce.h — T124: Dead code elimination pass
#ifndef ORION_PASS_DCE_H
#define ORION_PASS_DCE_H

#include "graph.h"

// Mark outputs, walk backwards marking live nodes, remove unmarked.
// Returns true if any nodes were eliminated.
bool orion_pass_dce(OrionGraph* graph);

#endif // ORION_PASS_DCE_H
