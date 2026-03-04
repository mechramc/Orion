// compiler/pass_identity.h — T125: Identity elimination pass
#ifndef ORION_PASS_IDENTITY_H
#define ORION_PASS_IDENTITY_H

#include "graph.h"

// Remove identity ops: cast(fp16->fp16), reshape to same shape, identity nodes.
// Rewires consumers to use the input directly.
// Returns true if any nodes were eliminated.
bool orion_pass_identity(OrionGraph* graph);

#endif // ORION_PASS_IDENTITY_H
