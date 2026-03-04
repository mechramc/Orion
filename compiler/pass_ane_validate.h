// compiler/pass_ane_validate.h — T132: ANE constraint validation pass
#ifndef ORION_PASS_ANE_VALIDATE_H
#define ORION_PASS_ANE_VALIDATE_H

#include "graph.h"

typedef struct {
    int constraint_id;       // 1-11 per docs/ane_constraints.md
    int node_idx;            // Offending node (-1 if graph-level)
    char message[256];
} OrionANEConstraintError;

typedef struct {
    bool valid;
    int n_errors;
    OrionANEConstraintError errors[32]; // Max 32 constraint violations
} OrionANEValidationResult;

// Check all 11 ANE constraints from docs/ane_constraints.md:
// 1. No concat op
// 2. Output buffer uniformity (warning)
// 3. Alphabetical output ordering (warning)
// 4. Min tensor size >= 49KB equivalent
// 5. (Runtime) ~119 compile limit — not checkable here
// 6. (Runtime) SDPA causal masks — checked by decomposition
// 7. (Runtime) Weights baked — not checkable here
// 8. (Runtime) BLOBFILE offset — checked in codegen
// 9. (Runtime) milText must be NSData — not checkable here
// 10. No gelu op (must be decomposed)
// 11. Weight dict must be @{} — not checkable here
OrionANEValidationResult orion_pass_ane_validate(const OrionGraph* graph);

#endif // ORION_PASS_ANE_VALIDATE_H
