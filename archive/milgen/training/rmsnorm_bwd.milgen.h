#ifndef ORION_RMSNORM_BWD_H
#define ORION_RMSNORM_BWD_H

#import "../../core/mil_builder.h"
#import "../../core/ane_runtime.h"

// MIL generator for RMSNorm backward on ANE.
// Optional offload from CPU — provides ~1x speedup (neutral) but
// reduces CPU contention when overlapping with dW cblas_sgemm.

// TODO(M3): Implement
NSString* orion_milgen_rmsnorm_bwd(int layer_idx, int dim);

#endif // ORION_RMSNORM_BWD_H
