// compiler/model_config.h — compatibility include for compiler callers
#ifndef ORION_COMPILER_MODEL_CONFIG_H
#define ORION_COMPILER_MODEL_CONFIG_H

// OrionModelConfig has one canonical pure-C definition. Do not mirror it here:
// runtime and compiler code must share the same ABI layout.
#include "../core/model_config.h"

#endif // ORION_COMPILER_MODEL_CONFIG_H
