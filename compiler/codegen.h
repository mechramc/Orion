// compiler/codegen.h — T120: MIL code generation from OrionGraph
// ObjC (needs NSString for MIL text output).

#ifndef ORION_CODEGEN_H
#define ORION_CODEGEN_H

#import <Foundation/Foundation.h>
#include "graph.h"

// Generate MIL program text from a graph.
// Walks nodes in topological order and emits MIL text.
// func_name: function name in MIL (usually "main").
// Returns valid MIL program text or nil on error.
NSString* orion_codegen_mil(const OrionGraph* graph, const char* func_name);

#endif // ORION_CODEGEN_H
