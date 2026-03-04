// compiler/mil_diff.h — T138: MIL text comparison tool
#ifndef ORION_MIL_DIFF_H
#define ORION_MIL_DIFF_H

#import <Foundation/Foundation.h>
#include <stdbool.h>

// Compare two MIL text strings, ignoring whitespace differences.
// Returns true if semantically equivalent, false if different.
// If different, writes a human-readable diff to diff_buf.
bool orion_mil_diff(NSString* a, NSString* b, char* diff_buf, int buf_size);

// Structural comparison: check that both MIL programs have:
// - Same number of ops (by type)
// - Same input/output signatures
// - Same weight references
// More tolerant than line-by-line diff.
bool orion_mil_structural_equiv(NSString* a, NSString* b, char* diff_buf, int buf_size);

#endif // ORION_MIL_DIFF_H
