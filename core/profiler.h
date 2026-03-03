#ifndef ORION_PROFILER_H
#define ORION_PROFILER_H

#import <Foundation/Foundation.h>

/// Profiling utilities for Orion runtime.
/// Reports: latency (p50/p90), tokens/sec, TFLOPS, memory, ANE utilization.

typedef struct {
    double prefill_ms;
    double decode_ms_p50;
    double decode_ms_p90;
    double tokens_per_sec;
    double tflops;
    double ane_utilization;  // time in orion_eval / total wall time
    size_t rss_bytes;        // resident set size
} OrionPerfMetrics;

/// Start a profiling session (resets all accumulators).
void orion_prof_begin(void);

/// Record a single eval timing sample.
void orion_prof_record_eval(double eval_ms, double total_ms);

/// Record a decode token timing sample.
void orion_prof_record_decode(double decode_ms);

/// Finalize and compute aggregate metrics.
OrionPerfMetrics orion_prof_finish(int total_tokens, double total_flops);

/// Print metrics to stdout in a formatted table.
void orion_prof_print(const OrionPerfMetrics* m);

/// Get current process RSS in bytes.
size_t orion_get_rss(void);

#endif // ORION_PROFILER_H
