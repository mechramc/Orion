#import "profiler.h"
#import <mach/mach.h>

// TODO(M1): Implement profiling utilities
// - Use mach_absolute_time for high-res timing
// - Collect samples in a ring buffer for p50/p90 percentile computation
// - Compute TFLOPS from op counts and wall time

void orion_prof_begin(void) {
    // TODO: Reset all accumulators and sample buffers
}

void orion_prof_record_eval(double eval_ms, double total_ms) {
    // TODO: Accumulate eval time fraction for ANE utilization
}

void orion_prof_record_decode(double decode_ms) {
    // TODO: Add to decode latency samples for percentile computation
}

OrionPerfMetrics orion_prof_finish(int total_tokens, double total_flops) {
    // TODO: Compute aggregate metrics
    return (OrionPerfMetrics){0};
}

void orion_prof_print(const OrionPerfMetrics* m) {
    // TODO: Formatted table output matching ANEgpt style
}

size_t orion_get_rss(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &count) == KERN_SUCCESS) {
        return info.resident_size;
    }
    return 0;
}
