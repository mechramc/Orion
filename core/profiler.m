#import "profiler.h"
#import <mach/mach.h>
#import <stdlib.h>
#import <string.h>
#import <stdio.h>

// T044-T045: Profiler core + formatted print

#define MAX_SAMPLES 4096

static struct {
    double decode_samples[MAX_SAMPLES];
    int decode_count;
    double eval_time_sum;
    double total_time_sum;
    double prefill_ms;
    size_t rss_start;
} g_prof;

void orion_prof_begin(void) {
    memset(&g_prof, 0, sizeof(g_prof));
    g_prof.rss_start = orion_get_rss();
}

void orion_prof_record_eval(double eval_ms, double total_ms) {
    g_prof.eval_time_sum += eval_ms;
    g_prof.total_time_sum += total_ms;
}

void orion_prof_record_decode(double decode_ms) {
    if (g_prof.decode_count < MAX_SAMPLES) {
        g_prof.decode_samples[g_prof.decode_count++] = decode_ms;
    }
}

static int cmp_double(const void* a, const void* b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

OrionPerfMetrics orion_prof_finish(int total_tokens __attribute__((unused)), double total_flops) {
    OrionPerfMetrics m = {0};

    m.prefill_ms = g_prof.prefill_ms;

    if (g_prof.decode_count > 0) {
        qsort(g_prof.decode_samples, g_prof.decode_count,
              sizeof(double), cmp_double);

        int p50_idx = g_prof.decode_count / 2;
        int p90_idx = (int)(g_prof.decode_count * 0.9);
        if (p90_idx >= g_prof.decode_count) p90_idx = g_prof.decode_count - 1;

        m.decode_ms_p50 = g_prof.decode_samples[p50_idx];
        m.decode_ms_p90 = g_prof.decode_samples[p90_idx];

        double total_decode_ms = 0;
        for (int i = 0; i < g_prof.decode_count; i++) {
            total_decode_ms += g_prof.decode_samples[i];
        }
        m.tokens_per_sec = 1000.0 * g_prof.decode_count / total_decode_ms;
    }

    if (total_flops > 0 && g_prof.total_time_sum > 0) {
        m.tflops = total_flops / (g_prof.total_time_sum * 1e-3) / 1e12;
    }

    m.ane_utilization = g_prof.total_time_sum > 0 ?
        g_prof.eval_time_sum / g_prof.total_time_sum : 0;

    m.rss_bytes = orion_get_rss();

    return m;
}

void orion_prof_print(const OrionPerfMetrics* m) {
    fprintf(stderr, "\n--- Orion Performance ---\n");
    fprintf(stderr, "  Prefill:       %8.1f ms\n", m->prefill_ms);
    fprintf(stderr, "  Decode p50:    %8.1f ms/tok\n", m->decode_ms_p50);
    fprintf(stderr, "  Decode p90:    %8.1f ms/tok\n", m->decode_ms_p90);
    fprintf(stderr, "  Throughput:    %8.1f tok/s\n", m->tokens_per_sec);
    if (m->tflops > 0)
        fprintf(stderr, "  TFLOPS:        %8.3f\n", m->tflops);
    if (m->ane_utilization > 0)
        fprintf(stderr, "  ANE util:      %7.1f%%\n", m->ane_utilization * 100);
    fprintf(stderr, "  RSS:           %6.1f MB\n", m->rss_bytes / (1024.0 * 1024.0));
    fprintf(stderr, "-------------------------\n");
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
