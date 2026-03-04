// compiler/topo.c — T123: Topological sort (Kahn's algorithm)

#include "topo.h"
#include <stdlib.h>
#include <string.h>

int* orion_topo_sort(const OrionGraph* graph, int* out_count) {
    if (!graph || !out_count) return NULL;

    int n = graph->n_nodes;
    *out_count = 0;

    // Compute in-degree: count total input edges per node.
    // add(x, x) has in_degree=2 because two input slots reference another node.
    int* in_degree = (int*)calloc(n, sizeof(int));
    for (int i = 0; i < n; i++) {
        if (!graph->nodes[i].is_live) continue;
        for (int j = 0; j < graph->nodes[i].n_inputs; j++) {
            int inp = graph->nodes[i].inputs[j];
            if (inp >= 0 && inp < n && graph->nodes[inp].is_live) {
                in_degree[i]++;
            }
        }
    }

    int* queue = (int*)malloc(n * sizeof(int));
    int* result = (int*)malloc(n * sizeof(int));
    int q_head = 0, q_tail = 0;
    int r_count = 0;

    for (int i = 0; i < n; i++) {
        if (graph->nodes[i].is_live && in_degree[i] == 0) {
            queue[q_tail++] = i;
        }
    }

    while (q_head < q_tail) {
        int cur = queue[q_head++];
        result[r_count++] = cur;

        // For every node that references cur, decrement in-degree once per reference.
        for (int i = 0; i < n; i++) {
            if (!graph->nodes[i].is_live) continue;
            int decrements = 0;
            for (int j = 0; j < graph->nodes[i].n_inputs; j++) {
                if (graph->nodes[i].inputs[j] == cur) {
                    decrements++;
                }
            }
            if (decrements > 0) {
                in_degree[i] -= decrements;
                if (in_degree[i] == 0) {
                    queue[q_tail++] = i;
                }
            }
        }
    }

    free(in_degree);
    free(queue);

    int live_count = 0;
    for (int i = 0; i < n; i++) {
        if (graph->nodes[i].is_live) live_count++;
    }

    if (r_count != live_count) {
        free(result);
        return NULL;
    }

    *out_count = r_count;
    return result;
}
