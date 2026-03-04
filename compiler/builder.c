// compiler/builder.c — T118: Graph builder API

#include "builder.h"
#include <string.h>
#include <stdio.h>
#include <math.h>

// Helper: create a node with basic fields
static OrionNode make_node(OrionOp op, const char* name, OrionDtype dtype, int shape[4]) {
    OrionNode n;
    memset(&n, 0, sizeof(n));
    n.op = op;
    n.dtype = dtype;
    n.is_live = true;
    if (name) snprintf(n.name, ORION_MAX_NAME, "%s", name);
    if (shape) memcpy(n.shape, shape, 4 * sizeof(int));
    return n;
}

int orion_gb_input(OrionGraph* g, const char* name, OrionDtype dtype, int shape[4]) {
    OrionNode n = make_node(ORION_OP_INPUT, name, dtype, shape);
    int idx = orion_graph_add_node(g, &n);
    if (idx >= 0 && g->n_inputs < ORION_MAX_GRAPH_IO) {
        snprintf(g->inputs[g->n_inputs].name, ORION_MAX_NAME, "%s", name);
        g->inputs[g->n_inputs].node_idx = idx;
        g->n_inputs++;
    }
    return idx;
}

int orion_gb_const_scalar(OrionGraph* g, const char* name, OrionDtype dtype, float value) {
    OrionNode n = make_node(ORION_OP_CONST, name, dtype, NULL);
    n.attrs.scalar_val = value;
    return orion_graph_add_node(g, &n);
}

int orion_gb_const_weight(OrionGraph* g, const char* name, OrionDtype dtype,
                          int shape[4], const char* blob_path, uint64_t offset) {
    OrionNode n = make_node(ORION_OP_CONST, name, dtype, shape);
    if (blob_path) snprintf(n.attrs.blob_path, sizeof(n.attrs.blob_path), "%s", blob_path);
    n.attrs.blob_offset = offset;
    return orion_graph_add_node(g, &n);
}

int orion_gb_const_int32(OrionGraph* g, const char* name, int shape[4],
                         const int* values, int n_values) {
    OrionNode n = make_node(ORION_OP_CONST, name, ORION_DTYPE_INT32, shape);
    // Store small int arrays in the scalar_val and perm fields
    if (n_values <= 4) {
        for (int i = 0; i < n_values; i++) n.attrs.perm[i] = values[i];
    }
    return orion_graph_add_node(g, &n);
}

int orion_gb_const_bool(OrionGraph* g, const char* name, bool value) {
    OrionNode n = make_node(ORION_OP_CONST, name, ORION_DTYPE_BOOL, NULL);
    n.attrs.scalar_val = value ? 1.0f : 0.0f;
    return orion_graph_add_node(g, &n);
}

int orion_gb_conv1x1(OrionGraph* g, int input, int weight, int bias, const char* name,
                     int out_channels, int seq) {
    int shape[4] = {1, out_channels, 1, seq};
    OrionNode n = make_node(ORION_OP_CONV1X1, name, ORION_DTYPE_FP16, shape);
    n.inputs[0] = input;
    n.inputs[1] = weight;
    n.n_inputs = 2;
    n.attrs.bias_input = bias; // -1 for no bias
    if (bias >= 0) {
        n.inputs[2] = bias;
        n.n_inputs = 3;
    }
    return orion_graph_add_node(g, &n);
}

// Helper for binary ops — infer shape from first non-scalar input
static int binary_op(OrionGraph* g, OrionOp op, int a, int b, const char* name) {
    const int* shape = g->nodes[a].shape;
    // If first operand is scalar (all zeros), use second operand's shape
    if (shape[0] == 0 && shape[1] == 0 && shape[2] == 0 && shape[3] == 0) {
        shape = g->nodes[b].shape;
    }
    OrionNode n = make_node(op, name, g->nodes[a].dtype, (int*)shape);
    n.inputs[0] = a;
    n.inputs[1] = b;
    n.n_inputs = 2;
    return orion_graph_add_node(g, &n);
}

int orion_gb_add(OrionGraph* g, int a, int b, const char* name) {
    return binary_op(g, ORION_OP_ADD, a, b, name);
}
int orion_gb_sub(OrionGraph* g, int a, int b, const char* name) {
    return binary_op(g, ORION_OP_SUB, a, b, name);
}
int orion_gb_mul(OrionGraph* g, int a, int b, const char* name) {
    return binary_op(g, ORION_OP_MUL, a, b, name);
}

int orion_gb_matmul(OrionGraph* g, int a, int b, bool transpose_x, bool transpose_y,
                    const char* name, int out_shape[4]) {
    OrionNode n = make_node(ORION_OP_MATMUL, name, g->nodes[a].dtype, out_shape);
    n.inputs[0] = a;
    n.inputs[1] = b;
    n.n_inputs = 2;
    n.attrs.transpose_x = transpose_x;
    n.attrs.transpose_y = transpose_y;
    return orion_graph_add_node(g, &n);
}

int orion_gb_reshape(OrionGraph* g, int input, int shape_node, const char* name,
                     int out_shape[4]) {
    OrionNode n = make_node(ORION_OP_RESHAPE, name, g->nodes[input].dtype, out_shape);
    n.inputs[0] = shape_node; // shape constant
    n.inputs[1] = input;
    n.n_inputs = 2;
    return orion_graph_add_node(g, &n);
}

int orion_gb_transpose(OrionGraph* g, int input, int perm_node, const char* name,
                       int perm[4], int out_shape[4]) {
    OrionNode n = make_node(ORION_OP_TRANSPOSE, name, g->nodes[input].dtype, out_shape);
    n.inputs[0] = perm_node;
    n.inputs[1] = input;
    n.n_inputs = 2;
    memcpy(n.attrs.perm, perm, 4 * sizeof(int));
    return orion_graph_add_node(g, &n);
}

int orion_gb_cast(OrionGraph* g, int input, OrionDtype target_dtype,
                  const char* name, int out_shape[4]) {
    OrionNode n = make_node(ORION_OP_CAST, name, target_dtype, out_shape ? out_shape : g->nodes[input].shape);
    n.inputs[0] = input;
    n.n_inputs = 1;
    n.attrs.cast_dtype = target_dtype;
    return orion_graph_add_node(g, &n);
}

// Helper for unary ops — inherit shape from input
static int unary_op(OrionGraph* g, OrionOp op, int input, const char* name) {
    OrionNode n = make_node(op, name, g->nodes[input].dtype, g->nodes[input].shape);
    n.inputs[0] = input;
    n.n_inputs = 1;
    return orion_graph_add_node(g, &n);
}

int orion_gb_relu(OrionGraph* g, int input, const char* name) {
    return unary_op(g, ORION_OP_RELU, input, name);
}
int orion_gb_tanh(OrionGraph* g, int input, const char* name) {
    return unary_op(g, ORION_OP_TANH, input, name);
}
int orion_gb_sigmoid(OrionGraph* g, int input, const char* name) {
    return unary_op(g, ORION_OP_SIGMOID, input, name);
}
int orion_gb_neg(OrionGraph* g, int input, const char* name) {
    return unary_op(g, ORION_OP_NEG, input, name);
}
int orion_gb_sqrt(OrionGraph* g, int input, const char* name) {
    return unary_op(g, ORION_OP_SQRT, input, name);
}
int orion_gb_rsqrt(OrionGraph* g, int input, const char* name) {
    return unary_op(g, ORION_OP_RSQRT, input, name);
}
int orion_gb_identity(OrionGraph* g, int input, const char* name) {
    return unary_op(g, ORION_OP_IDENTITY, input, name);
}

int orion_gb_softmax(OrionGraph* g, int input, int axis, const char* name) {
    OrionNode n = make_node(ORION_OP_SOFTMAX, name, g->nodes[input].dtype, g->nodes[input].shape);
    n.inputs[0] = input;
    n.n_inputs = 1;
    n.attrs.axis = axis;
    return orion_graph_add_node(g, &n);
}

int orion_gb_exp(OrionGraph* g, int input, const char* name) {
    return unary_op(g, ORION_OP_EXP, input, name);
}

int orion_gb_pow(OrionGraph* g, int base, int exponent, const char* name) {
    return binary_op(g, ORION_OP_POW, base, exponent, name);
}

int orion_gb_reduce_sum(OrionGraph* g, int input, int axes, bool keep_dims,
                        const char* name, int out_shape[4]) {
    OrionNode n = make_node(ORION_OP_REDUCE_SUM, name, g->nodes[input].dtype, out_shape);
    n.inputs[0] = input;
    n.inputs[1] = axes;
    n.n_inputs = 2;
    n.attrs.keep_dims = keep_dims;
    return orion_graph_add_node(g, &n);
}

int orion_gb_reduce_mean(OrionGraph* g, int input, int axes, bool keep_dims,
                         const char* name, int out_shape[4]) {
    OrionNode n = make_node(ORION_OP_REDUCE_MEAN, name, g->nodes[input].dtype, out_shape);
    n.inputs[0] = input;
    n.inputs[1] = axes;
    n.n_inputs = 2;
    n.attrs.keep_dims = keep_dims;
    return orion_graph_add_node(g, &n);
}

int orion_gb_slice(OrionGraph* g, int input, int begin_node, int end_node,
                   const char* name, int out_shape[4]) {
    OrionNode n = make_node(ORION_OP_SLICE, name, g->nodes[input].dtype, out_shape);
    n.inputs[0] = begin_node;
    n.inputs[1] = end_node;
    n.inputs[2] = input;
    n.n_inputs = 3;
    return orion_graph_add_node(g, &n);
}

void orion_gb_output(OrionGraph* g, int node, const char* name) {
    if (!g || node < 0 || node >= g->n_nodes) return;
    g->nodes[node].is_output = true;
    if (g->n_outputs < ORION_MAX_GRAPH_IO) {
        snprintf(g->outputs[g->n_outputs].name, ORION_MAX_NAME, "%s", name);
        g->outputs[g->n_outputs].node_idx = node;
        g->n_outputs++;
    }
}

// ---- Composites ----

int orion_gb_layernorm(OrionGraph* g, int input, int gamma_weight, int beta_weight,
                       float eps, const char* prefix, int dim, int seq) {
    char buf[ORION_MAX_NAME];

    (void)dim; // Shape info carried by input node

    // axes = [1] for channel dim
    int ax_shape[4] = {1,0,0,0};
    int ax_val = 1;
    snprintf(buf, sizeof(buf), "%s_ax", prefix);
    int axes = orion_gb_const_int32(g, buf, ax_shape, &ax_val, 1);

    // mean = reduce_mean(x, axis=1, keep_dims=true)
    int mean_shape[4] = {1, 1, 1, seq};
    snprintf(buf, sizeof(buf), "%s_mean", prefix);
    int mean = orion_gb_reduce_mean(g, input, axes, true, buf, mean_shape);

    // centered = x - mean
    snprintf(buf, sizeof(buf), "%s_cent", prefix);
    int cent = orion_gb_sub(g, input, mean, buf);

    // sq = centered * centered
    snprintf(buf, sizeof(buf), "%s_sq", prefix);
    int sq = orion_gb_mul(g, cent, cent, buf);

    // var = reduce_mean(sq, axis=1, keep_dims=true)
    snprintf(buf, sizeof(buf), "%s_var", prefix);
    int var = orion_gb_reduce_mean(g, sq, axes, true, buf, mean_shape);

    // eps constant
    snprintf(buf, sizeof(buf), "%s_eps", prefix);
    int eps_c = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, eps);

    // var + eps
    snprintf(buf, sizeof(buf), "%s_veps", prefix);
    int veps = orion_gb_add(g, var, eps_c, buf);

    // pow(var+eps, -0.5) = rsqrt
    snprintf(buf, sizeof(buf), "%s_nhalf", prefix);
    int nhalf = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, -0.5f);
    snprintf(buf, sizeof(buf), "%s_rstd", prefix);
    int rstd = orion_gb_pow(g, veps, nhalf, buf);

    // norm = centered * rstd
    snprintf(buf, sizeof(buf), "%s_norm", prefix);
    int norm = orion_gb_mul(g, cent, rstd, buf);

    // scaled = norm * gamma
    snprintf(buf, sizeof(buf), "%s_scaled", prefix);
    int scaled = orion_gb_mul(g, norm, gamma_weight, buf);

    // out = scaled + beta
    snprintf(buf, sizeof(buf), "%s_out", prefix);
    int out = orion_gb_add(g, scaled, beta_weight, buf);

    return out;
}

int orion_gb_gelu(OrionGraph* g, int input, const char* prefix, int dim __attribute__((unused)), int seq __attribute__((unused))) {
    char buf[ORION_MAX_NAME];

    // x^2
    snprintf(buf, sizeof(buf), "%s_x2", prefix);
    int x2 = orion_gb_mul(g, input, input, buf);

    // x^3
    snprintf(buf, sizeof(buf), "%s_x3", prefix);
    int x3 = orion_gb_mul(g, x2, input, buf);

    // 0.044715 * x^3
    snprintf(buf, sizeof(buf), "%s_c1", prefix);
    int c1 = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, 0.044715f);
    snprintf(buf, sizeof(buf), "%s_cx3", prefix);
    int cx3 = orion_gb_mul(g, x3, c1, buf);

    // x + 0.044715 * x^3
    snprintf(buf, sizeof(buf), "%s_inner", prefix);
    int inner = orion_gb_add(g, input, cx3, buf);

    // sqrt(2/pi) * inner
    snprintf(buf, sizeof(buf), "%s_c2", prefix);
    int c2 = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, 0.7979f);
    snprintf(buf, sizeof(buf), "%s_scaled", prefix);
    int scaled = orion_gb_mul(g, inner, c2, buf);

    // tanh
    snprintf(buf, sizeof(buf), "%s_th", prefix);
    int th = orion_gb_tanh(g, scaled, buf);

    // 1 + tanh
    snprintf(buf, sizeof(buf), "%s_one", prefix);
    int one = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, 1.0f);
    snprintf(buf, sizeof(buf), "%s_onep", prefix);
    int onep = orion_gb_add(g, th, one, buf);

    // 0.5 * x
    snprintf(buf, sizeof(buf), "%s_half", prefix);
    int half = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, 0.5f);
    snprintf(buf, sizeof(buf), "%s_hx", prefix);
    int hx = orion_gb_mul(g, input, half, buf);

    // out = 0.5 * x * (1 + tanh(...))
    snprintf(buf, sizeof(buf), "%s_out", prefix);
    int out = orion_gb_mul(g, hx, onep, buf);

    return out;
}

int orion_gb_silu(OrionGraph* g, int input, const char* prefix, int dim __attribute__((unused)), int seq __attribute__((unused))) {
    char buf[ORION_MAX_NAME];

    snprintf(buf, sizeof(buf), "%s_sig", prefix);
    int sig = orion_gb_sigmoid(g, input, buf);

    snprintf(buf, sizeof(buf), "%s_out", prefix);
    int out = orion_gb_mul(g, input, sig, buf);

    return out;
}

int orion_gb_rmsnorm(OrionGraph* g, int input, int weight, float eps,
                     const char* prefix, int dim, int seq) {
    char buf[ORION_MAX_NAME];

    // x^2
    snprintf(buf, sizeof(buf), "%s_sq", prefix);
    int sq = orion_gb_mul(g, input, input, buf);

    // axes = [1]
    int ax_shape[4] = {1,0,0,0};
    int ax_val = 1;
    snprintf(buf, sizeof(buf), "%s_ax", prefix);
    int axes = orion_gb_const_int32(g, buf, ax_shape, &ax_val, 1);

    // reduce_sum(x^2, axis=1, keep_dims=true)
    int sum_shape[4] = {1, 1, 1, seq};
    snprintf(buf, sizeof(buf), "%s_ss", prefix);
    int ss = orion_gb_reduce_sum(g, sq, axes, true, buf, sum_shape);

    // / dim
    float inv_dim = 1.0f / (float)dim;
    snprintf(buf, sizeof(buf), "%s_invd", prefix);
    int invd = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, inv_dim);
    snprintf(buf, sizeof(buf), "%s_ms", prefix);
    int ms = orion_gb_mul(g, ss, invd, buf);

    // + eps
    snprintf(buf, sizeof(buf), "%s_eps", prefix);
    int eps_c = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, eps);
    snprintf(buf, sizeof(buf), "%s_mse", prefix);
    int mse = orion_gb_add(g, ms, eps_c, buf);

    // pow(mse, -0.5) = rsqrt
    snprintf(buf, sizeof(buf), "%s_nhalf", prefix);
    int nhalf = orion_gb_const_scalar(g, buf, ORION_DTYPE_FP16, -0.5f);
    snprintf(buf, sizeof(buf), "%s_rrms", prefix);
    int rrms = orion_gb_pow(g, mse, nhalf, buf);

    // x * rrms
    snprintf(buf, sizeof(buf), "%s_xr", prefix);
    int xr = orion_gb_mul(g, input, rrms, buf);

    // weight * normalized
    snprintf(buf, sizeof(buf), "%s_out", prefix);
    int out = orion_gb_mul(g, xr, weight, buf);

    return out;
}

int orion_gb_linear(OrionGraph* g, int input, const char* prefix,
                    int in_dim, int out_dim, int seq,
                    const char* weight_path, const char* bias_path) {
    char buf[ORION_MAX_NAME];

    // Weight: [out_dim, in_dim, 1, 1]
    int w_shape[4] = {out_dim, in_dim, 1, 1};
    snprintf(buf, sizeof(buf), "%s_W", prefix);
    int w = orion_gb_const_weight(g, buf, ORION_DTYPE_FP16, w_shape, weight_path, 64);

    int bias = -1;
    if (bias_path) {
        int b_shape[4] = {1, out_dim, 1, 1};
        snprintf(buf, sizeof(buf), "%s_b", prefix);
        bias = orion_gb_const_weight(g, buf, ORION_DTYPE_FP16, b_shape, bias_path, 64);
    }

    // Conv op name
    snprintf(buf, sizeof(buf), "%s_conv", prefix);
    int conv = orion_gb_conv1x1(g, input, w, -1, buf, out_dim, seq);

    if (bias >= 0) {
        // out = conv + bias
        snprintf(buf, sizeof(buf), "%s_out", prefix);
        return orion_gb_add(g, conv, bias, buf);
    } else {
        // No bias — alias conv as identity for naming consistency
        snprintf(buf, sizeof(buf), "%s_out", prefix);
        return orion_gb_identity(g, conv, buf);
    }
}
