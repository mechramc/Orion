// compiler/codegen.m — T120: MIL code generation from OrionGraph

#import "codegen.h"
#import "topo.h"
#include <string.h>

// Format a tensor type string: "tensor<fp16, [1,768,1,64]>"
static NSString* fmt_tensor_type(OrionDtype dtype, const int shape[4]) {
    return [NSString stringWithFormat:@"tensor<%s, [%d,%d,%d,%d]>",
            orion_dtype_mil_name(dtype), shape[0], shape[1], shape[2], shape[3]];
}

// Format a node reference by name
static NSString* ref(const OrionGraph* g, int idx) {
    return @(g->nodes[idx].name);
}

// Emit a single const node
static void emit_const(NSMutableString* m, const OrionNode* n) {
    if (n->attrs.blob_path[0]) {
        // Weight reference via BLOBFILE
        [m appendFormat:@"        %@ %s = const()[name=string(\"%s\"), "
         "val=%@(BLOBFILE(path=string(\"%s\"), offset=uint64(%llu)))];\n",
         fmt_tensor_type(n->dtype, n->shape), n->name, n->name,
         fmt_tensor_type(n->dtype, n->shape), n->attrs.blob_path, n->attrs.blob_offset];
    } else if (n->dtype == ORION_DTYPE_INT32) {
        // Int32 tensor constant (small, stored in perm[])
        int ndim = n->shape[0]; // First element of shape is count for 1D constants
        if (ndim == 1) {
            [m appendFormat:@"        tensor<int32, [1]> %s = const()[name=string(\"%s\"), "
             "val=tensor<int32, [1]>([%d])];\n",
             n->name, n->name, n->attrs.perm[0]];
        } else if (ndim == 2) {
            [m appendFormat:@"        tensor<int32, [2]> %s = const()[name=string(\"%s\"), "
             "val=tensor<int32, [2]>([%d,%d])];\n",
             n->name, n->name, n->attrs.perm[0], n->attrs.perm[1]];
        } else {
            [m appendFormat:@"        tensor<int32, [4]> %s = const()[name=string(\"%s\"), "
             "val=tensor<int32, [4]>([%d,%d,%d,%d])];\n",
             n->name, n->name,
             n->attrs.perm[0], n->attrs.perm[1], n->attrs.perm[2], n->attrs.perm[3]];
        }
    } else if (n->dtype == ORION_DTYPE_BOOL) {
        [m appendFormat:@"        bool %s = const()[name=string(\"%s\"), val=bool(%s)];\n",
         n->name, n->name, n->attrs.scalar_val > 0 ? "true" : "false"];
    } else if (n->dtype == ORION_DTYPE_STRING) {
        [m appendFormat:@"        string %s = const()[name=string(\"%s\"), val=string(\"%s\")];\n",
         n->name, n->name, n->attrs.blob_path]; // Reuse blob_path for string value
    } else {
        // Scalar fp16/fp32
        [m appendFormat:@"        %s %s = const()[name=string(\"%s\"), val=%s(%g)];\n",
         orion_dtype_mil_name(n->dtype), n->name, n->name,
         orion_dtype_mil_name(n->dtype), n->attrs.scalar_val];
    }
}

// Emit conv1x1
static void emit_conv(NSMutableString* m, const OrionGraph* g, const OrionNode* n) {
    NSString* prefix = @(n->name);
    // We need to emit the conv constants inline (strides, padding, etc.)
    // These are always the same for 1x1 conv on ANE
    // Emit conv parameters as inline constants
    [m appendFormat:@"        string %@_pt = const()[name=string(\"%@_pt\"), val=string(\"valid\")];\n", prefix, prefix];
    [m appendFormat:@"        tensor<int32, [2]> %@_st = const()[name=string(\"%@_st\"), val=tensor<int32, [2]>([1,1])];\n", prefix, prefix];
    [m appendFormat:@"        tensor<int32, [4]> %@_pd = const()[name=string(\"%@_pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n", prefix, prefix];
    [m appendFormat:@"        tensor<int32, [2]> %@_dl = const()[name=string(\"%@_dl\"), val=tensor<int32, [2]>([1,1])];\n", prefix, prefix];
    [m appendFormat:@"        int32 %@_gr = const()[name=string(\"%@_gr\"), val=int32(1)];\n", prefix, prefix];

    // Conv op — bias handled as separate add node (ANE doesn't support bias= in conv)
    [m appendFormat:@"        %@ %s = conv("
     "dilations=%@_dl, groups=%@_gr, pad=%@_pd, pad_type=%@_pt, strides=%@_st, "
     "weight=%@, x=%@)[name=string(\"%s\")];\n",
     fmt_tensor_type(n->dtype, n->shape), n->name,
     prefix, prefix, prefix, prefix, prefix,
     ref(g, n->inputs[1]), // weight
     ref(g, n->inputs[0]), // input
     n->name];
}

// Emit a single node as MIL text
static void emit_node(NSMutableString* m, const OrionGraph* g, int idx) {
    const OrionNode* n = &g->nodes[idx];
    if (!n->is_live) return;

    NSString* type_str = fmt_tensor_type(n->dtype, n->shape);

    switch (n->op) {
        case ORION_OP_INPUT:
            // Inputs are emitted in the function signature, not body
            break;

        case ORION_OP_CONST:
            emit_const(m, n);
            break;

        case ORION_OP_CONV1X1:
            emit_conv(m, g, n);
            break;

        case ORION_OP_ADD:
            [m appendFormat:@"        %@ %s = add(x=%@, y=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), ref(g, n->inputs[1]), n->name];
            break;

        case ORION_OP_SUB:
            [m appendFormat:@"        %@ %s = sub(x=%@, y=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), ref(g, n->inputs[1]), n->name];
            break;

        case ORION_OP_MUL:
            [m appendFormat:@"        %@ %s = mul(x=%@, y=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), ref(g, n->inputs[1]), n->name];
            break;

        case ORION_OP_MATMUL: {
            // ANE MIL parser requires named const refs for bool params
            [m appendFormat:@"        bool %s_tx = const()[name=string(\"%s_tx\"), val=bool(%@)];\n",
             n->name, n->name, n->attrs.transpose_x ? @"true" : @"false"];
            [m appendFormat:@"        bool %s_ty = const()[name=string(\"%s_ty\"), val=bool(%@)];\n",
             n->name, n->name, n->attrs.transpose_y ? @"true" : @"false"];
            [m appendFormat:@"        %@ %s = matmul(transpose_x=%s_tx, transpose_y=%s_ty, x=%@, y=%@)[name=string(\"%s\")];\n",
             type_str, n->name, n->name, n->name,
             ref(g, n->inputs[0]), ref(g, n->inputs[1]), n->name];
            break;
        }

        case ORION_OP_RESHAPE:
            [m appendFormat:@"        %@ %s = reshape(shape=%@, x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), ref(g, n->inputs[1]), n->name];
            break;

        case ORION_OP_TRANSPOSE:
            [m appendFormat:@"        %@ %s = transpose(perm=%@, x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), ref(g, n->inputs[1]), n->name];
            break;

        case ORION_OP_CAST: {
            // For cast, we need a string const with the target dtype name
            // The builder should have created a string const for this,
            // but for simplicity we emit inline
            NSString* dtype_str = @(orion_dtype_mil_name(n->attrs.cast_dtype));
            [m appendFormat:@"        string %s_dt = const()[name=string(\"%s_dt\"), val=string(\"%@\")];\n",
             n->name, n->name, dtype_str];
            [m appendFormat:@"        %@ %s = cast(dtype=%s_dt, x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, n->name, ref(g, n->inputs[0]), n->name];
            break;
        }

        case ORION_OP_RELU:
            [m appendFormat:@"        %@ %s = relu(x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), n->name];
            break;

        case ORION_OP_TANH:
            [m appendFormat:@"        %@ %s = tanh(x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), n->name];
            break;

        case ORION_OP_SIGMOID:
            [m appendFormat:@"        %@ %s = sigmoid(x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), n->name];
            break;

        case ORION_OP_SOFTMAX: {
            [m appendFormat:@"        int32 %s_ax = const()[name=string(\"%s_ax\"), val=int32(%d)];\n",
             n->name, n->name, n->attrs.axis];
            [m appendFormat:@"        %@ %s = softmax(axis=%s_ax, x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, n->name, ref(g, n->inputs[0]), n->name];
            break;
        }

        case ORION_OP_EXP:
            [m appendFormat:@"        %@ %s = exp(x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), n->name];
            break;

        case ORION_OP_POW:
            [m appendFormat:@"        %@ %s = pow(x=%@, y=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), ref(g, n->inputs[1]), n->name];
            break;

        case ORION_OP_REDUCE_SUM: {
            // Emit keep_dims inline
            [m appendFormat:@"        bool %s_kd = const()[name=string(\"%s_kd\"), val=bool(%s)];\n",
             n->name, n->name, n->attrs.keep_dims ? "true" : "false"];
            [m appendFormat:@"        %@ %s = reduce_sum(x=%@, axes=%@, keep_dims=%s_kd)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), ref(g, n->inputs[1]),
             n->name, n->name];
            break;
        }

        case ORION_OP_REDUCE_MEAN: {
            [m appendFormat:@"        bool %s_kd = const()[name=string(\"%s_kd\"), val=bool(%s)];\n",
             n->name, n->name, n->attrs.keep_dims ? "true" : "false"];
            [m appendFormat:@"        %@ %s = reduce_mean(x=%@, axes=%@, keep_dims=%s_kd)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), ref(g, n->inputs[1]),
             n->name, n->name];
            break;
        }

        case ORION_OP_REDUCE_MAX: {
            [m appendFormat:@"        bool %s_kd = const()[name=string(\"%s_kd\"), val=bool(%s)];\n",
             n->name, n->name, n->attrs.keep_dims ? "true" : "false"];
            [m appendFormat:@"        %@ %s = reduce_max(x=%@, axes=%@, keep_dims=%s_kd)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), ref(g, n->inputs[1]),
             n->name, n->name];
            break;
        }

        case ORION_OP_NEG:
            [m appendFormat:@"        %@ %s = neg(x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), n->name];
            break;

        case ORION_OP_SQRT:
            [m appendFormat:@"        %@ %s = sqrt(x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), n->name];
            break;

        case ORION_OP_RSQRT:
            [m appendFormat:@"        %@ %s = rsqrt(x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), n->name];
            break;

        case ORION_OP_IDENTITY:
            [m appendFormat:@"        %@ %s = identity(x=%@)[name=string(\"%s\")];\n",
             type_str, n->name, ref(g, n->inputs[0]), n->name];
            break;

        case ORION_OP_SLICE:
            [m appendFormat:@"        %@ %s = slice_by_index(begin=%@, end=%@, x=%@)[name=string(\"%s\")];\n",
             type_str, n->name,
             ref(g, n->inputs[0]), ref(g, n->inputs[1]), ref(g, n->inputs[2]),
             n->name];
            break;

        default:
            [m appendFormat:@"        // UNSUPPORTED OP: %s\n", orion_op_name(n->op)];
            break;
    }
}

NSString* orion_codegen_mil(const OrionGraph* graph, const char* func_name) {
    if (!graph || graph->n_nodes == 0) return nil;

    // Get topological order
    int count = 0;
    int* order = orion_topo_sort(graph, &count);
    if (!order) return nil;

    NSMutableString* m = [NSMutableString string];

    // Header
    [m appendString:@"program(1.3)\n"];
    [m appendString:@"[buildInfo = dict<string, string>({"
     "{\"coremlc-component-MIL\", \"3510.2.1\"}, "
     "{\"coremlc-version\", \"3505.4.1\"}, "
     "{\"coremltools-component-milinternal\", \"\"}, "
     "{\"coremltools-version\", \"9.0\"}"
     "})]\n"];
    [m appendString:@"{\n"];

    // Function signature
    [m appendFormat:@"    func %s<ios18>(", func_name ? func_name : "main"];
    for (int i = 0; i < graph->n_inputs; i++) {
        if (i > 0) [m appendString:@", "];
        int node_idx = graph->inputs[i].node_idx;
        const OrionNode* inp = &graph->nodes[node_idx];
        [m appendFormat:@"%@ %s",
         fmt_tensor_type(inp->dtype, inp->shape),
         graph->inputs[i].name];
    }
    [m appendString:@") {\n"];

    // Body: emit each node in topological order
    for (int i = 0; i < count; i++) {
        emit_node(m, graph, order[i]);
    }

    // Return tuple
    [m appendString:@"    } -> ("];
    for (int i = 0; i < graph->n_outputs; i++) {
        if (i > 0) [m appendString:@", "];
        [m appendString:@(graph->nodes[graph->outputs[i].node_idx].name)];
    }
    [m appendString:@");\n"];
    [m appendString:@"}\n"];

    free(order);
    return m;
}
