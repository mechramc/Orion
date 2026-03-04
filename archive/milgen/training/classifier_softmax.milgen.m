#import "classifier_softmax.milgen.h"

// T070: Classifier forward on ANE — embed @ x → logits
// Input:  fp16 [1, dim, 1, seq]
// Output: fp32 [1, vocab, 1, seq]
// Note: Classifier backward rejected by ANE (32000-input-channel conv) → stays on CPU

NSString* orion_milgen_classifier_fwd(int dim, int vocab) {
    // The classifier is just a linear layer: logits = W_embed^T @ x
    // where W_embed is [vocab, dim] (embedding table used as classifier)
    // We use conv: [vocab, dim, 1, 1] @ [1, dim, 1, seq] → [1, vocab, 1, seq]

    // Note: seq comes from the input shape, but for MIL we need it as a literal.
    // Use 256 (Stories110M training seq_len).
    int seq = 256;

    NSMutableString *body = [NSMutableString string];

    // Linear projection using embedding weights
    [body appendString:orion_mil_linear("cls", "x", dim, vocab, seq,
                                         "@model_path/embed.bin", NULL)];

    // Cast to fp32
    [body appendFormat:
        @"        string dt32 = const()[name=string(\"dt32\"), val=string(\"fp32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> output = cast(dtype=dt32, x=cls_out)[name=string(\"output\")];\n",
        vocab, seq];

    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp16, [1,%d,1,%d]> x", dim, seq];

    return orion_mil_program(body, @[input_decl], @"output");
}

// T071: Vocab softmax on ANE — softmax over 32000 classes
// Input:  fp16 [1, vocab, 1, seq]
// Output: fp32 [1, vocab, 1, seq]

NSString* orion_milgen_vocab_softmax(int vocab, int seq_len) {
    NSMutableString *body = [NSMutableString string];

    // Softmax along axis=1 (channel/vocab dimension)
    [body appendFormat:
        @"        int32 sm_ax = const()[name=string(\"sm_ax\"), val=int32(1)];\n"
         "        tensor<fp16, [1,%d,1,%d]> sm = softmax(axis=sm_ax, x=x)[name=string(\"sm\")];\n",
        vocab, seq_len];

    // Cast to fp32
    [body appendFormat:
        @"        string dt32 = const()[name=string(\"dt32\"), val=string(\"fp32\")];\n"
         "        tensor<fp32, [1,%d,1,%d]> output = cast(dtype=dt32, x=sm)[name=string(\"output\")];\n",
        vocab, seq_len];

    NSString *input_decl = [NSString stringWithFormat:
        @"tensor<fp16, [1,%d,1,%d]> x", vocab, seq_len];

    return orion_mil_program(body, @[input_decl], @"output");
}
