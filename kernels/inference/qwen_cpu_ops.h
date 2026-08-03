#ifndef ORION_QWEN_CPU_OPS_H
#define ORION_QWEN_CPU_OPS_H

/// Qwen RMSNorm without bias.
/// out = x / sqrt(mean(x^2) + eps) * weight
void orion_qwen_cpu_rmsnorm(const float* x, const float* weight, int dim, float eps, float* out);

/// Qwen SwiGLU MLP block:
/// hidden = silu(x @ gate_proj^T) * (x @ up_proj^T)
/// out = hidden @ down_proj^T
///
/// Weights are expected in Orion blob layout:
/// - gate_proj: [d_ff, d_model]
/// - up_proj:   [d_ff, d_model]
/// - down_proj: [d_model, d_ff]
void orion_qwen_cpu_swiglu_ffn(const float* x,
                               const float* gate_proj,
                               const float* up_proj,
                               const float* down_proj,
                               int d_model,
                               int d_ff,
                               float* out);

/// Qwen3.5 full-attention CPU reference path without RoPE.
/// This follows the actual Qwen3.5 projection semantics:
/// - q_proj: [2 * (n_head * head_dim), d_model]
/// - query, gate = split(q_proj(x), 2)
/// - k_proj / v_proj: [n_kv_head * head_dim, d_model]
/// - o_proj: [d_model, n_head * head_dim]
/// - query/key use per-head RMSNorm; post-attention applies sigmoid(gate)
/// This is still a staging implementation and intentionally skips rotary.
///
/// Inputs:
/// - x_seq: [seq_len, d_model]
/// - q_proj: [2 * (n_head * head_dim), d_model]
/// - k_proj: [n_kv_head * head_dim, d_model]
/// - v_proj: [n_kv_head * head_dim, d_model]
/// - o_proj: [d_model, n_head * head_dim]
/// - q_norm / k_norm: [head_dim]
///
/// Output:
/// - out_seq: [seq_len, d_model]
void orion_qwen_cpu_full_attention_prefill_no_rope(const float* x_seq,
                                                   int seq_len,
                                                   const float* q_proj,
                                                   const float* k_proj,
                                                   const float* v_proj,
                                                   const float* o_proj,
                                                   const float* q_norm,
                                                   const float* k_norm,
                                                   int d_model,
                                                   int n_head,
                                                   int n_kv_head,
                                                   int head_dim,
                                                   float* out_seq);

/// Qwen3.5 full-attention CPU reference path with text-only RoPE.
/// For text-only usage, the three MRoPE axes share the same scalar token position,
/// so standard RoPE over the partial rotary sub-dimension is sufficient.
void orion_qwen_cpu_full_attention_prefill_with_rope(const float* x_seq,
                                                     int seq_len,
                                                     const float* q_proj,
                                                     const float* k_proj,
                                                     const float* v_proj,
                                                     const float* o_proj,
                                                     const float* q_norm,
                                                     const float* k_norm,
                                                     int d_model,
                                                     int n_head,
                                                     int n_kv_head,
                                                     int head_dim,
                                                     float rope_theta,
                                                     float partial_rotary_factor,
                                                     float* out_seq);

/// Qwen3.5 full-attention CPU reference path from precomputed projection outputs.
/// Inputs are the direct outputs of the ANE prefill q/kv graphs, transposed back
/// to CPU row-major layout:
/// - q_proj_out_seq: [seq_len, 2 * (n_head * head_dim)]
/// - k_proj_out_seq: [seq_len, n_kv_head * head_dim]
/// - v_proj_out_seq: [seq_len, n_kv_head * head_dim]
///
/// This keeps the post-projection semantics identical to the regular CPU path:
/// per-head RMSNorm, partial RoPE, grouped-query attention, sigmoid gate, o_proj.
void orion_qwen_cpu_full_attention_from_projections_with_rope(
    const float* q_proj_out_seq,
    const float* k_proj_out_seq,
    const float* v_proj_out_seq,
    int seq_len,
    const float* o_proj,
    const float* q_norm,
    const float* k_norm,
    int d_model,
    int n_head,
    int n_kv_head,
    int head_dim,
    float rope_theta,
    float partial_rotary_factor,
    float* out_seq);

/// Qwen3.5 linear-attention CPU preparation path.
/// This implements the front half of Qwen3.5GatedDeltaNet:
/// - in_proj_qkv / in_proj_z / in_proj_a / in_proj_b
/// - depthwise causal conv1d + silu
/// - split into q/k/v
/// - beta = sigmoid(b)
/// - g = -exp(A_log) * softplus(a + dt_bias)
///
/// It intentionally stops before the gated delta recurrent core.
void orion_qwen_cpu_linear_attention_prep(const float* x_seq,
                                          int seq_len,
                                          const float* in_proj_qkv,
                                          const float* in_proj_z,
                                          const float* in_proj_a,
                                          const float* in_proj_b,
                                          const float* conv1d,
                                          const float* dt_bias,
                                          const float* a_log,
                                          int d_model,
                                          int num_k_heads,
                                          int num_v_heads,
                                          int head_k_dim,
                                          int head_v_dim,
                                          int conv_kernel,
                                          float* query_out,
                                          float* key_out,
                                          float* value_out,
                                          float* z_out,
                                          float* beta_out,
                                          float* g_out);

/// Qwen3.5 linear-attention CPU reference path.
/// This extends the prep path with:
/// - l2-normalized recurrent gated delta core
/// - per-head RMSNormGated using z as gate input
/// - out_proj back to d_model
///
/// It intentionally omits cache handling and only targets short prefill smoke.
void orion_qwen_cpu_linear_attention_recurrent_prefill(const float* x_seq,
                                                       int seq_len,
                                                       const float* in_proj_qkv,
                                                       const float* in_proj_z,
                                                       const float* in_proj_a,
                                                       const float* in_proj_b,
                                                       const float* conv1d,
                                                       const float* dt_bias,
                                                       const float* a_log,
                                                       const float* norm_weight,
                                                       const float* out_proj,
                                                       int d_model,
                                                       int num_k_heads,
                                                       int num_v_heads,
                                                       int head_k_dim,
                                                       int head_v_dim,
                                                       int conv_kernel,
                                                       float* out_seq);

#endif // ORION_QWEN_CPU_OPS_H
