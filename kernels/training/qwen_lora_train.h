#ifndef ORION_QWEN_LORA_TRAIN_H
#define ORION_QWEN_LORA_TRAIN_H

#import <Foundation/Foundation.h>
#import "../../model/weight_loader.h"
#import "qwen_lora_cpu_ops.h"

typedef struct OrionQwen9BCPUTrainContext OrionQwen9BCPUTrainContext;

typedef struct {
    int step;
    int layer_idx;
    float lr;
    float beta1;
    float beta2;
    float eps;
    OrionLoRAMatrix q_proj;
    OrionLoRAMatrix v_proj;
    OrionQwenStreamingCEContext *ce_ctx;
    int owns_ce_ctx;
    OrionQwen9BCPUTrainContext *cpu_ctx;
    int owns_cpu_ctx;
} OrionQwen9BLoRATrainer;

typedef struct {
    float loss;
    double q_grad_abs_sum;
    double v_grad_abs_sum;
    double q_param_abs_sum;
    double v_param_abs_sum;
    int predicted_token;
} OrionQwen9BLoRASmokeResult;

typedef struct {
    int items_completed;
    double loss_sum;
    float loss_first;
    float loss_last;
    float loss_min;
    float loss_max;
    float loss_avg;
    double q_grad_abs_sum_last;
    double v_grad_abs_sum_last;
    double q_param_abs_sum_last;
    double v_param_abs_sum_last;
    int predicted_token_last;
} OrionQwen9BLoRABatchResult;

void orion_qwen9b_lora_trainer_init(OrionQwen9BLoRATrainer *trainer,
                                    const OrionQwen35Manifest *manifest,
                                    int layer_idx,
                                    int rank,
                                    float alpha,
                                    float lr,
                                    unsigned int seed);

void orion_qwen9b_lora_trainer_free(OrionQwen9BLoRATrainer *trainer);

int orion_qwen9b_lora_trainer_attach_ce_context(OrionQwen9BLoRATrainer *trainer,
                                                const char *embed_blob_path,
                                                const OrionQwen35Manifest *manifest);

int orion_qwen9b_lora_trainer_attach_cpu_train_context(OrionQwen9BLoRATrainer *trainer,
                                                       const char *blob_dir,
                                                       const OrionQwen35Manifest *manifest);

void orion_qwen9b_lora_trainer_zero_grad(OrionQwen9BLoRATrainer *trainer);

void orion_qwen9b_lora_trainer_scale_grad(OrionQwen9BLoRATrainer *trainer,
                                          float scale);

void orion_qwen9b_lora_trainer_step(OrionQwen9BLoRATrainer *trainer);

int orion_qwen9b_lora_trainer_save(const OrionQwen9BLoRATrainer *trainer,
                                   const char *out_dir);

int orion_qwen9b_lora_trainer_load(OrionQwen9BLoRATrainer *trainer,
                                   const char *in_dir);

int orion_qwen9b_lora_trainer_compare(const OrionQwen9BLoRATrainer *lhs,
                                      const OrionQwen9BLoRATrainer *rhs,
                                      float atol);

int orion_qwen9b_lora_train_smoke1(const char *blob_dir,
                                   OrionQwen9BLoRATrainer *trainer,
                                   int input_token,
                                   int target_token,
                                   OrionQwen9BLoRASmokeResult *out_result);

int orion_qwen9b_lora_train_accumulate1(const char *blob_dir,
                                        OrionQwen9BLoRATrainer *trainer,
                                        int input_token,
                                        int target_token,
                                        OrionQwen9BLoRASmokeResult *out_result);

int orion_qwen9b_lora_train_accumulate_hidden1(const char *blob_dir,
                                               const OrionQwen35Manifest *manifest,
                                               OrionQwen9BLoRATrainer *trainer,
                                               const float *hidden_in,
                                               int target_token,
                                               OrionQwen9BLoRASmokeResult *out_result);

int orion_qwen9b_lora_train_hidden_batch(const char *blob_dir,
                                         const OrionQwen35Manifest *manifest,
                                         OrionQwen9BLoRATrainer *trainer,
                                         const float *const *hidden_batch,
                                         const int *target_tokens,
                                         int item_count,
                                         OrionQwen9BLoRABatchResult *out_result);

int orion_qwen9b_lora_train_smoke1_ane_qv_base(const char *blob_dir,
                                               OrionQwen9BLoRATrainer *trainer,
                                               int input_token,
                                               int target_token,
                                               OrionQwen9BLoRASmokeResult *out_result);

int orion_qwen9b_lora_frozen_prefix_hidden(const char *blob_dir,
                                           const OrionQwen35Manifest *manifest,
                                           int input_token,
                                           float *hidden_out);

int orion_qwen9b_lora_ane_train_bridge_last_compile_cache_hit(void);
int orion_qwen9b_lora_ane_train_bridge_last_q_cache_hit(void);
int orion_qwen9b_lora_ane_train_bridge_last_kv_cache_hit(void);
const char *orion_qwen9b_lora_ane_train_bridge_last_compile_cache_source(void);

#endif // ORION_QWEN_LORA_TRAIN_H
