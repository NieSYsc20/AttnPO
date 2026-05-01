#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
train_dataset=deepscaler
train_files="./data/train/preprocessed_data/${train_dataset}.parquet"
val_files="['./data/test/preprocessed_data/gsm8k*4.parquet','./data/test/preprocessed_data/math*4.parquet','./data/test/preprocessed_data/aime2024*16.parquet']"
val_before_train=True
MODEL_PATH=".model/DeepSeek-R1-Distill-Qwen-1.5B"
GPU_PER_NODE=8
NNODE=1

batch_size=128
n_rollout=8
max_response_length=16384
LR=2e-6
SAVE_FREQ=10
TEST_FREQ=20
ROLLTEMP=1

traineff_coff=0.2
fgreward_heads="[[16,2],[20,9],[23,2]]"
alpha=2
beta=2
boardline=0

PROJECT_NAME="AttnPO" 
EXP_NAME="DS-1.5B"

ROLL_OUT_DIR="./rollout_data/$EXP_NAME"
mkdir -p $ROLL_OUT_DIR

mkdir -p ./log/


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=rloo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    length_penalty.method=$method \
    length_penalty.traineff_coff=$traineff_coff \
    length_penalty.enable_fgreward=$enable_fgreward \
    length_penalty.fgreward_heads=$fgreward_heads \
    length_penalty.alpha=$alpha \
    length_penalty.beta=$beta \
    length_penalty.boardline=$boardline \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=$batch_size \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=$n_rollout \
    actor_rollout_ref.rollout.temperature=$ROLLTEMP \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.val_before_train=$val_before_train \
    trainer.n_gpus_per_node=$GPU_PER_NODE \
    trainer.nnodes=$NNODE \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=10 \
    trainer.rollout_data_dir=$ROLL_OUT_DIR $@ 2>&1 | tee ./log/$EXP_NAME.log
