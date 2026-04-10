#!/bin/bash
set -euo pipefail
set -x

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-100000000000}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
GPU_NUM="${GPU_num:-1}"
N_AGENTS="${N_AGENTS:-4}"

MODEL_0="${MODEL_0:-Qwen/Qwen3-1.7B}"
DATASET_NAME="${DATASET_NAME:-polaris}"
BENCHMARK="${BENCHMARK:-AIME24}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-math_turn_graph_mvp}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
TRAIN_SAMPLE_NUM="${TRAIN_SAMPLE_NUM:-4}"
VALIDATE_SAMPLE_NUM="${VALIDATE_SAMPLE_NUM:-3}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"
VAL_FREQ="${VAL_FREQ:-10}"

MODEL_0_CONFIG_PATH="models.model_0.ppo_trainer_config"
MODEL_0_RESOURCE="resource.n_gpus_per_node=$GPU_NUM $MODEL_0_CONFIG_PATH.trainer.n_gpus_per_node=$GPU_NUM $MODEL_0_CONFIG_PATH.trainer.nnodes=1 $MODEL_0_CONFIG_PATH.actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_NUM"

python3 -m pettingllms.trainer.train \
  --config-path "${REPO_ROOT}/pettingllms/config/math" \
  --config-name math_turn_graph_mvp \
  ${MODEL_0_RESOURCE} \
  base_models.policy_0.path="${MODEL_0}" \
  training.experiment_name="${EXPERIMENT_NAME}" \
  training.total_training_steps="${TRAIN_STEPS}" \
  training.train_batch_size="${TRAIN_BATCH_SIZE}" \
  training.train_sample_num="${TRAIN_SAMPLE_NUM}" \
  training.validate_sample_num="${VALIDATE_SAMPLE_NUM}" \
  training.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  training.max_response_length="${MAX_RESPONSE_LENGTH}" \
  training.val_freq="${VAL_FREQ}" \
  env.dataset="${DATASET_NAME}" \
  env.benchmark="${BENCHMARK}" \
  env.n_agents="${N_AGENTS}"
