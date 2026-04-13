set -x

export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export HYDRA_FULL_ERROR=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LD_LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH}

TOTAL_GPUS="${GPU_num:-${N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
MODEL_COUNT=2
if [ "$TOTAL_GPUS" -lt "$MODEL_COUNT" ]; then
  echo "Need at least $MODEL_COUNT GPUs for shared-orchestrator hybrid training."
  exit 1
fi

EQUAL_GPUS_PER_MODEL=$((TOTAL_GPUS / MODEL_COUNT))
UNUSED_GPUS=$((TOTAL_GPUS - EQUAL_GPUS_PER_MODEL * MODEL_COUNT))

ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-${MODEL_0:-Qwen/Qwen3-8B}}"
WORKER_MODEL="${WORKER_MODEL:-${MODEL_1:-Qwen/Qwen3-8B}}"
WORKER_TRAINABLE="${WORKER_TRAINABLE:-false}"
WORKER_OPTIMIZATION_MODE="${WORKER_OPTIMIZATION_MODE:-prompt}"
ORCHESTRATOR_TRAINABLE="${ORCHESTRATOR_TRAINABLE:-true}"
ORCHESTRATOR_OPTIMIZATION_MODE="${ORCHESTRATOR_OPTIMIZATION_MODE:-prompt}"

echo "Hybrid shared orchestrator split: TOTAL_GPUS=$TOTAL_GPUS PER_MODEL=$EQUAL_GPUS_PER_MODEL UNUSED_GPUS=$UNUSED_GPUS"

resource_overrides="resource.n_gpus_per_node=$TOTAL_GPUS resource.nnodes=$NNODES"

model_0_config_path="models.model_0.ppo_trainer_config"
model_0_resource="$model_0_config_path.trainer.n_gpus_per_node=$EQUAL_GPUS_PER_MODEL \
$model_0_config_path.trainer.n_training_gpus_per_node=$EQUAL_GPUS_PER_MODEL \
$model_0_config_path.trainer.nnodes=$NNODES \
$model_0_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$EQUAL_GPUS_PER_MODEL"

model_1_config_path="models.model_1.ppo_trainer_config"
model_1_resource="$model_1_config_path.trainer.n_gpus_per_node=$EQUAL_GPUS_PER_MODEL \
$model_1_config_path.trainer.n_training_gpus_per_node=$EQUAL_GPUS_PER_MODEL \
$model_1_config_path.trainer.nnodes=$NNODES \
$model_1_config_path.actor_rollout_ref.rollout.tensor_model_parallel_size=$EQUAL_GPUS_PER_MODEL"

python3 -m pettingllms.trainer.train --config-path ../config/mas_graph --config-name math_decomposer_selector_hybrid_shared \
    $resource_overrides \
    $model_0_resource \
    $model_1_resource \
    base_models.policy_0.path="$ORCHESTRATOR_MODEL" \
    base_models.policy_1.path="$WORKER_MODEL" \
    orchestrator_defaults.trainable=$ORCHESTRATOR_TRAINABLE \
    orchestrator_defaults.optimization_mode=$ORCHESTRATOR_OPTIMIZATION_MODE \
    worker_defaults.trainable=$WORKER_TRAINABLE \
    worker_defaults.optimization_mode=$WORKER_OPTIMIZATION_MODE \
    training.project_name=multi-grpo \
    training.experiment_name=math_decomposer_selector_hybrid_shared \
    training.total_training_steps=200 \
    training.train_batch_size=16 \
    training.train_sample_num=4 \
    training.validate_sample_num=1 \
    training.max_prompt_length=8192 \
    training.max_response_length=4096 \
    training.val_freq=10 \
    env.dataset=polaris \
    env.benchmark=AIME24
