#!/bin/bash
#SBATCH --job-name=multi-grpo
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=8:00:00
#SBATCH --mem=0
#SBATCH -p lem-gpu-short
#SBATCH --verbose
#SBATCH --gres=gpu:hopper:2

set -euo pipefail

###############################################################################
# User settings
###############################################################################
export SIF_S3="${SIF_S3:-s3min-tomasznaskret-1712063354/user/jmoska/stronger_mas.sif}"
export DATASET_S3="${DATASET_S3:-s3min-tomasznaskret-1712063354/user/your_user/PettingLLMs/datasets/}"
export S3_OUTPUT="${S3_OUTPUT:-s3min-tomasznaskret-1712063354/user/jmoska/MultiGRPO/output/}"

export MODEL_0="${MODEL_0:-Qwen/Qwen2.5-7B-Instruct}"
export MODEL_1="${MODEL_1:-Qwen/Qwen2.5-7B-Instruct}"

export DATASET_NAME="${DATASET_NAME:-polaris}"
export BENCHMARK="${BENCHMARK:-AIME24}"

export NNODES="${NNODES:-1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"

export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export TRAIN_SAMPLE_NUM="${TRAIN_SAMPLE_NUM:-8}"
export VALIDATE_SAMPLE_NUM="${VALIDATE_SAMPLE_NUM:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
export VAL_FREQ="${VAL_FREQ:-10}"

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-moska-phd-research}"
export WANDB_PROJECT="${WANDB_PROJECT:-pettingllms-l3}"
export WANDB_NAME="${WANDB_NAME:-pettingllms-l3-${SLURM_JOB_ID}}"

###############################################################################
# Resolve repo location on host
###############################################################################
export HOST_REPO_DIR="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

if [[ ! -d "$HOST_REPO_DIR" ]]; then
  echo "ERROR: Could not resolve repo root."
  exit 1
fi

if [[ ! -f "$HOST_REPO_DIR/requirements_venv.txt" ]]; then
  echo "ERROR: $HOST_REPO_DIR does not look like the expected repo root."
  exit 1
fi

###############################################################################
# Scratch layout
###############################################################################
export RUN_ROOT="${TMPDIR:-/tmp}/pettingllms_${SLURM_JOB_ID}"
export LOCAL_SIF="$RUN_ROOT/$(basename "$SIF_S3")"
export LOCAL_DATASETS="$RUN_ROOT/datasets"
export LOCAL_HF_HOME="$RUN_ROOT/huggingface"
export LOCAL_WANDB_DIR="$RUN_ROOT/wandb"
export LOCAL_TRITON_CACHE="$RUN_ROOT/triton"
export LOCAL_TORCH_EXTENSIONS="$RUN_ROOT/torch_extensions"
export LOCAL_APPTAINER_CACHE="$RUN_ROOT/apptainer"
export LOCAL_OUTPUT="$RUN_ROOT/output"

mkdir -p \
  "$RUN_ROOT" \
  "$LOCAL_DATASETS" \
  "$LOCAL_HF_HOME" \
  "$LOCAL_WANDB_DIR" \
  "$LOCAL_TRITON_CACHE" \
  "$LOCAL_TORCH_EXTENSIONS" \
  "$LOCAL_APPTAINER_CACHE" \
  "$LOCAL_OUTPUT"

###############################################################################
# Download inputs
###############################################################################
echo "Downloading SIF from s3v2:$SIF_S3"
rclone copy --progress "s3v2:$SIF_S3" "$RUN_ROOT/"

echo "Downloading datasets from s3v2:$DATASET_S3"
rclone copy --progress "s3v2:$DATASET_S3" "$LOCAL_DATASETS/"

echo "Dataset contents:"
ls -lah "$LOCAL_DATASETS" || true
ls -lah "$LOCAL_DATASETS/math" || true
ls -lah "$LOCAL_DATASETS/code" || true
ls -lah "$LOCAL_DATASETS/sudoku_environments" || true

###############################################################################
# Apptainer host-side cache
###############################################################################
export APPTAINER_TMPDIR="$LOCAL_APPTAINER_CACHE"
export APPTAINER_CACHEDIR="$LOCAL_APPTAINER_CACHE"

###############################################################################
# Environment passed into container
###############################################################################
export APPTAINERENV_TMPDIR="/tmp/tmpdir"

export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
export APPTAINERENV_HF_HOME="/tmp/tmpdir/huggingface"
export APPTAINERENV_TRANSFORMERS_CACHE="/tmp/tmpdir/huggingface/transformers"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="/tmp/tmpdir/huggingface/hub"

export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
export APPTAINERENV_WANDB_ENTITY="${WANDB_ENTITY}"
export APPTAINERENV_WANDB_PROJECT="${WANDB_PROJECT}"
export APPTAINERENV_WANDB_NAME="${WANDB_NAME}"
export APPTAINERENV_WANDB_DIR="/tmp/tmpdir/wandb"
export APPTAINERENV_WANDB_CACHE_DIR="/tmp/tmpdir/wandb/.cache"
export APPTAINERENV_WANDB_CONFIG_DIR="/tmp/tmpdir/wandb/.config"

export APPTAINERENV_TRITON_CACHE_DIR="/tmp/tmpdir/triton"
export APPTAINERENV_TORCH_EXTENSIONS_DIR="/tmp/tmpdir/torch_extensions"

###############################################################################
# Command run inside container
###############################################################################
COMMAND=$(cat <<BASH_EOF
set -euo pipefail

cd /workspace/PettingLLMs

mkdir -p /tmp/tmpdir/output/checkpoints
mkdir -p /tmp/tmpdir/ray
mkdir -p /tmp/tmpdir/huggingface/transformers
mkdir -p /tmp/tmpdir/huggingface/hub
mkdir -p /tmp/tmpdir/wandb/.cache
mkdir -p /tmp/tmpdir/wandb/.config
mkdir -p /tmp/tmpdir/triton
mkdir -p /tmp/tmpdir/torch_extensions

echo "Repo mounted from host:"
pwd
ls -lah .

echo "Datasets mounted from scratch:"
ls -lah /workspace/PettingLLMs/datasets || true
ls -lah /workspace/PettingLLMs/datasets/math || true

echo "Using model_0=${MODEL_0}"
echo "Using model_1=${MODEL_1}"
echo "Using dataset=${DATASET_NAME} benchmark=${BENCHMARK}"

python3 -m pettingllms.trainer.train \
  --config-path /workspace/PettingLLMs/pettingllms/config/math \
  --config-name math_L3_model \
  resource.nnodes="${NNODES}" \
  resource.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  base_models.policy_0.path="${MODEL_0}" \
  base_models.policy_1.path="${MODEL_1}" \
  models.model_1.ppo_trainer_config.actor_rollout_ref.model.path="${MODEL_1}" \
  training.experiment_name="pettingllms-l3-${SLURM_JOB_ID}" \
  training.model_checkpoints_dir=/tmp/tmpdir/output/checkpoints \
  training.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  training.train_batch_size="${TRAIN_BATCH_SIZE}" \
  training.train_sample_num="${TRAIN_SAMPLE_NUM}" \
  training.validate_sample_num="${VALIDATE_SAMPLE_NUM}" \
  training.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  training.max_response_length="${MAX_RESPONSE_LENGTH}" \
  training.val_freq="${VAL_FREQ}" \
  env.dataset="${DATASET_NAME}" \
  env.benchmark="${BENCHMARK}" \
  hydra.run.dir=/tmp/tmpdir/output/run
BASH_EOF
)

###############################################################################
# Run inside Apptainer
###############################################################################
srun apptainer exec --nv \
  --mount type=bind,src="$RUN_ROOT",dst=/tmp/tmpdir \
  --mount type=bind,src="$HOST_REPO_DIR",dst=/workspace/PettingLLMs \
  --mount type=bind,src="$LOCAL_DATASETS",dst=/workspace/PettingLLMs/datasets \
  "$LOCAL_SIF" \
  bash -c "$COMMAND"

###############################################################################
# Upload results
###############################################################################
echo "Copying outputs to s3v2:$S3_OUTPUT/${SLURM_JOB_ID}/"
rclone copy --progress "$RUN_ROOT/output" "s3v2:$S3_OUTPUT/${SLURM_JOB_ID}/"

###############################################################################
# Cleanup
###############################################################################
if [[ -n "${RUN_ROOT:-}" && -d "$RUN_ROOT" ]]; then
  rm -rf "$RUN_ROOT"
fi