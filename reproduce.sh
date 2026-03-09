#!/bin/bash
#SBATCH --job-name=multi-grpo
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=8:00:00
#SBATCH --mem=0
#SBATCH -p lem-gpu-short
#SBATCH --verbose
#SBATCH --gres=gpu:hopper:4

set -euo pipefail

###############################################################################
# User settings
###############################################################################
export SIF_S3="${SIF_S3:-s3min-tomasznaskret-1712063354/user/jmoska/stronger_mas.sif}"
export S3_OUTPUT="${S3_OUTPUT:-s3min-tomasznaskret-1712063354/user/jmoska/MultiGRPO/output/}"

# Which dataset loaders to run
export PREPARE_CODE_DATA="${PREPARE_CODE_DATA:-0}"
export PREPARE_MATH_DATA="${PREPARE_MATH_DATA:-1}"
export PREPARE_SOKOBAN_DATA="${PREPARE_SOKOBAN_DATA:-0}"

# Original repo training script to run after dataset prep
# Examples:
#   scripts/train/math/math_L1_prompt.sh
#   scripts/train/math/math_L3_model.sh
#   scripts/train/code/code_single_policy.sh
#   scripts/train/code/code_two_policy.sh
#   scripts/train/plan/plan_path_single.sh
#   scripts/train/plan/plan_path_two_policy.sh
#   scripts/train/games/sokoban_two_policy.sh
#   scripts/train/games/sokodu_single.sh
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/math/math_L1_prompt.sh}"

# Optional Hugging Face token / Weights & Biases
export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY=moska-phd-research
export WANDB_PROJECT=pettingllms-quickstart
export WANDB_NAME=first_run

###############################################################################
# Scratch layout
###############################################################################


export RUN_ROOT="${TMPDIR:-/tmp}/pettingllms_${SLURM_JOB_ID}"
export LOCAL_SIF="$RUN_ROOT/$(basename "$SIF_S3")"
export LOCAL_HF_HOME="$RUN_ROOT/huggingface"
export LOCAL_DATASETS="$RUN_ROOT/datasets"
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
# Download SIF
###############################################################################
echo "Downloading SIF from s3v2:$SIF_S3"
rclone copy --progress "s3v2:$SIF_S3" "$RUN_ROOT/"

###############################################################################
# Apptainer env passed into container
###############################################################################
export APPTAINER_TMPDIR="$LOCAL_APPTAINER_CACHE"
export APPTAINER_CACHEDIR="$LOCAL_APPTAINER_CACHE"

export APPTAINERENV_TMPDIR="/tmp/tmpdir"
export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
export APPTAINERENV_WANDB_API_KEY="$WANDB_API_KEY"

export APPTAINERENV_WANDB_ENTITY="$WANDB_ENTITY"
export APPTAINERENV_WANDB_PROJECT="$WANDB_PROJECT"
export APPTAINERENV_WANDB_NAME="$WANDB_NAME"

export APPTAINERENV_HF_HOME="/tmp/tmpdir/huggingface"
export APPTAINERENV_TRANSFORMERS_CACHE="/tmp/tmpdir/huggingface/transformers"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="/tmp/tmpdir/huggingface/hub"

export APPTAINERENV_WANDB_DIR="/tmp/tmpdir/wandb"
export APPTAINERENV_WANDB_CACHE_DIR="/tmp/tmpdir/wandb/.cache"
export APPTAINERENV_WANDB_CONFIG_DIR="/tmp/tmpdir/wandb/.config"

export APPTAINERENV_TRITON_CACHE_DIR="/tmp/tmpdir/triton"
export APPTAINERENV_TORCH_EXTENSIONS_DIR="/tmp/tmpdir/torch_extensions"

###############################################################################
# Command: prepare datasets like upstream quick start, then run original script
###############################################################################
COMMAND=$(cat <<BASH_EOF
set -euo pipefail

cd /workspace/PettingLLMs

mkdir -p /tmp/tmpdir/output
mkdir -p /tmp/tmpdir/huggingface/transformers
mkdir -p /tmp/tmpdir/huggingface/hub
mkdir -p /tmp/tmpdir/wandb/.cache
mkdir -p /tmp/tmpdir/wandb/.config
mkdir -p /tmp/tmpdir/triton
mkdir -p /tmp/tmpdir/torch_extensions
mkdir -p /workspace/PettingLLMs/datasets

echo "=== Dataset preparation ==="
if [[ "${PREPARE_CODE_DATA}" == "1" ]]; then
  echo "Running: python scripts/dataprocess/load_code.py"
  python scripts/dataprocess/load_code.py
fi

if [[ "${PREPARE_MATH_DATA}" == "1" ]]; then
  echo "Running: python scripts/dataprocess/load_math.py"
  python scripts/dataprocess/load_math.py
fi

if [[ "${PREPARE_SOKOBAN_DATA}" == "1" ]]; then
  echo "Running: python scripts/dataprocess/load_sokoban.py"
  python scripts/dataprocess/load_sokoban.py
fi

echo "=== Dataset directories after preparation ==="
ls -lah /workspace/PettingLLMs/datasets || true
ls -lah /workspace/PettingLLMs/datasets/code || true
ls -lah /workspace/PettingLLMs/datasets/math || true
ls -lah /workspace/PettingLLMs/datasets/sudoku_environments || true

echo "=== Training ==="
echo "Running: bash ${TRAIN_SCRIPT}"
bash "${TRAIN_SCRIPT}"
BASH_EOF
)

###############################################################################
# Run inside Apptainer
###############################################################################
srun apptainer exec --nv \
  --mount type=bind,src="$RUN_ROOT",dst=/tmp/tmpdir \
  --mount type=bind,src="$LOCAL_DATASETS",dst=/workspace/PettingLLMs/datasets \
  "$LOCAL_SIF" \
  bash -c "$COMMAND"

###############################################################################
# Collect outputs if present
###############################################################################
echo "Collecting outputs..."
if [[ -d "$RUN_ROOT/output" ]]; then
  echo "Copying outputs to s3v2:$S3_OUTPUT/${SLURM_JOB_ID}/"
  rclone copy --progress "$RUN_ROOT/output" "s3v2:$S3_OUTPUT/${SLURM_JOB_ID}/"
fi

###############################################################################
# Cleanup
###############################################################################
if [[ -n "${RUN_ROOT:-}" && -d "$RUN_ROOT" ]]; then
  rm -rf "$RUN_ROOT"
fi