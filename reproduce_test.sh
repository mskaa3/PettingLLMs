#!/bin/bash
#SBATCH --job-name=multi-grpo
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH -p lem-gpu-short
#SBATCH --verbose
#SBATCH --gres=gpu:hopper:2

set -euo pipefail

###############################################################################
# User-configurable settings
###############################################################################
export SIF_S3="${SIF_S3:-s3min-tomasznaskret-1712063354/user/jmoska/stronger_mas.sif}"
export S3_OUTPUT="${S3_OUTPUT:-s3min-tomasznaskret-1712063354/user/jmoska/MultiGRPO/output/}"

export PREPARE_CODE_DATA="${PREPARE_CODE_DATA:-0}"
export PREPARE_MATH_DATA="${PREPARE_MATH_DATA:-1}"
export PREPARE_SOKOBAN_DATA="${PREPARE_SOKOBAN_DATA:-0}"

export TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/math/math_L1_prompt.sh}"
export MODEL_0="${MODEL_0:-Qwen/Qwen3-1.7B}"
export GPU_num="${GPU_num:-1}"

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-moska-phd-research}"
export WANDB_PROJECT="${WANDB_PROJECT:-multi-grpo}"
export WANDB_NAME="${WANDB_NAME:-first_run}"
export WANDB_MODE="${WANDB_MODE:-online}"

# 1 = use repo cloned on HPC and mount it into container
# 0 = use repo baked into image
export USE_HOST_REPO="${USE_HOST_REPO:-1}"

# Paths inside the container image
export CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-/workspace/PettingLLMs}"
export IMAGE_PYTHON="${IMAGE_PYTHON:-/opt/venv/bin/python}"

###############################################################################
# Repo location on host
###############################################################################
REPO_MOUNT_ARGS=()

if [[ "${USE_HOST_REPO}" == "1" ]]; then
  START_DIR="${HOST_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
  HOST_REPO_DIR="$START_DIR"

  while [[ "$HOST_REPO_DIR" != "/" ]]; do
    if [[ -f "$HOST_REPO_DIR/requirements_venv.txt" && \
          -d "$HOST_REPO_DIR/pettingllms" && \
          -d "$HOST_REPO_DIR/scripts" ]]; then
      break
    fi
    HOST_REPO_DIR="$(dirname "$HOST_REPO_DIR")"
  done

  if [[ "$HOST_REPO_DIR" == "/" ]]; then
    echo "ERROR: Could not locate repo root from start dir: $START_DIR"
    echo "Tip: run 'sbatch' from the repo root, or set HOST_REPO_DIR explicitly."
    exit 1
  fi

  if [[ ! -f "$HOST_REPO_DIR/$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Training script not found: $HOST_REPO_DIR/$TRAIN_SCRIPT"
    exit 1
  fi

  export HOST_REPO_DIR
  echo "Using host repo: $HOST_REPO_DIR"
  REPO_MOUNT_ARGS=(--mount "type=bind,src=$HOST_REPO_DIR,dst=$CONTAINER_REPO_DIR")
else
  echo "Using repo baked into image at: $CONTAINER_REPO_DIR"
fi

###############################################################################
# Scratch layout
###############################################################################
export RUN_ROOT="${TMPDIR:-/tmp}/pettingllms_${SLURM_JOB_ID}"
export LOCAL_SIF="$RUN_ROOT/$(basename "$SIF_S3")"
export LOCAL_HF_HOME="$RUN_ROOT/huggingface"
export LOCAL_WANDB_DIR="$RUN_ROOT/wandb"
export LOCAL_TRITON_CACHE="$RUN_ROOT/triton"
export LOCAL_TORCH_EXTENSIONS="$RUN_ROOT/torch_extensions"
export LOCAL_APPTAINER_CACHE="$RUN_ROOT/apptainer"
export LOCAL_OUTPUT="$RUN_ROOT/output"
export LOCAL_HOME="$RUN_ROOT/home"

mkdir -p \
  "$RUN_ROOT" \
  "$LOCAL_HF_HOME" \
  "$LOCAL_WANDB_DIR" \
  "$LOCAL_TRITON_CACHE" \
  "$LOCAL_TORCH_EXTENSIONS" \
  "$LOCAL_APPTAINER_CACHE" \
  "$LOCAL_OUTPUT" \
  "$LOCAL_HOME"

###############################################################################
# Download SIF from S3
###############################################################################
echo "Downloading SIF from s3v2:$SIF_S3"
rclone copy --progress "s3v2:$SIF_S3" "$RUN_ROOT/"

if [[ ! -f "$LOCAL_SIF" ]]; then
  echo "ERROR: SIF was not downloaded to $LOCAL_SIF"
  exit 1
fi

###############################################################################
# Apptainer host/cache settings
###############################################################################
export APPTAINER_TMPDIR="$LOCAL_APPTAINER_CACHE"
export APPTAINER_CACHEDIR="$LOCAL_APPTAINER_CACHE"

###############################################################################
# Environment passed into container
###############################################################################
export APPTAINERENV_HOME="/tmp/tmpdir/home"
export APPTAINERENV_TMPDIR="/tmp/tmpdir"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="${CONTAINER_REPO_DIR}:${CONTAINER_REPO_DIR}/verl"
export APPTAINERENV_IMAGE_PYTHON="${IMAGE_PYTHON}"
export APPTAINERENV_CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR}"

export APPTAINERENV_GPU_num="${GPU_num}"
export APPTAINERENV_MODEL_0="${MODEL_0}"
export APPTAINERENV_TRAIN_SCRIPT="${TRAIN_SCRIPT}"


export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
export APPTAINERENV_HF_HOME="/tmp/tmpdir/huggingface"
export APPTAINERENV_TRANSFORMERS_CACHE="/tmp/tmpdir/huggingface/transformers"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="/tmp/tmpdir/huggingface/hub"
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0

export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
export APPTAINERENV_WANDB_ENTITY="${WANDB_ENTITY}"
export APPTAINERENV_WANDB_PROJECT="${WANDB_PROJECT}"
export APPTAINERENV_WANDB_NAME="${WANDB_NAME}"
export APPTAINERENV_WANDB_MODE="${WANDB_MODE}"
export APPTAINERENV_WANDB_DIR="/tmp/tmpdir/wandb"
export APPTAINERENV_WANDB_CACHE_DIR="/tmp/tmpdir/wandb/.cache"
export APPTAINERENV_WANDB_CONFIG_DIR="/tmp/tmpdir/wandb/.config"

export APPTAINERENV_TRITON_CACHE_DIR="/tmp/tmpdir/triton"
export APPTAINERENV_TORCH_EXTENSIONS_DIR="/tmp/tmpdir/torch_extensions"

###############################################################################
# Command run inside container
###############################################################################
COMMAND=$(cat <<'BASH_EOF'
set -euo pipefail

cd "${CONTAINER_REPO_DIR}"

mkdir -p /tmp/tmpdir/output
mkdir -p /tmp/tmpdir/home
mkdir -p /tmp/tmpdir/huggingface/transformers
mkdir -p /tmp/tmpdir/huggingface/hub
mkdir -p /tmp/tmpdir/wandb/.cache
mkdir -p /tmp/tmpdir/wandb/.config
mkdir -p /tmp/tmpdir/triton
mkdir -p /tmp/tmpdir/torch_extensions
mkdir -p datasets

echo "=== Runtime paths ==="
echo "PWD: $(pwd)"
echo "TRAIN_SCRIPT: ${TRAIN_SCRIPT}"
echo "MODEL_0: ${MODEL_0}"
echo "GPU_num: ${GPU_num}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_ENTITY: ${WANDB_ENTITY}"
echo "WANDB_NAME: ${WANDB_NAME}"
echo "WANDB_MODE: ${WANDB_MODE}"
ls -lah .

echo "=== Python preflight ==="
echo "IMAGE_PYTHON=${IMAGE_PYTHON}"
"${IMAGE_PYTHON}" -V
"${IMAGE_PYTHON}" - <<'PY'
import sys
import numpy as np
import numpy.core.multiarray as ma
import torch
import vllm

print("sys.executable:", sys.executable)
print("numpy:", np.__version__, np.__file__)
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("numpy.core.multiarray.generic:", hasattr(ma, "generic"))
print("numpy.core.multiarray.complexfloating:", hasattr(ma, "complexfloating"))

assert sys.executable.startswith("/opt/venv/"), sys.executable
assert np.__file__.startswith("/opt/venv/"), np.__file__
assert hasattr(ma, "generic")
assert hasattr(ma, "complexfloating")
PY

echo "=== Dataset preparation ==="
if [[ "${PREPARE_CODE_DATA}" == "1" ]]; then
  "${IMAGE_PYTHON}" scripts/dataprocess/load_code.py
fi

if [[ "${PREPARE_MATH_DATA}" == "1" ]]; then
  "${IMAGE_PYTHON}" scripts/dataprocess/load_math.py
fi

if [[ "${PREPARE_SOKOBAN_DATA}" == "1" ]]; then
  "${IMAGE_PYTHON}" scripts/dataprocess/load_sokoban.py
fi

echo "=== Dataset directories after preparation ==="
ls -lah datasets || true
ls -lah datasets/code || true
ls -lah datasets/math || true
ls -lah datasets/sudoku_environments || true

echo "=== Prepare runtime training script copy ==="
RUNTIME_TRAIN_SCRIPT="/tmp/tmpdir/$(basename "${TRAIN_SCRIPT}")"
cp "${TRAIN_SCRIPT}" "${RUNTIME_TRAIN_SCRIPT}"
chmod +x "${RUNTIME_TRAIN_SCRIPT}"

# Patch only the runtime copy, never the host repo file
"${IMAGE_PYTHON}" - <<'PY'
import os
import re
from pathlib import Path

p = Path(f"/tmp/tmpdir/{Path(os.environ['TRAIN_SCRIPT']).name}")
text = p.read_text()

replacements = [
    (r'base_models\.policy_0\.path="[^"]*"', f'base_models.policy_0.path="{os.environ["MODEL_0"]}"'),
    (r'training\.project_name=[^\\\s]+', f'training.project_name={os.environ["WANDB_PROJECT"]}'),
    (r'training\.entity=[^\\\s]+', f'training.entity={os.environ["WANDB_ENTITY"]}'),
    (r'training\.experiment_name=[^\\\s]+', f'training.experiment_name={os.environ["WANDB_NAME"]}'),
]

for pattern, replacement in replacements:
    text = re.sub(pattern, replacement, text)

p.write_text(text)
print(f"Prepared runtime script: {p}")
PY

echo "=== Runtime training script ==="
cat "${RUNTIME_TRAIN_SCRIPT}"
echo

echo "=== Training ==="
bash "${RUNTIME_TRAIN_SCRIPT}"
BASH_EOF
)

###############################################################################
# Run inside Apptainer
###############################################################################
srun apptainer exec --nv --cleanenv \
  --mount "type=bind,src=$RUN_ROOT,dst=/tmp/tmpdir" \
  "${REPO_MOUNT_ARGS[@]}" \
  "$LOCAL_SIF" \
  bash -lc "$COMMAND"

###############################################################################
# Collect outputs
###############################################################################
echo "Collecting outputs..."
if [[ -d "$RUN_ROOT/output" ]]; then
  echo "Copying outputs to s3v2:$S3_OUTPUT/${SLURM_JOB_ID}/"
  rclone copy --progress "$RUN_ROOT/output" "s3v2:$S3_OUTPUT/${SLURM_JOB_ID}/"
fi

if [[ -n "${RUN_ROOT:-}" && -d "$RUN_ROOT" ]]; then
  rm -rf "$RUN_ROOT"
fi