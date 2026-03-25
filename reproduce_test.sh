#!/bin/bash
#SBATCH --job-name=multi-grpo
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=8:00:00
#SBATCH --mem=0
#SBATCH -p lem-gpu-short
#SBATCH --verbose
#SBATCH --gres=gpu:hopper:1

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

# Mode B constants
export CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR:-/workspace/PettingLLMs}"
export IMAGE_PYTHON="${IMAGE_PYTHON:-/opt/venv/bin/python}"

###############################################################################
# Repo location on host
###############################################################################
START_DIR="${HOST_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
HOST_REPO_DIR="$START_DIR"

while [[ "$HOST_REPO_DIR" != "/" ]]; do
  if [[ -f "$HOST_REPO_DIR/requirements_venv.txt" && \
        -d "$HOST_REPO_DIR/pettingllms" && \
        -d "$HOST_REPO_DIR/scripts" && \
        -d "$HOST_REPO_DIR/verl" ]]; then
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

if [[ ! -d "$HOST_REPO_DIR/verl/verl" ]]; then
  echo "ERROR: Host repo does not have initialized submodules."
  echo "Run: git submodule update --init --recursive"
  exit 1
fi

export HOST_REPO_DIR
echo "Using host repo: $HOST_REPO_DIR"
ls -lah "$HOST_REPO_DIR"

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
export APPTAINERENV_PATH="/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_PYTHONPATH="${CONTAINER_REPO_DIR}:${CONTAINER_REPO_DIR}/verl"
export APPTAINERENV_IMAGE_PYTHON="${IMAGE_PYTHON}"
export APPTAINERENV_CONTAINER_REPO_DIR="${CONTAINER_REPO_DIR}"

export APPTAINERENV_GPU_num="${GPU_num}"
export APPTAINERENV_MODEL_0="${MODEL_0}"
export APPTAINERENV_TRAIN_SCRIPT="${TRAIN_SCRIPT}"
export APPTAINERENV_PREPARE_CODE_DATA="${PREPARE_CODE_DATA}"
export APPTAINERENV_PREPARE_MATH_DATA="${PREPARE_MATH_DATA}"
export APPTAINERENV_PREPARE_SOKOBAN_DATA="${PREPARE_SOKOBAN_DATA}"

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

export APPTAINERENV_VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export APPTAINERENV_VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export APPTAINERENV_VLLM_USE_V1="${VLLM_USE_V1:-1}"
export APPTAINERENV_VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export APPTAINERENV_TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export APPTAINERENV_NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export APPTAINERENV_NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-0}"

###############################################################################
# Command run inside container
###############################################################################
COMMAND=$(cat <<'BASH_EOF'
set -euo pipefail

export PATH="/opt/venv/bin:$PATH"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${CONTAINER_REPO_DIR}:${CONTAINER_REPO_DIR}/verl${PYTHONPATH:+:${PYTHONPATH}}"

cd "${CONTAINER_REPO_DIR}"

mkdir -p /tmp/tmpdir/output
mkdir -p /tmp/tmpdir/home
mkdir -p /tmp/tmpdir/huggingface/transformers
mkdir -p /tmp/tmpdir/huggingface/hub
mkdir -p /tmp/tmpdir/wandb/.cache
mkdir -p /tmp/tmpdir/wandb/.config
mkdir -p /tmp/tmpdir/triton
mkdir -p /tmp/tmpdir/torch_extensions
mkdir -p data

echo "=== Runtime paths ==="
echo "PWD: $(pwd)"
echo "HOST TRAIN_SCRIPT: ${TRAIN_SCRIPT}"
echo "MODEL_0: ${MODEL_0}"
echo "GPU_num: ${GPU_num}"
echo "WANDB_PROJECT: ${WANDB_PROJECT}"
echo "WANDB_ENTITY: ${WANDB_ENTITY}"
echo "WANDB_NAME: ${WANDB_NAME}"
echo "WANDB_MODE: ${WANDB_MODE}"
echo "PATH: ${PATH}"
echo "PYTHONPATH: ${PYTHONPATH}"
ls -lah .

echo "=== Python preflight ==="
which python
python -V
"${IMAGE_PYTHON}" -V

python - <<'PY'
import sys
import numpy as np
import numpy.core.multiarray as ma
import torch
import vllm
import datasets

print("sys.executable:", sys.executable)
print("numpy:", np.__version__, np.__file__)
print("torch:", torch.__version__)
print("vllm:", vllm.__version__)
print("datasets:", datasets.__version__)
print("numpy.core.multiarray.generic:", hasattr(ma, "generic"))
print("numpy.core.multiarray.complexfloating:", hasattr(ma, "complexfloating"))

assert sys.executable.startswith("/opt/venv/"), sys.executable
assert np.__file__.startswith("/opt/venv/"), np.__file__
assert hasattr(ma, "generic")
assert hasattr(ma, "complexfloating")
PY

echo "=== Dataset preparation ==="
if [[ "${PREPARE_CODE_DATA}" == "1" ]]; then
  python scripts/dataprocess/load_code.py
fi

if [[ "${PREPARE_MATH_DATA}" == "1" ]]; then
  python scripts/dataprocess/load_math.py
fi

if [[ "${PREPARE_SOKOBAN_DATA}" == "1" ]]; then
  python scripts/dataprocess/load_sokoban.py
fi

echo "=== Dataset directories after preparation ==="
ls -lah data || true
ls -lah data/code || true
ls -lah data/math || true
ls -lah data/sudoku_environments || true

echo "=== Prepare runtime training script copy ==="
RUNTIME_TRAIN_SCRIPT="/tmp/tmpdir/$(basename "${TRAIN_SCRIPT}")"
export RUNTIME_TRAIN_SCRIPT
cp "${TRAIN_SCRIPT}" "${RUNTIME_TRAIN_SCRIPT}"
chmod +x "${RUNTIME_TRAIN_SCRIPT}"

python - <<'PY'
import os
import re
from pathlib import Path

p = Path(os.environ["RUNTIME_TRAIN_SCRIPT"])
src = p.read_text()

changes = {}

def sub_once(pattern, repl, label, flags=0):
    global src
    new_src, n = re.subn(pattern, repl, src, count=1, flags=flags)
    changes[label] = n
    src = new_src

# Use the image venv python, not system python
sub_once(r'\bpython3\s+-m\s+', 'python -m ', 'python3->python')

# Do not let upstream script collapse visibility to a single GPU
sub_once(r'export CUDA_VISIBLE_DEVICES=0\s*', '', 'drop CUDA_VISIBLE_DEVICES=0')
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
  --mount "type=bind,src=$HOST_REPO_DIR,dst=$CONTAINER_REPO_DIR" \
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
