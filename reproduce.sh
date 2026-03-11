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
# User settings
###############################################################################
export SIF_S3="${SIF_S3:-s3min-tomasznaskret-1712063354/user/jmoska/stronger_mas.sif}"
export S3_OUTPUT="${S3_OUTPUT:-s3min-tomasznaskret-1712063354/user/jmoska/MultiGRPO/output/}"

export PREPARE_CODE_DATA="${PREPARE_CODE_DATA:-0}"
export PREPARE_MATH_DATA="${PREPARE_MATH_DATA:-1}"
export PREPARE_SOKOBAN_DATA="${PREPARE_SOKOBAN_DATA:-0}"

export TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/math/math_L1_prompt.sh}"
export MODEL_0="${MODEL_0:-Qwen/Qwen3-1.7B}"

export HF_TOKEN="${HF_TOKEN:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_ENTITY="${WANDB_ENTITY:-moska-phd-research}"
export WANDB_PROJECT="${WANDB_PROJECT:-pettingllms-quickstart}"
export WANDB_NAME="${WANDB_NAME:-first_run}"

###############################################################################
# Repo location on host
###############################################################################
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
echo "Using HOST_REPO_DIR=$HOST_REPO_DIR"
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

mkdir -p \
  "$RUN_ROOT" \
  "$LOCAL_HF_HOME" \
  "$LOCAL_WANDB_DIR" \
  "$LOCAL_TRITON_CACHE" \
  "$LOCAL_TORCH_EXTENSIONS" \
  "$LOCAL_APPTAINER_CACHE" \
  "$LOCAL_OUTPUT"

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
export APPTAINERENV_TMPDIR="/tmp/tmpdir"

export APPTAINERENV_HF_TOKEN="${HF_TOKEN}"
export APPTAINERENV_HF_HOME="/tmp/tmpdir/huggingface"
export APPTAINERENV_TRANSFORMERS_CACHE="/tmp/tmpdir/huggingface/transformers"
export APPTAINERENV_HUGGINGFACE_HUB_CACHE="/tmp/tmpdir/huggingface/hub"
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=1

export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
export APPTAINERENV_WANDB_ENTITY="${WANDB_ENTITY}"
export APPTAINERENV_WANDB_PROJECT="${WANDB_PROJECT}"
export APPTAINERENV_WANDB_NAME="${WANDB_NAME}"
export APPTAINERENV_WANDB_DIR="/tmp/tmpdir/wandb"
export APPTAINERENV_WANDB_CACHE_DIR="/tmp/tmpdir/wandb/.cache"
export APPTAINERENV_WANDB_CONFIG_DIR="/tmp/tmpdir/wandb/.config"

export APPTAINERENV_TRITON_CACHE_DIR="/tmp/tmpdir/triton"
export APPTAINERENV_TORCH_EXTENSIONS_DIR="/tmp/tmpdir/torch_extensions"

# Point Python at mounted repo + vendored verl
export APPTAINERENV_PYTHONPATH="/workspace/PettingLLMs:/workspace/PettingLLMs/verl:${PYTHONPATH:-}"

# Force non-V1 vLLM in container environment
export APPTAINERENV_VLLM_USE_V1=0
export APPTAINERENV_VLLM_USE_FLASHINFER_SAMPLER=0

###############################################################################
# Command run inside container
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
mkdir -p datasets

echo "=== Mounted repo ==="
pwd
ls -lah .

echo "=== Environment preflight ==="
python3 - <<'PY'
import os
import traceback

print("VLLM_USE_V1 =", os.environ.get("VLLM_USE_V1"))
print("VLLM_USE_FLASHINFER_SAMPLER =", os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"))
print("HF_HOME =", os.environ.get("HF_HOME"))
print("PYTHONPATH =", os.environ.get("PYTHONPATH"))

print("\\n=== Version check ===")
try:
    import numpy, scipy, transformers
    print("numpy:", numpy.__version__, numpy.__file__)
    print("scipy:", scipy.__version__, scipy.__file__)
    print("transformers:", transformers.__version__, transformers.__file__)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)

print("\\n=== sklearn import check ===")
try:
    import importlib.util
    print("sklearn available:", importlib.util.find_spec("sklearn") is not None)
except Exception:
    traceback.print_exc()
    raise SystemExit(1)

print("\\n=== SciPy import check ===")
try:
    from scipy.special import comb
    print("scipy.special OK")
except Exception:
    traceback.print_exc()
    raise SystemExit(1)

print("\\n=== Transformers import check ===")
try:
    from transformers import AutoModelForCausalLM, GenerationMixin
    print("transformers imports OK")
except Exception:
    traceback.print_exc()
    raise SystemExit(1)

print("\\n=== verl import check ===")
try:
    from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
    print("verl imports OK")
except Exception:
    traceback.print_exc()
    raise SystemExit(1)
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
ls -lah datasets || true
ls -lah datasets/code || true
ls -lah datasets/math || true
ls -lah datasets/sudoku_environments || true

echo "=== Training setup ==="

# Force non-V1 vLLM in this shell too
export VLLM_USE_V1=0
export VLLM_USE_FLASHINFER_SAMPLER=0

# Patch upstream script if it hardcodes these values
sed -i 's/^export VLLM_USE_V1=.*/export VLLM_USE_V1=0/' "${TRAIN_SCRIPT}" || true
sed -i 's/^export VLLM_USE_FLASHINFER_SAMPLER=.*/export VLLM_USE_FLASHINFER_SAMPLER=0/' "${TRAIN_SCRIPT}" || true

# Patch placeholder model path if present
sed -i 's|base_models.policy_0.path="your base model path"|base_models.policy_0.path="${MODEL_0}"|g' "${TRAIN_SCRIPT}" || true

# Remove invalid Hydra override if still present
sed -i 's|training.resample_freq=3\\\\||g' "${TRAIN_SCRIPT}" || true

echo "Runtime VLLM_USE_V1=\${VLLM_USE_V1:-unset}"
echo "Runtime VLLM_USE_FLASHINFER_SAMPLER=\${VLLM_USE_FLASHINFER_SAMPLER:-unset}"
grep -n "VLLM_USE_V1" "${TRAIN_SCRIPT}" || true
grep -n "VLLM_USE_FLASHINFER_SAMPLER" "${TRAIN_SCRIPT}" || true

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
  --mount type=bind,src="$HOST_REPO_DIR",dst=/workspace/PettingLLMs \
  "$LOCAL_SIF" \
  bash -c "$COMMAND"

###############################################################################
# Collect outputs
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