#!/bin/bash
#SBATCH --job-name=multi-grpo
#SBATCH --nodes=2
#SBATCH --cpus-per-gpu=4
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH -p lem-gpu-short
#SBATCH --verbose
#SBATCH --gres=gpu:hopper:4

set -euo pipefail


export SIF_S3="${SIF_S3:-s3min-tomasznaskret-1712063354/user/jmoska/stronger_mas_old_v2.sif}"
export S3_OUTPUT="${S3_OUTPUT:-s3min-tomasznaskret-1712063354/user/jmoska/MultiGRPO/output/}"

export PREPARE_CODE_DATA="${PREPARE_CODE_DATA:-0}"
export PREPARE_MATH_DATA="${PREPARE_MATH_DATA:-1}"
export PREPARE_SOKOBAN_DATA="${PREPARE_SOKOBAN_DATA:-0}"

export TRAIN_SCRIPT="${TRAIN_SCRIPT:-scripts/train/math/math_L1_prompt.sh}"
export MODEL_0="${MODEL_0:-Qwen/Qwen3-8B}"
export NNODES="${NNODES:-${SLURM_NNODES:-2}}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"


export WANDB_ENTITY="moska-phd-research"
export WANDB_PROJECT="multi-grpo"
export WANDB_NAME="first_run_qwen8"

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
export APPTAINERENV_HF_HUB_ENABLE_HF_TRANSFER=0

export APPTAINERENV_WANDB_API_KEY="${WANDB_API_KEY}"
export APPTAINERENV_WANDB_ENTITY="${WANDB_ENTITY}"
export APPTAINERENV_WANDB_PROJECT="${WANDB_PROJECT}"
export APPTAINERENV_WANDB_NAME="${WANDB_NAME}"
export APPTAINERENV_WANDB_DIR="/tmp/tmpdir/wandb"
export APPTAINERENV_WANDB_CACHE_DIR="/tmp/tmpdir/wandb/.cache"
export APPTAINERENV_WANDB_CONFIG_DIR="/tmp/tmpdir/wandb/.config"

export APPTAINERENV_TRITON_CACHE_DIR="/tmp/tmpdir/triton"
export APPTAINERENV_TORCH_EXTENSIONS_DIR="/tmp/tmpdir/torch_extensions"
export APPTAINERENV_NNODES="${NNODES}"
export APPTAINERENV_N_GPUS_PER_NODE="${N_GPUS_PER_NODE}"
export APPTAINERENV_GPU_num="${N_GPUS_PER_NODE}"

# Point Python at mounted repo + vendored verl
export APPTAINERENV_PYTHONPATH="/workspace/PettingLLMs:/workspace/PettingLLMs/verl:${PYTHONPATH:-}"


###############################################################################
# Ray bootstrap
###############################################################################
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

if [[ "$head_node_ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<<"$head_node_ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
  else
    head_node_ip=${ADDR[0]}
  fi
fi

ray_port="${RAY_PORT:-6379}"
export ip_head="${head_node_ip}:${ray_port}"
export RAY_ADDRESS="${ip_head}"
export APPTAINERENV_ip_head="${ip_head}"
export APPTAINERENV_RAY_ADDRESS="${RAY_ADDRESS}"

start_ray_on_node() {
  local node_name="$1"
  local ray_args="$2"
  srun --overlap --nodes=1 --ntasks=1 -w "$node_name" apptainer exec --nv \
    --mount type=bind,src="$RUN_ROOT",dst=/tmp/tmpdir \
    --mount type=bind,src="$HOST_REPO_DIR",dst=/workspace/PettingLLMs \
    "$LOCAL_SIF" \
    bash -lc "cd /workspace/PettingLLMs && ray start ${ray_args} --num-cpus ${SLURM_CPUS_PER_TASK:-16} --num-gpus ${N_GPUS_PER_NODE} --block" &
}

echo "Starting Ray head on ${head_node} at ${ip_head}"
start_ray_on_node "$head_node" "--head --node-ip-address=${head_node_ip} --port=${ray_port}"
sleep 10

worker_num=$((NNODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "Starting Ray worker ${i} on ${node_i}"
  start_ray_on_node "$node_i" "--address ${ip_head}"
  sleep 5
done

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
mkdir -p data


echo "=== Mounted repo ==="
pwd
ls -lah .

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

echo "=== Training setup ==="

# Patch placeholder model path if present
sed -i 's|base_models.policy_0.path="your base model path"|base_models.policy_0.path="${MODEL_0}"|g' "${TRAIN_SCRIPT}" || true


echo "=== Training ==="
echo "Running: bash ${TRAIN_SCRIPT}"
bash "${TRAIN_SCRIPT}"
BASH_EOF
)

###############################################################################
# Run inside Apptainer
###############################################################################
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" apptainer exec --nv \
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

if [[ -n "${RUN_ROOT:-}" && -d "$RUN_ROOT" ]]; then
  rm -rf "$RUN_ROOT"
fi
