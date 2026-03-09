FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PETTINGLLMS_REPO=https://github.com/pettingllms-ai/PettingLLMs.git
ARG PETTINGLLMS_REF=main
ARG TORCH_CUDA_ARCH_LIST=9.0

ENV TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=8 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_USE_V1=1 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    python-is-python3 \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    pkg-config \
    ninja-build \
    libaio-dev \
    libnuma-dev \
    libibverbs-dev \
    openssh-client \
    rsync \
    jq \
    unzip \
    wget \
    && curl -fsSL https://rclone.org/install.sh | bash \
    && rm -rf /var/lib/apt/lists/*

# Use a virtualenv on Ubuntu 24.04 to avoid externally-managed Python issues.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /workspace
RUN git clone --recursive --branch ${PETTINGLLMS_REF} ${PETTINGLLMS_REPO} PettingLLMs \
    && cd /workspace/PettingLLMs \
    && git submodule update --init --recursive

WORKDIR /workspace/PettingLLMs

# PettingLLMs upstream setup currently targets Python 3.12 + torch 2.7.1 + cu128.
# We keep that tested user-space stack inside a CUDA 13.1 container so it can run on
# clusters with newer 13.x drivers while preserving the repo's known-good package set.
RUN python -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install ninja \
    && MAX_JOBS=${MAX_JOBS} python -m pip install flash-attn==2.8.3 --no-build-isolation

RUN cd /workspace/PettingLLMs/verl && python -m pip install -e .
RUN python -m pip install -r requirements_venv.txt
RUN python -m pip install -e .

# Helpful runtime defaults for Apptainer/Singularity jobs.
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    NCCL_IB_DISABLE=1 \
    NCCL_NET_GDR_LEVEL=0

WORKDIR /workspace/PettingLLMs