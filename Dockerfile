FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG TORCH_CUDA_ARCH_LIST=9.0
ARG MAX_JOBS=8

ENV TZ=Etc/UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_USE_V1=1 \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    NCCL_IB_DISABLE=1 \
    NCCL_NET_GDR_LEVEL=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/PettingLLMs-build

COPY . .

RUN bash setup.bash

ENV PATH="/opt/PettingLLMs-build/pettingllms_venv/bin:${PATH}"

WORKDIR /workspace/PettingLLMs

CMD ["/bin/bash"]
