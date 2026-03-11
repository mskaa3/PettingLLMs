FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PETTINGLLMS_REPO=https://github.com/mskaa3/PettingLLMs.git
ARG PETTINGLLMS_REF=dev
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
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    NCCL_IB_DISABLE=1 \
    NCCL_NET_GDR_LEVEL=0

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

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel

RUN python -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install ninja \
    && MAX_JOBS=${MAX_JOBS} python -m pip install flash-attn==2.8.3 --no-build-isolation

WORKDIR /tmp/build

RUN git clone --recursive --branch ${PETTINGLLMS_REF} ${PETTINGLLMS_REPO} PettingLLMs \
    && cd PettingLLMs \
    && git submodule update --init --recursive \
    && python -m pip install -r requirements_venv.txt \
    && rm -rf /tmp/build

# Final compatibility override for this platform
RUN python -m pip install --no-cache-dir --force-reinstall \
    "numpy==1.26.4" \
    "scipy==1.13.1"

RUN python -m pip uninstall -y scikit-learn
RUN python -m pip install torchdata

RUN python - <<'PY'
import inspect
import re
from pathlib import Path
import vllm.v1.worker.gpu_model_runner as mod

p = Path(inspect.getfile(mod))
src = p.read_text()

pattern = re.compile(
    r"(?m)^(\s*)logit_indices = np\.cumsum\(num_scheduled_tokens\) - 1\n"
    r"\1return hidden_states, hidden_states\[logit_indices\]"
)

replacement = (
    r"\1logit_indices = np.cumsum(num_scheduled_tokens) - 1\n"
    r"\1logit_indices_device = torch.tensor(logit_indices.tolist(), dtype=torch.long, device=self.device)\n"
    r"\1return hidden_states, hidden_states[logit_indices_device]"
)

new_src, count = pattern.subn(replacement, src, count=1)

if count == 0:
    print(f"Patch target not found in {p}")
    for needle in [
        "logit_indices = np.cumsum(num_scheduled_tokens) - 1",
        "hidden_states[logit_indices]",
        "logit_indices_device = torch.from_numpy(logit_indices).to(",
    ]:
        print(f"{needle!r}: {needle in src}")
    raise SystemExit(1)

p.write_text(new_src)
print(f"Patched {p} ({count} occurrence(s))")
PY


WORKDIR /workspace/PettingLLMs