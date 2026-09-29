# syntax=docker/dockerfile:1.6
#
# MiniMax-M3 W4A16 GPTQ + b12x + EAGLE3 serving image for two DGX Sparks: a re-pinned reproduction of
# a3refaat/spark-vllm-docker (branch minimax-m3-4bit-w4a16 at f942437) as tonyd2wild ran it, NOT a byte-identical
# rebuild of upstream's locally built vllm-node-minimax-m3-b12x image. Upstream builds Dockerfile + Dockerfile.deploy
# through build-deploy.sh with mutable inputs; this file merges its stages and pins every input it can:
#   * base image nvidia/cuda:13.2.0-devel-ubuntu24.04 by arm64 digest
#   * vLLM 979b56a66c969ab67655d2155ab4c6c5bed15f65 (the MiniMax-M3 release merge commit; fetched by SHA)
#   * FlashInfer d768c14e7cf5dd5df45a8a1de78ae815879f108a (v0.6.12, the version that vLLM ref requires), built from
#     source with upstream's flashinfer_cache.patch. Upstream normally downloads eugr's mutable
#     `prebuilt-flashinfer-current` wheels, whose July 2026 version is not recoverable.
#   * zyang-dev/nccl dgxspark-3node-ring at fab1850acd902672d79ca81c2e7fb8e1848c208c (branch head, last commit 2026-03-13)
#   * Rust 1.96.0 (the stable release on upstream's build date; upstream used `stable`)
#   * PyPI resolution frozen at 2026-07-05 (uv --exclude-newer), the day after upstream's last commit.
# Not pinnable: Ubuntu and NVIDIA apt packages (resolved at build time) and the rustup installer script.
ARG BUILD_JOBS=16

# =========================================================
# Stage 1: build image (CUDA devel, torch, NCCL mesh build, FlashInfer and vLLM wheels); every FROM pins a digest
# =========================================================
FROM docker.io/nvidia/cuda:13.2.0-devel-ubuntu24.04@sha256:a5b6256e470196fc1d5f8f62139d57d3662867746dfe1cb352d7652024047020 AS base

ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NINJAFLAGS="-j${BUILD_JOBS}" \
    MAKEFLAGS="-j${BUILD_JOBS}" \
    DG_JIT_USE_NVRTC=1 \
    USE_CUDNN=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=10 \
    UV_EXCLUDE_NEWER=2026-07-05T00:00:00Z \
    VLLM_BASE_DIR=/workspace/vllm

RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim cmake build-essential ninja-build \
    libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
    python3-dev python3-pip git wget \
    libibverbs1 libibverbs-dev rdma-core \
    devscripts debhelper fakeroot \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# The PyTorch index carries no upload dates, so torch is pinned exactly instead.
RUN UV_EXCLUDE_NEWER= uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 triton --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2" filelock pynvml requests tqdm

ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

WORKDIR $VLLM_BASE_DIR

# NCCL with mesh support (artifacts in /workspace/vllm/nccl/build/pkg/deb)
RUN git clone -b dgxspark-3node-ring https://github.com/zyang-dev/nccl.git && \
    cd nccl && git checkout --detach fab1850acd902672d79ca81c2e7fb8e1848c208c && \
    make -j ${BUILD_JOBS} src.build NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121" && \
    make pkg.debian.build && apt install -y --no-install-recommends --allow-downgrades ./build/pkg/deb/*.deb

# =========================================================
# (still stage 1) FlashInfer wheels from source
# =========================================================
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
RUN uv pip install packaging

RUN git clone --recursive https://github.com/flashinfer-ai/flashinfer.git /workspace/flashinfer && \
    cd /workspace/flashinfer && git checkout --detach d768c14e7cf5dd5df45a8a1de78ae815879f108a && \
    git submodule update --init --recursive
WORKDIR /workspace/flashinfer

# Upstream's patch to avoid re-downloading existing cubins
COPY flashinfer_cache.patch .
RUN patch -p1 < flashinfer_cache.patch && \
    sed -i -e 's/license = "Apache-2.0"/license = { text = "Apache-2.0" }/' -e '/license-files/d' pyproject.toml && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels-flashinfer -v && \
    cd flashinfer-cubin && uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels-flashinfer -v && \
    cd ../flashinfer-jit-cache && \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels-flashinfer -v && \
    cd .. && git rev-parse HEAD > /workspace/wheels-flashinfer/.flashinfer-commit

# =========================================================
# (still stage 1) vLLM wheel (MiniMax-M3 release merge + fused fp8-KV patch + Rust tool parser)
# =========================================================
WORKDIR $VLLM_BASE_DIR

# The commit is a merge on vLLM's deleted m3_release branch; GitHub serves it by SHA.
RUN git init -q vllm && cd vllm && \
    git remote add origin https://github.com/vllm-project/vllm.git && \
    git fetch -q origin 979b56a66c969ab67655d2155ab4c6c5bed15f65 && \
    git checkout -q FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "979b56a66c969ab67655d2155ab4c6c5bed15f65"
WORKDIR $VLLM_BASE_DIR/vllm

# Upstream's local vLLM patch (--apply-vllm-patch minimax-m3-fused-fp8-kv.patch) and helper scripts.
# Upstream also fetches vLLM PR 35568 and reverts commit c51df430 when needed; both are no-ops at this ref
# (PR 35568 merged 2026-05-15 and is an ancestor of the ref; the FlashInfer issue 3197 marker is absent), so they are omitted.
COPY patches/ /workspace/local-patches/
RUN git apply -v --whitespace=nowarn /workspace/local-patches/vllm/minimax-m3-fused-fp8-kv.patch

RUN python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test/cuda.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test/cuda.txt && \
    uv pip install -r requirements/build/cuda.txt

# Rust PyO3 tool parser for --tool-call-parser minimax_m3 (upstream --build-rust), toolchain fixed at 1.96.0.
ENV CARGO_HOME=/opt/rust/cargo \
    RUSTUP_HOME=/opt/rust/rustup \
    PATH=/opt/rust/cargo/bin:$PATH \
    VLLM_BUILD_RUST=1
RUN set -eux; \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup-init.sh; \
    sh /tmp/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain 1.96.0; \
    rm -f /tmp/rustup-init.sh; \
    rustc --version; cargo --version; \
    python3 /workspace/local-patches/strip-vllm-rs-binary.py .

RUN VLLM_REQUIRE_RUST_FRONTEND=1 uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels-vllm -v && \
    git rev-parse HEAD > /workspace/wheels-vllm/.vllm-commit

# =========================================================
# Stage 2: runtime image (upstream Dockerfile runner + Dockerfile.deploy b12x layer + the lane's launch-time mods)
# =========================================================
FROM docker.io/nvidia/cuda:13.2.0-devel-ubuntu24.04@sha256:a5b6256e470196fc1d5f8f62139d57d3662867746dfe1cb352d7652024047020 AS runner

ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS} \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    NINJAFLAGS="-j${BUILD_JOBS}" \
    MAKEFLAGS="-j${BUILD_JOBS}" \
    DG_JIT_USE_NVRTC=1 \
    USE_CUDNN=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1 \
    UV_LINK_MODE=copy \
    UV_EXCLUDE_NEWER=2026-07-05T00:00:00Z \
    VLLM_BASE_DIR=/workspace/vllm

COPY --from=base /workspace/vllm/nccl/build/pkg/deb /workspace/nccl-pkg
RUN apt update && \
    apt install -y --no-install-recommends \
    python3 python3-pip python3-dev vim curl git wget patch \
    libcudnn9-cuda-13 \
    libibverbs1 libibverbs-dev rdma-core \
    libxcb1 \
    && cd /workspace/nccl-pkg && apt install -y --no-install-recommends --allow-downgrades ./*.deb \
    && rm -rf /var/lib/apt/lists/* /workspace/nccl-pkg \
    && pip install uv

WORKDIR $VLLM_BASE_DIR

RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken" && \
    echo "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d  tiktoken_encodings/o200k_base.tiktoken" | sha256sum -c - && \
    echo "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7  tiktoken_encodings/cl100k_base.tiktoken" | sha256sum -c -

RUN UV_EXCLUDE_NEWER= uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 triton --index-url https://download.pytorch.org/whl/cu130 && \
    uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

COPY --from=base /workspace/wheels-flashinfer /workspace/wheels-flashinfer
COPY --from=base /workspace/wheels-vllm /workspace/wheels-vllm
RUN uv pip install /workspace/wheels-flashinfer/*.whl /workspace/wheels-vllm/*.whl && rm -rf /workspace/wheels-flashinfer /workspace/wheels-vllm

ARG TORCH_CUDA_ARCH_LIST="12.1a"
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST} \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    TIKTOKEN_ENCODINGS_BASE=/workspace/vllm/tiktoken_encodings \
    PATH=/workspace/vllm:$PATH

RUN uv pip install ray[default] fastsafetensors instanttensor

RUN rm /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2

# Dockerfile.deploy: vendored b12x kernel library on sys.path via a .pth (smoke-imported by verify-runtime.py,
# together with the vLLM surfaces the mods patch, as upstream's build does).
COPY vendor/b12x /opt/b12x
RUN printf '/opt/b12x\n' > /usr/local/lib/python3.12/dist-packages/b12x.pth

# Upstream applies the recipe's mods into a fresh container at launch (launch-cluster.sh: copy to
# /workspace/mods/<name>, cd there, ./run.sh). The lane A group (mods/minimax-m3-gptq-b12x-eagle3) expands to the
# list below; tonyd2wild's mod-prometheus-graceful-skip.patch is applied first, as his DEFAULT-CONFIG documents.
COPY mods /opt/vonk/build/mods
COPY td-patches /opt/vonk/build/td-patches
RUN set -eu; \
    cd /opt/vonk/build && patch -p1 < td-patches/mod-prometheus-graceful-skip.patch; \
    for mod in fix-prometheus-fastapi-routing fix-minimax-m3-reasoning-parser-robustness \
               fix-minimax-m3-compressed-tensors-swiglu-clamp fix-minimax-m3-indexer-proj-bf16-load \
               fix-minimax-m3-b12x-indexer fix-minimax-m3-b12x-msa fix-minimax-m3-b12x-dense \
               fix-minimax-m3-b12x-triton-prewarm fix-eagle3-draft-embed-quant; do \
        mkdir -p "/workspace/mods/$mod"; cp -a "/opt/vonk/build/mods/$mod/." "/workspace/mods/$mod/"; \
        (cd "/workspace/mods/$mod" && export WORKSPACE_DIR="$PWD" && chmod +x run.sh && ./run.sh); \
    done; \
    rm -rf /opt/vonk/build /workspace/mods

COPY verify-runtime.py /opt/vonk/verify-runtime.py
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
RUN python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
