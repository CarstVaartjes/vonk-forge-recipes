FROM nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1392f431f89f143d0d6d0fa397a2b9a6a236f8b3628cfd3afbf21e15ab4a98

LABEL org.opencontainers.image.source="https://github.com/microsoft/TRELLIS.2" \
      org.opencontainers.image.revision="75fbf0183001ed9876c8dbb35de6b68552ee08bd" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.pixal3d.revision="f7cf38429b0bd264f1995f0f8743a88b1c728b94" \
      io.vonk.platform="linux/arm64"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="12.1" \
    NATTEN_CUDA_ARCH="12.1" \
    NATTEN_N_WORKERS="4" \
    MAX_JOBS="4" \
    ATTN_BACKEND="sdpa" \
    SPARSE_ATTN_BACKEND="sdpa" \
    SPARSE_CONV_BACKEND="flex_gemm" \
    FLEX_GEMM_AUTOTUNE_CACHE_PATH="/opt/vonk/flexgemm-source/autotune_cache.json" \
    FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE="0" \
    PYTHONPATH="/opt/vonk/trellis2-source:/opt/vonk/pixal3d-source"

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libegl1 \
        libgl1 \
        libjpeg-dev \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu132 \
        torch==2.13.0 \
        torchvision==0.28.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        accelerate==1.13.0 \
        diffusers==0.37.1 \
        easydict==1.13 \
        einops==0.8.1 \
        huggingface-hub==0.36.0 \
        imageio==2.37.2 \
        imageio-ffmpeg==0.6.0 \
        kornia==0.8.2 \
        ninja==1.13.0 \
        numpy==2.3.5 \
        opencv-python-headless==4.12.0.88 \
        packaging==26.0 \
        Pillow==12.0.0 \
        plyfile==1.1.3 \
        psutil==7.2.2 \
        safetensors==0.8.0 \
        timm==1.0.22 \
        tqdm==4.67.1 \
        transformers==4.57.3 \
        trimesh==4.10.1 \
        zstandard==0.25.0

RUN git clone --filter=blob:none --no-checkout https://github.com/microsoft/TRELLIS.2.git /opt/vonk/trellis2-source \
    && git -C /opt/vonk/trellis2-source checkout 75fbf0183001ed9876c8dbb35de6b68552ee08bd \
    && git -C /opt/vonk/trellis2-source submodule update --init --recursive \
    && git clone --filter=blob:none --no-checkout https://github.com/TencentARC/Pixal3D.git /opt/vonk/pixal3d-source \
    && git -C /opt/vonk/pixal3d-source checkout f7cf38429b0bd264f1995f0f8743a88b1c728b94 \
    && git clone --filter=blob:none --no-checkout https://github.com/valeoai/NAF.git /opt/vonk/naf-source \
    && git -C /opt/vonk/naf-source checkout 37f2dfc180f2de53d98bd601109c0da0dd6b0f43 \
    && git clone --filter=blob:none --no-checkout https://github.com/EasternJournalist/utils3d.git /opt/vonk/utils3d-source \
    && git -C /opt/vonk/utils3d-source checkout 9a4eb15e4021b67b12c460c7057d642626897ec8 \
    && git clone --filter=blob:none --no-checkout https://github.com/JeffreyXiang/CuMesh.git /opt/vonk/cumesh-source \
    && git -C /opt/vonk/cumesh-source checkout 12289e1062f0603f2f0d0771b02e1395d247f26f \
    && git -C /opt/vonk/cumesh-source submodule update --init --recursive \
    && git clone --filter=blob:none --no-checkout https://github.com/JeffreyXiang/FlexGEMM.git /opt/vonk/flexgemm-source \
    && git -C /opt/vonk/flexgemm-source checkout 6dd94a859c26ee8246888502eada3dd8ad85532e \
    && git -C /opt/vonk/flexgemm-source submodule update --init --recursive \
    && git clone --filter=blob:none --no-checkout https://github.com/NVlabs/nvdiffrast.git /opt/vonk/nvdiffrast-source \
    && git -C /opt/vonk/nvdiffrast-source checkout 253ac4fcea7de5f396371124af597e6cc957bfae \
    && git clone --filter=blob:none --no-checkout https://github.com/SHI-Labs/NATTEN.git /opt/vonk/natten-source \
    && git -C /opt/vonk/natten-source checkout 39375ea1d2f3b1bb284315fa5f79f6ecbebb6382

COPY patches/apply-sdpa-arm64.py /opt/vonk/source/apply-sdpa-arm64.py
RUN python3 /opt/vonk/source/apply-sdpa-arm64.py \
        /opt/vonk/trellis2-source/trellis2 \
        /opt/vonk/pixal3d-source/pixal3d \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-build-isolation --no-deps /opt/vonk/cumesh-source \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-build-isolation --no-deps /opt/vonk/flexgemm-source \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-build-isolation --no-deps /opt/vonk/nvdiffrast-source \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-build-isolation --no-deps /opt/vonk/natten-source \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-build-isolation --no-deps /opt/vonk/utils3d-source \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-build-isolation --no-deps /opt/vonk/trellis2-source/o-voxel

COPY trellis2_job.py /opt/vonk/source/trellis2_job.py
COPY pixal3d_job.py /opt/vonk/source/pixal3d_job.py
COPY glb_validation.py /opt/vonk/source/glb_validation.py

RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && printf '%s\n' '#!/bin/sh' \
        'test "$1" = "--entrypoint" || { echo "missing signed entrypoint" >&2; exit 64; }' \
        'entrypoint="$2"; shift 2' \
        'case "$entrypoint" in /opt/vonk/source/trellis2_job.py|/opt/vonk/source/pixal3d_job.py) ;; *) echo "unsupported entrypoint" >&2; exit 64;; esac' \
        'exec python3 "$entrypoint" --entrypoint "$entrypoint" "$@"' \
        > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline /opt/vonk/source/*.py \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk \
    && chown -R 10001:10001 /opt/vonk

USER 10001:10001
ENTRYPOINT []
