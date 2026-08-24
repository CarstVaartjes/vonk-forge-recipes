FROM nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1392f431f89f143d0d6d0fa397a2b9a6a236f8b3628cfd3afbf21e15ab4a98

LABEL org.opencontainers.image.source="https://github.com/stepfun-ai/Step1X-3D" \
      org.opencontainers.image.revision="cb5ac944709c6c913109070c7b90c3447f57f3d4" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.step1x.model-revision="bf7084495b3a72222f36549b7942948aa4d9daa7"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/step1x \
    TORCH_CUDA_ARCH_LIST=12.1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgl1 \
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
        accelerate==1.5.2 \
        diffusers==0.32.2 \
        easydict==1.13 \
        einops==0.8.0 \
        huggingface-hub==0.26.2 \
        imageio==2.34.1 \
        jaxtyping==0.2.28 \
        kornia==0.8.0 \
        numpy==1.26.4 \
        omegaconf==2.3.0 \
        onnxruntime==1.21.0 \
        opencv-python-headless==4.10.0.84 \
        pillow==10.3.0 \
        plyfile==1.0.3 \
        PyMCubes==0.1.4 \
        pybind11==2.13.6 \
        pygltflib==1.16.4 \
        rembg==2.0.65 \
        safetensors==0.4.3 \
        scikit-image==0.23.2 \
        scipy==1.13.0 \
        sentencepiece==0.2.0 \
        timm==0.9.16 \
        tokenizers==0.21.0 \
        tqdm==4.66.2 \
        transformers==4.48.0 \
        trimesh==4.3.2 \
        xatlas==0.0.10 \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-build-isolation \
        "nvdiffrast @ git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae" \
        "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@fdaf9bd6fed7977e4c2056e7c77c640781e58fcd"

RUN curl --fail --location --silent --show-error \
        https://github.com/stepfun-ai/Step1X-3D/archive/cb5ac944709c6c913109070c7b90c3447f57f3d4.tar.gz \
        --output /tmp/step1x.tar.gz \
    && printf '%s  %s\n' \
        02c3643bf8fd421f4dbed358965c886d6cf5f6932fb9b6c06964dc9914493226 \
        /tmp/step1x.tar.gz | sha256sum --check --strict \
    && mkdir -p /opt/step1x \
    && tar --extract --gzip --file /tmp/step1x.tar.gz \
        --directory /opt/step1x --strip-components=1 \
    && rm /tmp/step1x.tar.gz

COPY prepare_upstream.py /tmp/prepare_upstream.py
RUN python3 /tmp/prepare_upstream.py \
    && rm /tmp/prepare_upstream.py \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-build-isolation \
        /opt/step1x/step1x3d_texture/custom_rasterizer \
        /opt/step1x/step1x3d_texture/differentiable_renderer

COPY run.py geometry.py label-geometry.py texture.py /opt/vonk/source/

RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && printf '%s\n' '#!/bin/sh' \
        'test "$1" = "--entrypoint" || { echo "missing signed entrypoint" >&2; exit 64; }' \
        'entrypoint="$2"; shift 2' \
        'exec python3 "$entrypoint" --entrypoint "$entrypoint" "$@"' \
        > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline /opt/vonk/source/*.py \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk \
    && chown -R 10001:10001 /opt/vonk

USER 10001:10001
ENTRYPOINT []
