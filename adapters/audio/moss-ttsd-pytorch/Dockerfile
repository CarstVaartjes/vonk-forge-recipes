FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/OpenMOSS/MOSS-TTS" \
      org.opencontainers.image.revision="934d6826b084c46a0d033402174d5f8ac4ed2519" \
      org.opencontainers.image.version="MOSS-TTSD v1.0" \
      org.opencontainers.image.licenses="Apache-2.0"
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY requirements.lock /tmp/
# The creator's torch 2.9.1+cu128 / transformers 5.0.0 stack; the model code itself is the
# pinned Model's remote code (trust_remote_code), so no MOSS-TTS source is installed.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3-dev \
        ca-certificates \
        libsndfile1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
        --requirement /tmp/requirements.lock \
    && rm /tmp/requirements.lock \
    && python3 -c "import torch, torchaudio, transformers, soundfile, librosa, einops"

COPY pipelines /opt/vonk/source/pipelines
RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/pipelines/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
