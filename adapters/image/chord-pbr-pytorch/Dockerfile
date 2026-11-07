FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/ubisoft/ubisoft-laforge-chord" \
      org.opencontainers.image.revision="675a29d1fee5e1b256833bd4edabd619be43d0c7" \
      org.opencontainers.image.licenses="LicenseRef-Ubisoft-Machine-Learning-License-Research-Only-Copyleft"
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/chord

COPY requirements.lock /tmp/requirements.lock
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3-dev \
        ca-certificates \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.lock \
    && rm /tmp/requirements.lock

COPY vendor/ubisoft-laforge-chord-675a29d1fee5e1b256833bd4edabd619be43d0c7.tar.gz /tmp/chord.tar.gz
RUN printf '%s\n' \
        'f16384def08e61fd5d46712ed055618a0befd22c2962c1af0c795aa02335a4f3  /tmp/chord.tar.gz' \
        | sha256sum --check --strict - \
    && install --directory --mode=0755 /opt/chord \
    && tar --extract --gzip --file=/tmp/chord.tar.gz \
        --directory=/opt/chord --strip-components=1 \
    && rm /tmp/chord.tar.gz \
    && python3 -c 'import imageio, omegaconf; from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel; from transformers import CLIPTextModel, CLIPTokenizer; from torchvision.transforms import v2; from chord import ChordModel'

COPY pipelines /opt/vonk/source/pipelines
RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/pipelines/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
