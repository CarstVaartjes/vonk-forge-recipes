FROM nvcr.io/nvidia/cuda:13.2.1-runtime-ubuntu24.04@sha256:a52783d8d73ace53998d4e740515e9942e73072dc7fbd5322917eb382a0bc7fb

LABEL org.opencontainers.image.source="https://github.com/OpenMOSS/MOVA" \
      org.opencontainers.image.revision="83bcbe8eab55ded920889239af93c112834ee5b8" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.mova.adapter="synchronized-video-audio-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1

ARG MOVA_REVISION=83bcbe8eab55ded920889239af93c112834ee5b8
ARG MOVA_ARCHIVE_SHA256=739c7c9754fc4e9fa374ac61b23df872becb1919443ff9b80777126af2fefec3

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        curl \
        ffmpeg \
        libsndfile1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && install --directory --mode=0755 /opt/mova-source \
    && curl --fail --location --silent --show-error \
        "https://github.com/OpenMOSS/MOVA/archive/${MOVA_REVISION}.tar.gz" \
        --output /tmp/mova.tar.gz \
    && printf '%s  %s\n' "${MOVA_ARCHIVE_SHA256}" /tmp/mova.tar.gz | sha256sum --check --strict \
    && tar --extract --gzip --file /tmp/mova.tar.gz --directory /opt/mova-source --strip-components=1 \
    && rm /tmp/mova.tar.gz

COPY requirements.lock /opt/vonk/build/requirements.lock

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --constraint /opt/vonk/build/requirements.lock \
        --index-url https://download.pytorch.org/whl/cu132 \
        torch==2.13.0 \
        torchvision==0.28.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
        --constraint /opt/vonk/build/requirements.lock \
        --index-url https://download.pytorch.org/whl/test/cu132 \
        torchaudio==2.11.0

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --constraint /opt/vonk/build/requirements.lock \
        accelerate==1.14.0 \
        descript-audiotools==0.7.2 \
        diffusers==0.36.0 \
        einops==0.8.2 \
        ftfy==6.3.1 \
        huggingface-hub==0.36.0 \
        imageio==2.37.4 \
        imageio-ffmpeg==0.6.0 \
        mmengine==0.10.7 \
        numpy==2.3.4 \
        Pillow==12.3.0 \
        protobuf==3.19.6 \
        safetensors==0.8.0 \
        sentencepiece==0.2.2 \
        tqdm==4.70.0 \
        transformers==4.57.6 \
        yunchang==0.6.4

COPY patches/apply-build-patches.py /opt/vonk/build/apply-build-patches.py

RUN python3 /opt/vonk/build/apply-build-patches.py

RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps --no-build-isolation /opt/mova-source \
    && python3 -m pip check \
    && python3 -c 'from mova.diffusion.pipelines.pipeline_mova import MOVA; print(MOVA.__name__)' \
    && ffprobe -version

COPY NOTICE /opt/vonk/source/NOTICE
COPY run.py /opt/vonk/source/pipelines/run.py

RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source/pipelines \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/pipelines/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
