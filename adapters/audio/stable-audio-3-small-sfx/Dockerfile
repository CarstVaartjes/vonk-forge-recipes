FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/Stability-AI/stable-audio-3" \
      org.opencontainers.image.revision="790793b89ddceea612be9ce42ff6830d18f6fa57" \
      org.opencontainers.image.licenses="MIT"
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/stable-audio-3

COPY requirements.lock /tmp/requirements.lock
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3-dev \
        ca-certificates \
        ffmpeg \
        libsndfile1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.lock \
    && rm /tmp/requirements.lock

COPY vendor/stable-audio-3-790793b89ddceea612be9ce42ff6830d18f6fa57.tar.gz /tmp/stable-audio-3.tar.gz
COPY patch-offline-models.py /tmp/patch-offline-models.py
RUN printf '%s\n' \
        '2afaf90f477b1e428540820e1e7775d884e4eb557fd3d887cb132ed0d395a9b4  /tmp/stable-audio-3.tar.gz' \
        | sha256sum --check --strict - \
    && install --directory --mode=0755 /opt/stable-audio-3 \
    && tar --extract --gzip --file=/tmp/stable-audio-3.tar.gz \
        --directory=/opt/stable-audio-3 --strip-components=1 \
    && python3 /tmp/patch-offline-models.py /opt/stable-audio-3 \
    && rm /tmp/stable-audio-3.tar.gz /tmp/patch-offline-models.py \
    && python3 -c 'import soundfile, torchaudio, torchcodec; from transformers import T5GemmaEncoderModel; from stable_audio_3 import StableAudioModel'

COPY pipelines /opt/vonk/source/pipelines
RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/pipelines/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
