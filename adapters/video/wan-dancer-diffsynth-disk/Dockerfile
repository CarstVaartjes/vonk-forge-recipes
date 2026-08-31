FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4

LABEL org.opencontainers.image.source="https://github.com/modelscope/DiffSynth-Studio" \
      org.opencontainers.image.revision="84f93fc4907b6c193be5501bab0b5c37f383033c" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    DIFFSYNTH_SKIP_DOWNLOAD=True \
    DIFFSYNTH_DISK_MAP_BUFFER_SIZE=268435456 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/diffsynth-studio

COPY requirements.lock /tmp/requirements.lock
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        libsndfile1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.lock \
    && rm /tmp/requirements.lock

COPY vendor/diffsynth-studio-84f93fc4907b6c193be5501bab0b5c37f383033c.tar.gz /tmp/diffsynth-studio.tar.gz
COPY patch-runtime.py /tmp/patch-runtime.py
RUN printf '%s  %s\n' \
        '5f0dfef5351341613e2c8ba96806bddb576e5e1f44aaa104c5d5c388cf44bc1b' \
        '/tmp/diffsynth-studio.tar.gz' \
        | sha256sum --check --strict - \
    && install --directory --mode=0755 /opt/diffsynth-studio \
    && tar --extract --gzip --file=/tmp/diffsynth-studio.tar.gz \
        --directory=/opt/diffsynth-studio --strip-components=1 \
    && python3 /tmp/patch-runtime.py /opt/diffsynth-studio \
    && rm /tmp/diffsynth-studio.tar.gz /tmp/patch-runtime.py \
    && python3 -m compileall -q /opt/diffsynth-studio

COPY generate.py /opt/vonk/source/generate.py
COPY run.py /opt/vonk/source/run.py
COPY NOTICE /opt/vonk/source/NOTICE
RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' \
        > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
