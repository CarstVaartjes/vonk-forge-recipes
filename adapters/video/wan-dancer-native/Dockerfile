FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4

LABEL org.opencontainers.image.source="https://github.com/Wan-Video/Wan-Dancer" \
      org.opencontainers.image.revision="e6c87a94ec733230dac15b924c015f6e6501e618" \
      org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

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

COPY vendor/wan-dancer-e6c87a94ec733230dac15b924c015f6e6501e618.tar.gz.part-000 /tmp/wan-dancer.tar.gz.part-000
COPY vendor/wan-dancer-e6c87a94ec733230dac15b924c015f6e6501e618.tar.gz.part-001 /tmp/wan-dancer.tar.gz.part-001
COPY vendor/wan-dancer-e6c87a94ec733230dac15b924c015f6e6501e618.tar.gz.part-002 /tmp/wan-dancer.tar.gz.part-002
COPY vendor/wan-dancer-e6c87a94ec733230dac15b924c015f6e6501e618.tar.gz.part-003 /tmp/wan-dancer.tar.gz.part-003
COPY patch-upstream.py /tmp/patch-upstream.py
RUN printf '%s  %s\n' \
        '4af92f148fc0b1768ea3f3582fa001c52aab560edf5f56a433c087ca893b0ba5' '/tmp/wan-dancer.tar.gz.part-000' \
        'cae05fdb1e077c83c6bb3f05ec9d40ba73ff1f381874705ce2a72fbc2a638324' '/tmp/wan-dancer.tar.gz.part-001' \
        '0504a1411ce41470194b8c96f0a0d7b2e7b27b2fe1f2e490b8d1dbee2e3af057' '/tmp/wan-dancer.tar.gz.part-002' \
        '0aa0cc7a62b187799a35216360726022645bccb4192ba7f5fbe6140f13835dde' '/tmp/wan-dancer.tar.gz.part-003' \
        | sha256sum --check --strict - \
    && cat /tmp/wan-dancer.tar.gz.part-000 \
        /tmp/wan-dancer.tar.gz.part-001 \
        /tmp/wan-dancer.tar.gz.part-002 \
        /tmp/wan-dancer.tar.gz.part-003 \
        > /tmp/wan-dancer.tar.gz \
    && printf '%s  %s\n' \
        '92c529d7727c75c6515ea990d27883a45bf566587cc9f5d325a0a488b9fa1649' \
        '/tmp/wan-dancer.tar.gz' \
        | sha256sum --check --strict - \
    && rm /tmp/wan-dancer.tar.gz.part-000 \
        /tmp/wan-dancer.tar.gz.part-001 \
        /tmp/wan-dancer.tar.gz.part-002 \
        /tmp/wan-dancer.tar.gz.part-003 \
    && install --directory --mode=0755 /opt/wan-dancer \
    && tar --extract --gzip --file=/tmp/wan-dancer.tar.gz \
        --directory=/opt/wan-dancer --strip-components=1 \
    && python3 /tmp/patch-upstream.py /opt/wan-dancer \
    && rm /tmp/wan-dancer.tar.gz /tmp/patch-upstream.py \
    && python3 -m compileall -q /opt/wan-dancer

COPY run.py /opt/vonk/source/run.py
COPY NOTICE /opt/vonk/source/NOTICE
RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' \
        > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
