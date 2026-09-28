FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/OpenMOSS/MOSS-VL" \
      org.opencontainers.image.revision="288cc2dcd145e53a3baf9c11fc9be298cf1b30d7" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.media.adapter="moss-vl-realtime-session-v1"

ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY requirements.lock /tmp/requirements.lock
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        ffmpeg \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.lock \
    && rm /tmp/requirements.lock

COPY NOTICE session.schema.json example-session.json input_contract.py run.py /opt/vonk/source/
RUN install --directory --mode=0755 /opt/vonk/bin /inputs /outputs \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk \
    && chown 10001:10001 /outputs

USER 10001:10001
ENTRYPOINT []
