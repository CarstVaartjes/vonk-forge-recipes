FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_REVISION=83071e1aec311d31e773d64d6872181b3bad0fe2
ARG COMFYUI_ARCHIVE_SHA256=fbcb570f26aaf67911d15e164907a0992fb9d5780ea17546f1e0476e9943977c

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl ffmpeg python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/${COMFYUI_REVISION}" -o /tmp/comfyui.tar.gz \
    && echo "${COMFYUI_ARCHIVE_SHA256}  /tmp/comfyui.tar.gz" | sha256sum -c - \
    && mkdir -p /opt/comfyui \
    && tar -xzf /tmp/comfyui.tar.gz --strip-components=1 -C /opt/comfyui \
    && rm /tmp/comfyui.tar.gz

RUN python3 -m pip install --break-system-packages --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.14.0 torchvision==0.29.0 torchaudio==2.11.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir -r /opt/comfyui/requirements.txt

COPY comfyui_runtime_smoke.py /tmp/comfyui_runtime_smoke.py
RUN cd /opt/comfyui \
    && PYTHONPATH=/opt/comfyui python3 /tmp/comfyui_runtime_smoke.py \
    && rm /tmp/comfyui_runtime_smoke.py

RUN groupadd --gid 10001 vonk \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin vonk \
    && mkdir -p /opt/vonk/bin /opt/vonk/source/workflows /outputs \
    && chown -R 10001:10001 /opt/vonk /outputs

COPY --chown=10001:10001 comfyui_job.py /opt/vonk/bin/comfyui-job
COPY --chown=10001:10001 workflows/ /opt/vonk/source/workflows/

RUN chmod 0555 /opt/vonk/bin/comfyui-job

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

USER 10001:10001
WORKDIR /opt/comfyui
ENTRYPOINT []
