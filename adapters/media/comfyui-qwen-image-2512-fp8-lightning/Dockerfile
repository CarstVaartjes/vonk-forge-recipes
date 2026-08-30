FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4

ARG DEBIAN_FRONTEND=noninteractive
ARG COMFYUI_REVISION=7a131a3afadc8200120f67f9236311a2c48b7445
ARG COMFYUI_ARCHIVE_SHA256=7e123716ae698194b3ded7ecbd8028b792d9015ce56d2318ebf4b8066efc6016

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
      torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir -r /opt/comfyui/requirements.txt

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
