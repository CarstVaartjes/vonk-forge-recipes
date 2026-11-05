FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/QwenLM/Qwen3-TTS" \
      org.opencontainers.image.revision="6cafe5582caea83df269c36b1ce62d953a9cc66b" \
      org.opencontainers.image.version="0.1.1" \
      org.opencontainers.image.licenses="Apache-2.0"
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY requirements.lock qwen-tts.lock /tmp/
# qwen-tts declares gradio for its demo CLI only; its runtime closure is requirements.lock,
# so the package itself installs without dependency resolution.
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
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --require-hashes --no-deps \
        --requirement /tmp/qwen-tts.lock \
    && rm /tmp/requirements.lock /tmp/qwen-tts.lock \
    && python3 -c "import qwen_tts, torchaudio, onnxruntime, sox, soundfile"

COPY pipelines /opt/vonk/source/pipelines
RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/pipelines/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
