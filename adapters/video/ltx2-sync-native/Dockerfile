FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/Lightricks/LTX-2" \
      org.opencontainers.image.revision="e98f6d927d83c92ef2be6aec76c8440b19306e39" \
      org.opencontainers.image.licenses="LTX-2 Community License Agreement" \
      io.vonk.agent-protocol.source-commit="cee017ae806f0ac4c0842a3652f8ef7ee57388b8" \
      io.vonk.agent-protocol.wheel-sha256="9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625" \
      io.vonk.ltx2.source-archive-sha256="80daf4cd8e3b29de8a0282f9fe11998ef60f9e38cff6534fdc5dc8e6f3cc7ff3"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3-dev \
        ca-certificates \
        curl \
        ffmpeg \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0 torchvision==0.26.0 \
        torchaudio==2.11.0

COPY requirements.lock /opt/vonk/source/requirements.lock
COPY wheels/vonk_agent_protocol-4.1.0-py3-none-any.whl /opt/vonk/source/wheels/vonk_agent_protocol-4.1.0-py3-none-any.whl

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --requirement /opt/vonk/source/requirements.lock \
    && printf '%s  %s\n' \
        9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625 \
        /opt/vonk/source/wheels/vonk_agent_protocol-4.1.0-py3-none-any.whl \
        | sha256sum --check --strict \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-index \
        /opt/vonk/source/wheels/vonk_agent_protocol-4.1.0-py3-none-any.whl \
    && curl --fail --location --silent --show-error \
        --output /tmp/ltx2-source.tar.gz \
        https://codeload.github.com/Lightricks/LTX-2/tar.gz/e98f6d927d83c92ef2be6aec76c8440b19306e39 \
    && printf '%s  %s\n' \
        80daf4cd8e3b29de8a0282f9fe11998ef60f9e38cff6534fdc5dc8e6f3cc7ff3 \
        /tmp/ltx2-source.tar.gz \
        | sha256sum --check --strict \
    && mkdir -p /opt/vonk/source/ltx2 \
    && tar --extract --gzip --file /tmp/ltx2-source.tar.gz \
        --directory /opt/vonk/source/ltx2 --strip-components=1 \
    && rm /tmp/ltx2-source.tar.gz \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --no-build-isolation --no-deps /opt/vonk/source/ltx2/packages/ltx-core \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --no-build-isolation --no-deps /opt/vonk/source/ltx2/packages/ltx-pipelines

COPY run.py /opt/vonk/source/run.py

RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' \
        > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

# Import the real pipeline before publishing an image; lazy transformers imports
# otherwise hide binary torch/torchaudio incompatibilities until the first job.
RUN python3 -c "import torchaudio; from transformers import PreTrainedModel; from ltx_pipelines.utils.types import PipelineOutput"

USER 10001:10001
ENTRYPOINT []
