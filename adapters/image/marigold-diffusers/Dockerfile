FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/prs-eth/Marigold" \
      org.opencontainers.image.revision="2bfbdeae5d10a50b71f1ba20c865d46e480f6010" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.adapter="marigold" \
      io.vonk.agent-protocol.wheel-sha256="9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625"

# The pipelines are mainline diffusers; the Marigold repository adds no code to the image.
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY requirements.lock /tmp/requirements.lock
COPY vonk_agent_protocol-4.1.0-py3-none-any.whl /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl
# build-essential and python3-dev: Triton JIT needs a compiler and Python.h in every
# CUDA runtime image that pip-installs torch (tests/recipes/test_job_runtime_toolchain.py).
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
    && printf '%s  %s\n' 9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625 \
        /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl | sha256sum --check --strict \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-index --no-deps \
        /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl \
    && rm /tmp/requirements.lock /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl \
    && python3 -c "from diffusers import MarigoldIntrinsicsPipeline, MarigoldNormalsPipeline, MarigoldDepthPipeline; from vonk_agent_protocol.job_inputs import RecipeJobInputManifest; import torch, PIL"

COPY run.py /opt/vonk/source/run.py
RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
