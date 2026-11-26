FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/hkchengrex/MMAudio" \
      org.opencontainers.image.revision="974010a026c731054592d8f777218bd9d85a6c24" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.adapter="mmaudio-pytorch" \
      io.vonk.agent-protocol.wheel-sha256="9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625"

# PYTHONPATH is a platform-owned name, so run.py gives the creator's source tree to the worker itself.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache

COPY requirements.lock /tmp/requirements.lock
COPY vonk_agent_protocol-4.1.0-py3-none-any.whl /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl
# build-essential and python3-dev: Triton JIT needs a compiler and Python.h in every
# CUDA runtime image that pip-installs torch (tests/recipes/test_job_runtime_toolchain.py);
# libsndfile1: soundfile (librosa) loads it.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3-dev \
        ca-certificates \
        curl \
        ffmpeg \
        libsndfile1 \
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
    && rm /tmp/requirements.lock /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl

# The creator's mmaudio package at the pinned commit (not its pyproject, which pulls gradio, hydra and
# the training stack). The weights are mounted by the recipe.
RUN curl --fail --location --retry 5 \
        https://codeload.github.com/hkchengrex/MMAudio/tar.gz/974010a026c731054592d8f777218bd9d85a6c24 \
        --output /tmp/mmaudio.tar.gz \
    && echo "b8ddbd5226a7aa1aa86030e86881b73c8ae8788469b62a923aebe9ce242e8d1d  /tmp/mmaudio.tar.gz" | sha256sum --check --strict \
    && mkdir /tmp/mmaudio-source \
    && tar --extract --gzip --file /tmp/mmaudio.tar.gz --strip-components=1 --directory /tmp/mmaudio-source \
    && install --directory --mode=0755 /opt/mmaudio \
    && cp -r /tmp/mmaudio-source/mmaudio /tmp/mmaudio-source/LICENSE /opt/mmaudio/ \
    && rm -rf /tmp/mmaudio.tar.gz /tmp/mmaudio-source

# prepare_upstream.py fails the build unless each patched text occurs exactly once.
COPY prepare_upstream.py /tmp/prepare_upstream.py
RUN python3 /tmp/prepare_upstream.py \
    && rm /tmp/prepare_upstream.py

COPY run.py worker.py /opt/vonk/source/
COPY build_smoke.py /tmp/build_smoke.py

# Build-time import smoke (no GPU, no weights): the creator's modules the worker uses, PyAV, open_clip,
# the adapter's own modules and the protocol model.
RUN cd /opt/vonk/source \
    && PYTHONPATH=/opt/vonk/source:/opt/mmaudio python3 /tmp/build_smoke.py \
    && rm /tmp/build_smoke.py \
    && install --directory --mode=0755 /opt/vonk/bin /inputs /outputs \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk \
    && chown 10001:10001 /outputs

USER 10001:10001
ENTRYPOINT []
