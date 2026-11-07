FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/amarjeet/dgx-spark-recipes" \
      org.opencontainers.image.revision="87ea80d6eb1de0eed1690e101da8161c9c8462de" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.lock /tmp/requirements.lock
# build-essential and python3-dev: flash-linear-attention's Triton kernels compile at first use.
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
    && rm /tmp/requirements.lock \
    && python3 -c "import accelerate, fastapi, PIL, safetensors, torch, transformers, uvicorn; import importlib.util as u; assert u.find_spec('fla')"

COPY app.py fp8.py decision_response.py /opt/vonk/source/
RUN python3 -m py_compile /opt/vonk/source/app.py /opt/vonk/source/fp8.py /opt/vonk/source/decision_response.py \
    && install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/app.py "$@"' > /opt/vonk/bin/clef-flash \
    && chmod 0755 /opt/vonk/bin/clef-flash \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
