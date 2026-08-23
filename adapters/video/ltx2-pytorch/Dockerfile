FROM nvcr.io/nvidia/cuda:13.2.1-runtime-ubuntu24.04@sha256:a52783d8d73ace53998d4e740515e9942e73072dc7fbd5322917eb382a0bc7fb

LABEL org.opencontainers.image.source="https://huggingface.co/Lightricks/LTX-2" \
      org.opencontainers.image.revision="dfcc2108383fe1aaa0584bdf55d368a4bdadd90c" \
      org.opencontainers.image.licenses="LTX-2 Community License Agreement" \
      io.vonk.ltx2.source-commit="400fd31054597515f47125691032c04b1c3ee24e"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        git \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu132 \
        torch==2.13.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/test/cu132 \
        torchaudio==2.11.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        "ltx-core @ git+https://github.com/Lightricks/LTX-2.git@400fd31054597515f47125691032c04b1c3ee24e#subdirectory=packages/ltx-core" \
        "ltx-pipelines @ git+https://github.com/Lightricks/LTX-2.git@400fd31054597515f47125691032c04b1c3ee24e#subdirectory=packages/ltx-pipelines"

COPY pipelines/run.py /opt/vonk/source/pipelines/run.py

RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source/pipelines \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/pipelines/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
