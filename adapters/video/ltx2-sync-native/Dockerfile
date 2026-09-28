FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/Lightricks/LTX-2" \
      org.opencontainers.image.revision="598ab41247a77dbfe29b5186e915bcf4f9040ec7" \
      org.opencontainers.image.licenses="LTX-2 Community License Agreement" \
      io.vonk.ltx2.source-archive-sha256="946465dfba0094f97014a18eb53850214fb7445c88870889f4ea05fb20de56aa"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        curl \
        ffmpeg \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu132 \
        torch==2.13.0 torchvision==0.28.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/test/cu132 \
        torchaudio==2.11.0

COPY requirements.lock /opt/vonk/source/requirements.lock

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --requirement /opt/vonk/source/requirements.lock \
    && curl --fail --location --silent --show-error \
        --output /tmp/ltx2-source.tar.gz \
        https://github.com/Lightricks/LTX-2/archive/598ab41247a77dbfe29b5186e915bcf4f9040ec7.tar.gz \
    && printf '%s  %s\n' \
        946465dfba0094f97014a18eb53850214fb7445c88870889f4ea05fb20de56aa \
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

COPY vonk_agent_protocol-3.0.0-py3-none-any.whl /opt/vonk/source/vonk_agent_protocol-3.0.0-py3-none-any.whl
RUN python3 -m pip install --break-system-packages --no-cache-dir --no-deps \
        /opt/vonk/source/vonk_agent_protocol-3.0.0-py3-none-any.whl

COPY run.py /opt/vonk/source/run.py

RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' \
        > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
