FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4

LABEL org.opencontainers.image.source="https://github.com/Tencent-Hunyuan/HunyuanVideo-Foley" \
      org.opencontainers.image.revision="df7b005b5023df2a9b73e1d66dd51d452799884e" \
      org.opencontainers.image.licenses="LicenseRef-Tencent-Hunyuan-Community" \
      io.vonk.transformers.revision="a957b7911a758d54597914b4479fe6e81424d64f" \
      io.vonk.audiotools.revision="348ebf2034ce24e2a91a553e3171cb00c0c71678"
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HUB_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY requirements.lock /tmp/requirements.lock
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ca-certificates \
        ffmpeg \
        git \
        libsndfile1 \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.lock \
    && rm /tmp/requirements.lock

COPY vendor/hunyuan-video-foley-df7b005b5023df2a9b73e1d66dd51d452799884e.tar.gz /tmp/hunyuan-video-foley.tar.gz
COPY vendor/transformers-a957b7911a758d54597914b4479fe6e81424d64f.tar.gz.part-000 /tmp/transformers.tar.gz.part-000
COPY vendor/transformers-a957b7911a758d54597914b4479fe6e81424d64f.tar.gz.part-001 /tmp/transformers.tar.gz.part-001
COPY vendor/transformers-a957b7911a758d54597914b4479fe6e81424d64f.tar.gz.part-002 /tmp/transformers.tar.gz.part-002
COPY vendor/audiotools-348ebf2034ce24e2a91a553e3171cb00c0c71678.tar.gz /tmp/audiotools.tar.gz
COPY patch-offline-models.py /tmp/patch-offline-models.py
RUN printf '%s\n' \
        'b4ea426256735091cb00938064bce2aed2d00fec6a5b49738500caa88bb00498  /tmp/transformers.tar.gz.part-000' \
        '6787714a5deeff0233a2502d881ff1bd743a10c5c0f728122b3a2b20c12c1d2e  /tmp/transformers.tar.gz.part-001' \
        'b38fd6701a1dec5e27aa339588e9f5359051a850b4ffe6cc2cd38f0f8a778f1a  /tmp/transformers.tar.gz.part-002' \
        | sha256sum --check --strict - \
    && cat \
        /tmp/transformers.tar.gz.part-000 \
        /tmp/transformers.tar.gz.part-001 \
        /tmp/transformers.tar.gz.part-002 \
        > /tmp/transformers.tar.gz \
    && printf '%s\n' \
        'b44048a39061ad8cdf4ec530760aa567b5563cc912dce0b42b243493b0eb21e2  /tmp/transformers.tar.gz' \
        | sha256sum --check --strict - \
    && rm /tmp/transformers.tar.gz.part-000 \
        /tmp/transformers.tar.gz.part-001 \
        /tmp/transformers.tar.gz.part-002 \
    && install --directory --mode=0755 /opt/hunyuan-video-foley \
    && install --directory --mode=0755 /opt/vendor/transformers /opt/vendor/audiotools \
    && tar --extract --gzip --file=/tmp/hunyuan-video-foley.tar.gz \
        --directory=/opt/hunyuan-video-foley --strip-components=1 \
    && tar --extract --gzip --file=/tmp/transformers.tar.gz \
        --directory=/opt/vendor/transformers --strip-components=1 \
    && tar --extract --gzip --file=/tmp/audiotools.tar.gz \
        --directory=/opt/vendor/audiotools --strip-components=1 \
    && python3 /tmp/patch-offline-models.py /opt/hunyuan-video-foley \
    && rm /tmp/hunyuan-video-foley.tar.gz /tmp/transformers.tar.gz /tmp/audiotools.tar.gz \
        /tmp/patch-offline-models.py \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --no-build-isolation --no-deps /opt/vendor/transformers /opt/vendor/audiotools

COPY pipelines /opt/vonk/source/pipelines
RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/pipelines/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
