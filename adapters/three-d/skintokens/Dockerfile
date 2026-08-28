FROM nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1392f431f89f143d0d6d0fa397a2b9a6a236f8b3628cfd3afbf21e15ab4a98

LABEL org.opencontainers.image.source="https://github.com/VAST-AI-Research/SkinTokens" \
      org.opencontainers.image.revision="273b691d35989d71cd17ff2895fdc735097b92d1" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.adapter="skintokens"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=12.1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential ca-certificates curl ninja-build python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --retry 5 \
        https://github.com/VAST-AI-Research/SkinTokens/archive/273b691d35989d71cd17ff2895fdc735097b92d1.tar.gz \
        --output /tmp/skintokens.tar.gz \
    && echo "f886ce830f8f6ed5a3eabebb9399244812ac17a44d5b51fe8853c381a214e334  /tmp/skintokens.tar.gz" | sha256sum --check --strict \
    && mkdir -p /opt/skintokens/models/Qwen3-0.6B \
    && tar --extract --gzip --file /tmp/skintokens.tar.gz --strip-components=1 --directory /opt/skintokens \
    && rm /tmp/skintokens.tar.gz \
    && curl --fail --location --retry 5 \
        https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/config.json \
        --output /opt/skintokens/models/Qwen3-0.6B/config.json \
    && echo "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd  /opt/skintokens/models/Qwen3-0.6B/config.json" | sha256sum --check --strict \
    && ln -s /models/target/experiments /opt/skintokens/experiments \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu132 \
        torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        transformers==5.15.1 diffusers==0.40.0 einops==0.8.2 \
        omegaconf==2.3.1 lightning==2.6.5 scipy==1.18.1 \
        trimesh==5.0.0 pygltflib==1.16.5 requests==2.32.5 \
        pillow==12.3.0 \
        flash-attn==2.8.3.post1 --no-build-isolation \
    && install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

COPY run.py /opt/vonk/source/run.py
COPY glb_validation.py /opt/vonk/source/glb_validation.py

RUN printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline /opt/vonk/source/run.py \
    && chown -R 10001:10001 /opt/vonk

USER 10001:10001
WORKDIR /opt/skintokens
ENTRYPOINT []
