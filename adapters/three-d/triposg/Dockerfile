FROM nvcr.io/nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1392f431f89f143d0d6d0fa397a2b9a6a236f8b3628cfd3afbf21e15ab4a98

LABEL org.opencontainers.image.source="https://github.com/VAST-AI-Research/TripoSG" \
      org.opencontainers.image.revision="fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.adapter="triposg"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=12.1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential ca-certificates curl ninja-build python3 python3-dev python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --retry 5 \
        https://github.com/VAST-AI-Research/TripoSG/archive/fc5c40990181e2a756c4e0b1c2f4d6b5202faf8c.tar.gz \
        --output /tmp/triposg.tar.gz \
    && echo "3d06f11eb795bcabea7863e670e9cea02f96bfc6ec3e6db20e015e5710653682  /tmp/triposg.tar.gz" | sha256sum --check --strict \
    && mkdir -p /opt/triposg \
    && tar --extract --gzip --file /tmp/triposg.tar.gz --strip-components=1 --directory /opt/triposg \
    && rm /tmp/triposg.tar.gz \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu132 \
        torch==2.13.0 torchvision==0.28.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        diffusers==0.40.0 transformers==5.15.1 einops==0.8.2 \
        huggingface-hub==1.28.0 opencv-python-headless==5.0.0.93 \
        trimesh==5.0.0 omegaconf==2.3.1 scikit-image==0.26.0 \
        peft==0.20.0 jaxtyping==0.3.11 typeguard==4.6.0 \
        pillow==12.3.0 safetensors==0.8.0 accelerate==1.14.0 \
        diso==0.1.4 \
    && install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

COPY run.py /opt/vonk/source/run.py
COPY glb_validation.py /opt/vonk/source/glb_validation.py

RUN printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline /opt/vonk/source/run.py \
    && chown -R 10001:10001 /opt/vonk

USER 10001:10001
WORKDIR /opt/triposg
ENTRYPOINT []
