FROM nvcr.io/nvidia/cuda:13.2.1-runtime-ubuntu24.04@sha256:a52783d8d73ace53998d4e740515e9942e73072dc7fbd5322917eb382a0bc7fb

LABEL org.opencontainers.image.source="https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni" \
      org.opencontainers.image.revision="4d47c0cc2bd0c4281963a7314ab330a5af36bfa8" \
      org.opencontainers.image.licenses="LicenseRef-Tencent-Hunyuan-3D-Omni-Community-License" \
      io.vonk.adapter="hunyuan3d-omni"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes ca-certificates curl libgl1 libglib2.0-0 python3 python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --retry 5 \
        https://github.com/Tencent-Hunyuan/Hunyuan3D-Omni/archive/4d47c0cc2bd0c4281963a7314ab330a5af36bfa8.tar.gz \
        --output /tmp/hunyuan3d-omni.tar.gz \
    && echo "1191700188114ac9fd257ed617c3a46bb523adc90a05316c2bbed433063e32d3  /tmp/hunyuan3d-omni.tar.gz" | sha256sum --check --strict \
    && mkdir -p /opt/hunyuan3d-omni \
    && tar --extract --gzip --file /tmp/hunyuan3d-omni.tar.gz --strip-components=1 --directory /opt/hunyuan3d-omni \
    && rm /tmp/hunyuan3d-omni.tar.gz \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu132 \
        torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        transformers==5.15.1 diffusers==0.40.0 accelerate==1.14.0 \
        timm==1.0.28 torchdiffeq==0.2.5 pytorch-lightning==2.6.5 \
        einops==0.8.2 omegaconf==2.3.1 pyyaml==6.0.3 \
        opencv-python-headless==5.0.0.93 pillow==12.3.0 \
        scikit-image==0.26.0 scipy==1.18.1 trimesh==5.0.0 tqdm==4.70.0 \
    && install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

COPY run.py /opt/vonk/source/run.py
COPY glb_validation.py /opt/vonk/source/glb_validation.py

RUN printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline /opt/vonk/source/*.py \
    && chown -R 10001:10001 /opt/vonk

USER 10001:10001
WORKDIR /opt/hunyuan3d-omni
ENTRYPOINT []
