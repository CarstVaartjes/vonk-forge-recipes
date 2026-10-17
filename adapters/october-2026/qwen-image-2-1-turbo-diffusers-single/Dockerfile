FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo" \
      org.opencontainers.image.revision="fc13ba1986fcb905854dc8de2f52eee395f49b5b" \
      org.opencontainers.image.licenses="Qwen-Research" \
      io.vonk.diffusers.revision="b800f828f3951c4d206e154411bcbfa693cd322e"

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
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.14.0 \
        torchvision==0.29.0 \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        "diffusers @ git+https://github.com/huggingface/diffusers.git@b800f828f3951c4d206e154411bcbfa693cd322e" \
        accelerate==1.15.0 \
        Pillow==12.3.0 \
        safetensors==0.8.0 \
        sentencepiece==0.2.2 \
        transformers==5.17.0

COPY vonk_agent_protocol-4.1.0-py3-none-any.whl /opt/vonk/vonk_agent_protocol-4.1.0-py3-none-any.whl
RUN python3 -m pip install --break-system-packages --no-cache-dir /opt/vonk/vonk_agent_protocol-4.1.0-py3-none-any.whl \
    && install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs
COPY job.py /opt/vonk/job.py
COPY --chmod=0755 launcher /opt/vonk/bin/diffusers-job
ENV HOME=/state HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
