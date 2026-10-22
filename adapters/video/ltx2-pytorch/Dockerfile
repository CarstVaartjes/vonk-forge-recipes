FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6

LABEL org.opencontainers.image.source="https://huggingface.co/Lightricks/LTX-2" \
      org.opencontainers.image.revision="dfcc2108383fe1aaa0584bdf55d368a4bdadd90c" \
      org.opencontainers.image.licenses="LTX-2 Community License Agreement" \
      io.vonk.ltx2.source-commit="9ec55f9f22798a3198d9c923856824821bc3317e" \
      io.vonk.agent-protocol.source-commit="cee017ae806f0ac4c0842a3652f8ef7ee57388b8" \
      io.vonk.agent-protocol.wheel-sha256="9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625" \
      io.vonk.ltx2.source-archive-sha256="4010dea65ad0408770b08714834f35eadf32ddb4ce340365d15fc016585b4b43"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3-dev \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        patch \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0 torchvision==0.26.0 \
        torchaudio==2.11.0

# PyTorch's cu130 headers require the matching nvcc/CCCL toolkit.
RUN python3 -c "import re, subprocess, torch; toolkit = re.search(r'release ([0-9]+[.][0-9]+),', subprocess.check_output(['nvcc', '--version'], text=True)).group(1); assert toolkit == torch.version.cuda, f'CUDA compiler/toolkit {toolkit} differs from torch {torch.version.cuda}'"

COPY nvfp4-sm121.patch /opt/vonk/source/nvfp4-sm121.patch
COPY requirements.lock /opt/vonk/source/requirements.lock
COPY wheels/vonk_agent_protocol-4.1.0-py3-none-any.whl /opt/vonk/source/wheels/vonk_agent_protocol-4.1.0-py3-none-any.whl

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --requirement /opt/vonk/source/requirements.lock \
    && printf '%s  %s\n' \
        9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625 \
        /opt/vonk/source/wheels/vonk_agent_protocol-4.1.0-py3-none-any.whl \
        | sha256sum --check --strict \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-index \
        /opt/vonk/source/wheels/vonk_agent_protocol-4.1.0-py3-none-any.whl \
    && curl --fail --location --silent --show-error \
        --output /tmp/ltx2-source.tar.gz \
        https://codeload.github.com/Lightricks/LTX-2/tar.gz/9ec55f9f22798a3198d9c923856824821bc3317e \
    && printf '%s  %s\n' \
        4010dea65ad0408770b08714834f35eadf32ddb4ce340365d15fc016585b4b43 \
        /tmp/ltx2-source.tar.gz \
        | sha256sum --check --strict \
    && mkdir -p /opt/vonk/source/ltx2 \
    && tar --extract --gzip --file /tmp/ltx2-source.tar.gz \
        --directory /opt/vonk/source/ltx2 --strip-components=1 \
    && rm /tmp/ltx2-source.tar.gz \
    && patch --directory=/opt/vonk/source/ltx2 --strip=1 --fuzz=0 \
        < /opt/vonk/source/nvfp4-sm121.patch \
    && TORCH_CUDA_ARCH_LIST="12.0 12.1" MAX_JOBS=2 \
        python3 -m pip install --break-system-packages --no-cache-dir \
        --no-build-isolation --no-deps /opt/vonk/source/ltx2/packages/ltx-kernels \
    && python3 -c "import torch; import nvfp4_cpp" \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --no-build-isolation --no-deps /opt/vonk/source/ltx2/packages/ltx-core \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --no-build-isolation --no-deps /opt/vonk/source/ltx2/packages/ltx-pipelines

RUN cuobjdump --list-elf "$(python3 -c 'import torch; import nvfp4_cpp; print(nvfp4_cpp.__file__)')" \
    | grep --quiet sm_121a

COPY pipelines/run.py /opt/vonk/source/pipelines/run.py

RUN install --directory --mode=0755 /opt/vonk/bin /opt/vonk/source/pipelines \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/pipelines/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

# Import the real pipeline before publishing an image; lazy transformers imports
# otherwise hide binary torch/torchaudio incompatibilities until the first job.
RUN python3 -c "import torchaudio; from transformers import PreTrainedModel; from ltx_pipelines.utils.types import PipelineOutput"

USER 10001:10001
ENTRYPOINT []
