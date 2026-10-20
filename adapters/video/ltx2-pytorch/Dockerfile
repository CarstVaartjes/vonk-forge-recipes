FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://huggingface.co/Lightricks/LTX-2" \
      org.opencontainers.image.revision="dfcc2108383fe1aaa0584bdf55d368a4bdadd90c" \
      org.opencontainers.image.licenses="LTX-2 Community License Agreement" \
      io.vonk.ltx2.source-commit="9ec55f9f22798a3198d9c923856824821bc3317e" \
      io.vonk.agent-protocol.source-commit="e6de69b20c098d4df2eb9e0294cede8acc552a3c" \
      io.vonk.agent-protocol.wheel-sha256="e80a989df7d99ba0e857738ee1daea901865b9fb2f77aca5d481a287e45b7195" \
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
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0 torchvision==0.26.0 \
        torchaudio==2.11.0

COPY requirements.lock /opt/vonk/source/requirements.lock
COPY wheels/vonk_agent_protocol-4.1.0-py3-none-any.whl /opt/vonk/source/wheels/vonk_agent_protocol-4.1.0-py3-none-any.whl

RUN python3 -m pip install --break-system-packages --no-cache-dir \
        --requirement /opt/vonk/source/requirements.lock \
    && printf '%s  %s\n' \
        e80a989df7d99ba0e857738ee1daea901865b9fb2f77aca5d481a287e45b7195 \
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
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --no-build-isolation --no-deps /opt/vonk/source/ltx2/packages/ltx-core \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --no-build-isolation --no-deps /opt/vonk/source/ltx2/packages/ltx-pipelines

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
