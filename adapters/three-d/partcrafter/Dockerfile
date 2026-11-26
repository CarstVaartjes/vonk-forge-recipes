FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/wgsxm/PartCrafter" \
      org.opencontainers.image.revision="3d773bf02fad51c7ab31a5615573fec93b287b30" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.adapter="partcrafter" \
      io.vonk.agent-protocol.wheel-sha256="9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625"

# PYTHONPATH is a platform-owned name, so run.py gives the creator's source to the worker itself.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY requirements.lock /tmp/requirements.lock
COPY vonk_agent_protocol-4.1.0-py3-none-any.whl /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl
# build-essential and python3-dev: Triton JIT needs a compiler and Python.h in every
# CUDA runtime image that pip-installs torch (tests/recipes/test_job_runtime_toolchain.py).
# Nothing here compiles CUDA, so the runtime base is enough.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3-dev \
        ca-certificates \
        curl \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu132 \
        --requirement /tmp/requirements.lock \
    && printf '%s  %s\n' 9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625 \
        /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl | sha256sum --check --strict \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-index --no-deps \
        /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl \
    && rm /tmp/requirements.lock /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl

# The creator's source at the pinned commit (its src package; the training code, datasets, renders and
# README assets are not used). The weights are mounted by the recipe.
RUN curl --fail --location --retry 5 \
        https://codeload.github.com/wgsxm/PartCrafter/tar.gz/3d773bf02fad51c7ab31a5615573fec93b287b30 \
        --output /tmp/partcrafter.tar.gz \
    && echo "d7d1cf92c8d642af134f225ab447ff63b3b4784f1516d0c133c41e7cd0e2ccb6  /tmp/partcrafter.tar.gz" | sha256sum --check --strict \
    && mkdir /tmp/partcrafter-source \
    && tar --extract --gzip --file /tmp/partcrafter.tar.gz --strip-components=1 --directory /tmp/partcrafter-source \
    && install --directory --mode=0755 /opt/partcrafter \
    && cp -r /tmp/partcrafter-source/src /tmp/partcrafter-source/LICENSE /opt/partcrafter/ \
    && rm -rf /tmp/partcrafter.tar.gz /tmp/partcrafter-source

# Fails the build, not the hardware start, when the creator's file no longer has the text we patch.
COPY prepare_upstream.py /tmp/prepare_upstream.py
RUN python3 /tmp/prepare_upstream.py /opt/partcrafter && rm /tmp/prepare_upstream.py

COPY run.py worker.py glb_validation.py /opt/vonk/source/

# Build-time import smoke (no GPU, no weights): exactly what the creator's script imports (not its
# render_utils, which needs pyrender), and the protocol model.
RUN PYTHONPATH=/opt/partcrafter python3 -c "from src.utils.data_utils import get_colored_mesh_composition; from src.pipelines.pipeline_partcrafter import PartCrafterPipeline; from src.utils.image_utils import prepare_image; from src.models.briarmbg import BriaRMBG; from vonk_agent_protocol.job_inputs import RecipeJobInputManifest" \
    && install --directory --mode=0755 /opt/vonk/bin /inputs /outputs \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk \
    && chown 10001:10001 /outputs

USER 10001:10001
ENTRYPOINT []
