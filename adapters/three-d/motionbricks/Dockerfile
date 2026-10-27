FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/NVlabs/GR00T-WholeBodyControl" \
      org.opencontainers.image.revision="45c03facbf848e0c743091b7d9c17952096ff908" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.adapter="motionbricks-g1" \
      io.vonk.agent-protocol.wheel-sha256="9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625"

# PYTHONPATH is a platform-owned name, so run.py puts /opt/vonk/motionbricks on sys.path itself.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY requirements.lock /tmp/requirements.lock
COPY vonk_agent_protocol-4.1.0-py3-none-any.whl /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl
# build-essential and python3-dev: Triton JIT needs a compiler and Python.h in every
# CUDA runtime image that pip-installs torch (tests/test_job_runtime_toolchain.py).
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        python3-dev \
        ca-certificates \
        python3 \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /tmp/requirements.lock \
    && printf '%s  %s\n' 9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625 \
        /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl | sha256sum --check --strict \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-index --no-deps \
        /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl \
    && rm /tmp/requirements.lock /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl

# The vendored package, the G1 skeleton and the plain-git sidecars (configs, skeletons,
# statistics); the four Git LFS checkpoints are mounted by the recipe.
COPY motionbricks /opt/vonk/motionbricks
COPY LICENSE NOTICE run.py /opt/vonk/source/

# Build-time import smoke (no GPU, no weights): TorchScript decorators compile at import,
# vector-quantize-pytorch and mujoco load their native parts, the protocol model imports.
# demo/utils.py and demo/controllers.py are never imported: controllers.py starts a keyboard listener.
RUN PYTHONPATH=/opt/vonk/motionbricks python3 -c "import motionbricks.exp_setup.experiment, motionbricks.motion_backbone.inference.motion_inference, motionbricks.motion_backbone.demo.full_agent, mujoco, vector_quantize_pytorch; from vonk_agent_protocol.job_inputs import RecipeJobInputManifest" \
    && install --directory --mode=0755 /opt/vonk/bin /inputs /outputs \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk \
    && chown 10001:10001 /outputs

USER 10001:10001
ENTRYPOINT []
