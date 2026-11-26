FROM nvcr.io/nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:d44d6dc249c2c8330d0b380c595a514f0e86137fcc717fe7a4389a25720ade37

LABEL org.opencontainers.image.source="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1" \
      org.opencontainers.image.revision="82920d643c0dc2f7bfd7255f45f62d386edfe60c" \
      org.opencontainers.image.licenses="LicenseRef-Tencent-Hunyuan-3D-2.1-Community-License" \
      io.vonk.adapter="hunyuan3d-2-1" \
      io.vonk.agent-protocol.wheel-sha256="9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625"

# PYTHONPATH is a platform-owned name, so run.py gives the creator's source trees to the worker itself.
# The two native extensions (custom_rasterizer, mesh_inpaint_processor) are built here for sm_121.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST=12.1 \
    FORCE_CUDA=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache

COPY requirements.lock requirements-sdist.txt /tmp/
COPY vonk_agent_protocol-4.1.0-py3-none-any.whl /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl
# Runtime libraries the locked aarch64 wheels link but do not bundle (DT_NEEDED of every .so in them,
# minus glibc and what each wheel carries): open3d needs libgl1, libegl1, libx11-6, libgfortran5, libidn2-0;
# pymeshlab libgl1, libopengl0, libx11-6, libx11-xcb1, libxcb1, libice6, libsm6, libfontconfig1,
# libfreetype6, libharfbuzz0b, libgmp10, libgpg-error0, libcom-err2; opencv-python libgl1, libglib2.0-0,
# libsm6, libice6, libxext6, libx11-6, libxcb1. libgomp1: OpenMP for the native builds.
# --ignore-installed: the lock pins wheel, which Debian's python3-wheel owns (pip cannot uninstall it).
# python-is-python3: compile_mesh_painter.sh runs `python -m pybind11 --includes`.
# HOME and XDG_CACHE_HOME: diffusers copies the paint model's custom modules into its cache.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        ca-certificates \
        curl \
        libcom-err2 \
        libegl1 \
        libfontconfig1 \
        libfreetype6 \
        libgfortran5 \
        libgl1 \
        libglib2.0-0 \
        libgmp10 \
        libgomp1 \
        libgpg-error0 \
        libharfbuzz0b \
        libice6 \
        libidn2-0 \
        libopengl0 \
        libsm6 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxext6 \
        ninja-build \
        python-is-python3 \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --ignore-installed --require-hashes \
        --extra-index-url https://download.pytorch.org/whl/cu132 \
        --requirement /tmp/requirements.lock \
    && python3 -m pip install --break-system-packages --no-cache-dir \
        --no-deps --no-build-isolation --require-hashes \
        --requirement /tmp/requirements-sdist.txt \
    && printf '%s  %s\n' 9db7484767d0eaec9a0c20a6f18c1c995570c5588ffed8bee3da4a70022cc625 \
        /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl | sha256sum --check --strict \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-index --no-deps \
        /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl \
    && rm /tmp/requirements.lock /tmp/requirements-sdist.txt /tmp/vonk_agent_protocol-4.1.0-py3-none-any.whl

# The creator's source at the pinned commit. The weights are mounted by the recipe.
RUN curl --fail --location --retry 5 \
        https://codeload.github.com/Tencent-Hunyuan/Hunyuan3D-2.1/tar.gz/82920d643c0dc2f7bfd7255f45f62d386edfe60c \
        --output /tmp/hunyuan3d.tar.gz \
    && echo "c26df8019841e8715095e680ba72907a515c101a95ad0d4c6ba0553e8b5732b6  /tmp/hunyuan3d.tar.gz" | sha256sum --check --strict \
    && install --directory --mode=0755 /opt/hunyuan3d \
    && tar --extract --gzip --file /tmp/hunyuan3d.tar.gz --strip-components=1 --directory /opt/hunyuan3d \
    && rm /tmp/hunyuan3d.tar.gz

# prepare_upstream.py fails the build unless each patched text occurs exactly once. Then the creator's own
# native builds (custom_rasterizer: pip install --no-build-isolation; mesh_inpaint_processor:
# compile_mesh_painter.sh), each asserted to hold sm_121 kernels or to import, so a missing GB10
# build fails here and not on the hardware start.
COPY prepare_upstream.py /tmp/prepare_upstream.py
RUN python3 /tmp/prepare_upstream.py \
    && rm /tmp/prepare_upstream.py \
    && python3 -m pip install --break-system-packages --no-cache-dir --no-build-isolation \
        /opt/hunyuan3d/hy3dpaint/custom_rasterizer \
    && cuobjdump --list-elf "$(python3 -c 'import torch, custom_rasterizer_kernel as k; print(k.__file__)')" | grep --quiet sm_121 \
    && cd /opt/hunyuan3d/hy3dpaint/DifferentiableRenderer \
    && bash compile_mesh_painter.sh \
    && cd / \
    && PYTHONPATH=/opt/hunyuan3d/hy3dpaint python3 -c 'import DifferentiableRenderer.mesh_inpaint_processor'

COPY run.py worker.py export_pbr_glb.py glb_validation.py /opt/vonk/source/
COPY build_smoke.py /tmp/build_smoke.py

# Build-time import smoke (no GPU, no weights): the native modules, the creator's pipelines, the mesh
# libraries, the upscaler classes, the adapter's own modules and the protocol model.
RUN cd /opt/vonk/source \
    && PYTHONPATH=/opt/vonk/source:/opt/hunyuan3d:/opt/hunyuan3d/hy3dshape:/opt/hunyuan3d/hy3dpaint python3 /tmp/build_smoke.py \
    && rm /tmp/build_smoke.py \
    && install --directory --mode=0755 /opt/vonk/bin /inputs /outputs \
    && printf '%s\n' '#!/bin/sh' 'exec python3 /opt/vonk/source/run.py "$@"' > /opt/vonk/bin/pytorch-pipeline \
    && chmod 0755 /opt/vonk/bin/pytorch-pipeline \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk \
    && chown 10001:10001 /outputs

USER 10001:10001
ENTRYPOINT []
