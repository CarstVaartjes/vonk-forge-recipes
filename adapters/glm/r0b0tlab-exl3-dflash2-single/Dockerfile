# r0b0tlab GLM-5.3-Flash EXL3 + DFlash2 on one DGX Spark (GB10 / SM121).
#
# Upstream ships no container image; it measured a host-native vLLM
# v0.30.0rc1 (a00a3544b93e) source build. This image applies the same anchored
# patches (0007..0012; 0006 is a subset of 0009 and is skipped, as upstream
# documents) to the digest-pinned official vLLM v0.30.0 ARM64 image
# (ced6857afa0e), installs the vllm_exl3_sm121 plugin and builds its CUDA
# extension for sm_120/sm_121. Every hunk must match exactly (apply_patches.py)
# or the build fails.
FROM docker.io/vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/glm53-flash-exl3-dflash2-sm121" \
      org.opencontainers.image.revision="2dfbdbe4840682ea1892bb1300e10d11d2ff4758" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      io.vonk.vllm.build-commit="ced6857afa0ea7b2e3f0846a62e1394e90f15607"

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="12.0;12.1" \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

# vLLM patches, applied to the installed package tree.
COPY apply_patches.py /tmp/exl3/apply_patches.py
COPY patches /tmp/exl3/patches
RUN set -eux; \
    site="$(python3 -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)')"; \
    python3 /tmp/exl3/apply_patches.py "$site" /tmp/exl3/patches/0007-*.patch /tmp/exl3/patches/0008-*.patch \
      /tmp/exl3/patches/0009-*.patch /tmp/exl3/patches/0010-*.patch /tmp/exl3/patches/0011-*.patch /tmp/exl3/patches/0012-*.patch

# Plugin + CUDA extension. The vLLM image keeps the CUDA 13 headers under
# nvidia/cu13 rather than /usr/local/cuda/include. setuptools 80.10.2 (the
# version upstream verified; the plugin needs >=77 for PEP 639 metadata) and
# wheel are installed to a throwaway directory used only for the build.
WORKDIR /opt/vllm-exl3-sm121
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md runtime.lock.json provenance.md ./
COPY src ./src
COPY csrc ./csrc
COPY vendored ./vendored
RUN set -eux; \
    pip install --no-cache-dir --target /tmp/exl3/buildtools "setuptools==80.10.2" "wheel==0.45.1"; \
    export PYTHONPATH=/tmp/exl3/buildtools MAX_JOBS=8; \
    inc="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include"; \
    export CPATH="$inc${CPATH:+:$CPATH}" CPLUS_INCLUDE_PATH="$inc${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}" C_INCLUDE_PATH="$inc${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"; \
    pip install --no-deps --no-build-isolation --no-cache-dir .; \
    python3 csrc/setup_ext.py bdist_wheel --dist-dir /tmp/exl3/wheel; \
    pip install --no-deps --no-cache-dir /tmp/exl3/wheel/*.whl; \
    python3 -c "import torch, vllm_exl3_sm121_ext as e; missing=[n for n in ('exl3_gemv','exl3_gemm','exl3_moe','exl3_moe_gather','exl3_moe_max_concurrency','exl3_reconstruct') if not hasattr(e,n)]; assert not missing, missing; print('vllm_exl3_sm121_ext', e.__file__)"; \
    python3 -c "from importlib.metadata import entry_points; assert any(e.name=='vllm_exl3_sm121' for e in entry_points(group='vllm.general_plugins'))"; \
    rm -rf /tmp/exl3 build /root/.cache

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --mode=0755 /opt/vonk/bin \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache \
    && printf '%s\n' '#!/bin/sh' 'set -eu' 'mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$UV_CACHE_DIR"' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

ENV HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache/xdg \
    XDG_CONFIG_HOME=/outputs/cache/config \
    HF_HOME=/outputs/cache/huggingface \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_HOME=/outputs/cache/torch \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    UV_CACHE_DIR=/outputs/cache/uv

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
