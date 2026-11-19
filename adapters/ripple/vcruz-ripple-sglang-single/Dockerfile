# vcruz305's one-Spark Ripple-65B-Preview EXL3 SAGE-REXL3 + DFlash2 lane on SGLang, Vonk Forge packaging.
# https://github.com/vcruz305/Ripple-65B-Preview-EXL3-SGLang-recipe at f4de1bdde823dd5544f10e4e60cdce5a232562d0
#
# The creator publishes no image; his scripts/setup.sh pip-installs sglang[all]==0.5.19 with torch==2.13.0 (cu130), builds
# turboderp-org/exllamav3 from source for the GPU's arch (12.1a on a GB10) and puts his sglang-exl3 plugin on the path. Here the
# lmsysorg/sglang v0.5.19-cu130 arm64 image (pinned by digest; sglang 0.5.19 pins torch==2.13.0) replaces that pip install,
# exllamav3 is built for sm_121a against the image's torch, and the plugin is the creator's measured pin. His serve_ripple.py,
# ripple_format.py and chat_template.jinja (the pack's own copy is byte-identical) are vendored byte-identical. Differences from setup.sh: the exllamav3 ref is inferred (setup.sh leaves
# EXLLAMA_V3_REF empty, so it builds the default-branch head at clone time; 27dbad01 is the last master commit before the
# creator's first commit, 2026-11-07T01:00Z); the plugin is pinned to the commit his README and docs/MEASUREMENTS.md name
# instead of master; a build-time sm_121 scan of the compiled extension is added; the Controller launcher, the non-root user
# and the engine cache paths are added.
FROM docker.io/lmsysorg/sglang@sha256:4cba07b0c68725991890c64843403396effb7faa39ac47261166b2299095c513

LABEL org.opencontainers.image.source="https://github.com/vcruz305/Ripple-65B-Preview-EXL3-SGLang-recipe" \
      org.opencontainers.image.revision="f4de1bdde823dd5544f10e4e60cdce5a232562d0" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

ARG EXLLAMAV3_REPO=https://github.com/turboderp-org/exllamav3.git
ARG EXLLAMAV3_REF=27dbad01f67d2ad0dd4ee857ebe8600c5baff4e6
ARG SGLANG_EXL3_REPO=https://github.com/vcruz305/sglang-exl3.git
ARG SGLANG_EXL3_REF=5d1c9efd4490b4f55d0a1c374e7458d78c26156c
ARG MAX_JOBS=3

# 1. exllamav3 runtime, built for GB10 against the image's torch. MAX_JOBS=3 keeps the nvcc peak inside a
#    128 GB unified-memory box. The check is the one of the creator's scripts/setup.sh --check.
RUN set -eux; \
    git clone --filter=blob:none "${EXLLAMAV3_REPO}" /opt/src/exllamav3; \
    cd /opt/src/exllamav3; \
    git checkout --detach "${EXLLAMAV3_REF}"; \
    test "$(git rev-parse HEAD)" = "${EXLLAMAV3_REF}"; \
    TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=${MAX_JOBS} CUDA_HOME=/usr/local/cuda \
      python3 -m pip install --no-build-isolation --no-deps .; \
    cd /; \
    python3 -c "import exllamav3, exllamav3_ext as e; print('exllamav3 imported'); assert hasattr(e, 'exl3_gemm'), 'exl3_gemm missing from exllamav3_ext'; assert hasattr(e, 'exl3_mgemm'), 'exl3_mgemm missing from exllamav3_ext'"

# 2. sglang-exl3 plugin, pure Python on PYTHONPATH (the creator's measured runs did not build its sglang_exl3_c
#    extension). The dist-info only registers the SGLang plugin entry point.
RUN set -eux; \
    git clone --filter=blob:none "${SGLANG_EXL3_REPO}" /opt/sglang-exl3; \
    cd /opt/sglang-exl3; \
    git checkout --detach "${SGLANG_EXL3_REF}"; \
    test "$(git rev-parse HEAD)" = "${SGLANG_EXL3_REF}"; \
    rm -rf .git

# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or
# sm_121a; every file with device code needs a loadable image; each file's listing is echoed). Covered: the compiled
# exllamav3_ext this build adds. Not covered: the base image's own kernels (SGLang, torch, flashinfer), served as published.
COPY scan-sm121.py /opt/scan-sm121.py
RUN set -eux; \
    cd /tmp; \
    EXT_EXL3="$(python3 -c 'import torch, exllamav3_ext; print(exllamav3_ext.__file__)')"; \
    python3 /opt/scan-sm121.py --require "$EXT_EXL3"; \
    rm /opt/scan-sm121.py

COPY scripts/serve_ripple.py scripts/ripple_format.py /opt/vonk/scripts/
COPY scripts/chat_template.jinja /opt/vonk/scripts/chat_template.jinja
COPY LICENSE /opt/vonk/LICENSE.ripple-sglang
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache \
    && chmod -R a+rX /opt/vonk /opt/sglang-exl3 \
    && test -x /opt/vonk/bin/sglang-serve

ENV PYTHONPATH=/opt/sglang-exl3/src \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
