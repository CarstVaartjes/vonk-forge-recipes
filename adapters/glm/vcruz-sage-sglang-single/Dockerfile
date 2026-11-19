# vcruz305's one-Spark GLM-5.3-Flash EXL3 SAGE 2.25bpw + DFlash2 lane on SGLang, Vonk Forge packaging.
# https://github.com/vcruz305/GLM-5.3-Flash-EXL3-SAGE-SGLang-Spark-recipe at 362fe245bc1ddd4de7bc50870ebfe3b0cb8ceb34
#
# The creator publishes no image, so the image is built from the creator's docker/Dockerfile: the lmsysorg/sglang:v0.5.22
# arm64 image pinned by digest (scripts/env.sh names the multi-arch index; the lock resolves its linux/arm64 manifest),
# the creator's exllamav3 fork (vcruz305/exllamav3 475461e9, CoopMK kernels) compiled for sm_121a, his sglang-exl3 plugin
# (ad9c2a06) on PYTHONPATH, his eight SGLang v0.5.22 patches (plus our 0009 for non-greedy sampling) and his fail-closed flashinfer stub. Differences from his
# Dockerfile: the build arguments are fixed here to the pins of his scripts/env.sh; a build-time sm_121 kernel scan of the
# compiled extension is added; the Controller launcher, the non-root user and the engine cache paths are added.
FROM docker.io/lmsysorg/sglang@sha256:06441be3de19a816ddbac917535e920cbfc7fc43171abc08417394f43634ffb6

LABEL org.opencontainers.image.source="https://github.com/vcruz305/GLM-5.3-Flash-EXL3-SAGE-SGLang-Spark-recipe" \
      org.opencontainers.image.revision="362fe245bc1ddd4de7bc50870ebfe3b0cb8ceb34" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

ARG EXLLAMAV3_REPO=https://github.com/vcruz305/exllamav3.git
ARG EXLLAMAV3_REF=475461e9a1a288e7c10eb472056239d3fb78b625
ARG SGLANG_EXL3_REPO=https://github.com/vcruz305/sglang-exl3.git
ARG SGLANG_EXL3_REF=ad9c2a06733bd5237c49f5bff1c644ac3652bb7f
ARG MAX_JOBS=3

# 1. exllamav3 runtime, built for GB10 against the image's torch. MAX_JOBS=3 keeps the nvcc peak inside a
#    128 GB unified-memory box.
RUN set -eux; \
    git clone --filter=blob:none "${EXLLAMAV3_REPO}" /opt/src/exllamav3; \
    cd /opt/src/exllamav3; \
    git checkout --detach "${EXLLAMAV3_REF}"; \
    test "$(git rev-parse HEAD)" = "${EXLLAMAV3_REF}"; \
    TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=${MAX_JOBS} CUDA_HOME=/usr/local/cuda \
      python3 -m pip install --no-build-isolation --no-deps .; \
    cd /; \
    python3 -c "import exllamav3, exllamav3_ext as e; print('exllamav3 imported'); assert hasattr(e, 'CoopMK'), 'CoopMK missing from exllamav3_ext'"

# 2. sglang-exl3 plugin, pure Python on PYTHONPATH (the creator's measured runs did not build its sglang_exl3_c
#    extension). The dist-info only registers the SGLang plugin entry point.
RUN set -eux; \
    git clone --filter=blob:none "${SGLANG_EXL3_REPO}" /opt/sglang-exl3; \
    cd /opt/sglang-exl3; \
    git checkout --detach "${SGLANG_EXL3_REF}"; \
    test "$(git rev-parse HEAD)" = "${SGLANG_EXL3_REF}"; \
    rm -rf .git; \
    mkdir -p /opt/sglang-exl3-dist/sglang_exl3-0.2.1.dist-info; \
    printf 'Metadata-Version: 2.1\nName: sglang-exl3\nVersion: 0.2.1\n' > /opt/sglang-exl3-dist/sglang_exl3-0.2.1.dist-info/METADATA; \
    printf '[sglang.srt.plugins]\nexl3 = sglang_exl3:register\n' > /opt/sglang-exl3-dist/sglang_exl3-0.2.1.dist-info/entry_points.txt

# 3. The creator's SGLang v0.5.22 patches for GLM-5.3-Flash EXL3 + DFlash2 on GB10 plus our 0009 for non-greedy sampling without flashinfer (docker/patches/README.md); the loop applies them in order.
COPY docker/patches/sglang-v0.5.22/ /opt/patches/sglang-v0.5.22/
RUN set -eux; \
    cd /sgl-workspace/sglang; \
    for p in /opt/patches/sglang-v0.5.22/*.patch; do git apply -p1 --verbose "$p"; done

# 4. Fail-closed flashinfer stub: SGLang imports flashinfer at module level; every call raises, and the probed
#    optional names are absent so SGLang picks its torch/triton fallbacks.
COPY docker/flashinfer-stub/ /opt/flashinfer-stub/

# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or
# sm_121a; every file with device code needs a loadable image; each file's listing is echoed). Covered: the compiled
# exllamav3_ext this build adds. Not covered: the base image's own kernels (SGLang, torch), served as published.
COPY scan-sm121.py /opt/scan-sm121.py
RUN set -eux; \
    cd /tmp; \
    EXT_EXL3="$(python3 -c 'import torch, exllamav3_ext; print(exllamav3_ext.__file__)')"; \
    python3 /opt/scan-sm121.py --require "$EXT_EXL3"; \
    rm /opt/scan-sm121.py

COPY scripts/make_view.py /opt/vonk/make_view.py
COPY LICENSE /opt/vonk/LICENSE.glm53-sage-sglang
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache \
    && chmod -R a+rX /opt/vonk /opt/sglang-exl3 /opt/sglang-exl3-dist /opt/flashinfer-stub \
    && test -x /opt/vonk/bin/sglang-serve

ENV PYTHONPATH=/opt/flashinfer-stub:/opt/sglang-exl3/src:/opt/sglang-exl3-dist \
    SGLANG_PLUGINS=exl3 \
    SGLANG_EXL3_REGISTER=1 \
    SGLANG_IS_FLASHINFER_AVAILABLE=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    EXLLAMAV3_TUNE_CACHE=/outputs/cache/exl3-tune

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
