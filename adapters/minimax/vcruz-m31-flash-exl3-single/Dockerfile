# vcruz305's one-Spark MiniMax-M3.1-Flash EXL3 SAGE 2.15bpw lane, Vonk Forge packaging.
# https://github.com/vcruz305/MiniMax-M3.1-Flash-EXL3-Spark-recipe at d7445a4317c5894828aaa34a702bdc84200cc456
#
# The creator publishes no image, so the image is built from the creator's Dockerfile: the vLLM project's published
# arm64 image vllm/vllm-openai:minimax-m31-flash pinned by digest, the creator's exllamav3 fork
# (vcruz305/exllamav3 feat/moe-act3-batched fdd77361) and his vllm-exl3 plugin (feat/minimax-q4-prefill 41fdb7d9), each
# cloned at its pinned commit and compiled for sm_121, as his scripts/build_image.sh does from local checkouts of the
# same commits. Differences from his Dockerfile: the two trees are cloned here instead of staged from checkouts; the
# build ends with a build-time sm_121 kernel scan of both compiled extensions; the Controller launcher, the non-root
# user and the engine cache paths are added.
FROM docker.io/vllm/vllm-openai@sha256:5d76347628973fb80519a919a2f2740b6645b62cba3fe2bd87b25f111bd0e948

LABEL org.opencontainers.image.source="https://github.com/vcruz305/MiniMax-M3.1-Flash-EXL3-Spark-recipe" \
      org.opencontainers.image.revision="d7445a4317c5894828aaa34a702bdc84200cc456" \
      org.opencontainers.image.licenses="MIT AND AGPL-3.0-only" \
      ai.vonkforge.runtime-interface="v1"

ARG EXLLAMAV3_REPO=https://github.com/vcruz305/exllamav3.git
ARG EXLLAMAV3_REV=fdd77361971144e8d175e09e7b50b7100739fd95
ARG PLUGIN_REPO=https://github.com/vcruz305/vllm-exl3.git
ARG PLUGIN_REV=41fdb7d99a441d37938c2a1bd600ac963c18a3cb

# sm_121 (GB10). MAX_JOBS is capped for the GB10's unified memory: a larger fan-out OOMs the host during nvcc.
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN set -eux; \
    if ! command -v git >/dev/null 2>&1; then \
        apt-get update; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    git clone "$EXLLAMAV3_REPO" /opt/exllamav3-local; \
    git -C /opt/exllamav3-local checkout "$EXLLAMAV3_REV"; \
    test "$(git -C /opt/exllamav3-local rev-parse HEAD)" = "$EXLLAMAV3_REV"; \
    git clone "$PLUGIN_REPO" /opt/vllm-exl3-local; \
    git -C /opt/vllm-exl3-local checkout "$PLUGIN_REV"; \
    test "$(git -C /opt/vllm-exl3-local rev-parse HEAD)" = "$PLUGIN_REV"; \
    PATCHER=/opt/vllm-exl3-local/tools/patch_exllamav3_aarch64.py; \
    if [ -f "$PATCHER" ]; then \
        sed -i 's|raise SystemExit(f"expected one x86 pause block in {path}")|print(f"already guarded (x86 pause): {path}")|' "$PATCHER"; \
        grep -q 'already guarded (x86 pause)' "$PATCHER"; \
        python3 "$PATCHER" /opt/exllamav3-local/exllamav3/exllamav3_ext; \
    fi; \
    CUSPARSE_DIR="$(find /usr/local/lib/python3*/dist-packages/nvidia -name cusparse.h -exec dirname {} \; 2>/dev/null | head -1)"; \
    if [ -z "$CUSPARSE_DIR" ]; then echo "ERROR: cusparse.h not found under NVIDIA Python packages" >&2; exit 2; fi; \
    export CPATH="$CUSPARSE_DIR${CPATH:+:$CPATH}"; \
    cd /opt/exllamav3-local; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .; \
    export EXL3_EXT_INCLUDE=/opt/exllamav3-local/exllamav3/exllamav3_ext; \
    cd /opt/vllm-exl3-local; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or
# sm_121a; every file with device code needs a loadable image; each file's listing is echoed). Covered: both compiled
# extensions this build adds, exllamav3_ext (exllamav3 fork) and vllm_exl3_c (plugin), each required to carry a
# loadable image. Not covered: the base image's own kernels (vLLM, FlashInfer), which the creator serves as published.
COPY scan-sm121.py /opt/scan-sm121.py
RUN set -eux; \
    cd /tmp; \
    EXT_EXL3="$(python3 -c 'import torch, exllamav3_ext; print(exllamav3_ext.__file__)')"; \
    EXT_PLUGIN="$(python3 -c 'import torch, vllm_exl3_c; print(vllm_exl3_c.__file__)')"; \
    python3 /opt/scan-sm121.py --require "$EXT_EXL3" --require "$EXT_PLUGIN"; \
    rm /opt/scan-sm121.py

COPY --chmod=0755 vllm-launcher.sh /opt/vonk/bin/vllm

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache

ENV VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache/xdg \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    CUDA_CACHE_PATH=/outputs/cache/cuda

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
