# MiniMax-M3.1-Flash EXL3 (SAGE 2.15 bpw) on two DGX Sparks, TP2: tonyd2wild's kit at 70c4e433.
# The kit serves vcruz305's one-Spark image (MiniMax-M3.1-Flash-EXL3-Spark-recipe d7445a43 Dockerfile) with two
# plugin patches (patches/tp2-minimax-indexer.patch, then patches/tp2-minimax-lmhead.patch) applied to the vllm-exl3
# checkout first. No prebuilt image of either exists, so this builds the same thing from the same pins: the vLLM
# image vllm/vllm-openai:minimax-m31-flash by digest, vcruz305/exllamav3 feat/moe-act3-batched at fdd77361, and
# vcruz305/vllm-exl3 feat/minimax-q4-prefill at 41fdb7d9 plus the kit's two patches, each compiled for sm_121.
# Where vcruz stages local git archives, this clones the same commits. A build-time scan then requires every
# compiled extension to carry code sm_121 can load.
FROM docker.io/vllm/vllm-openai@sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/MiniMax-M3.1-Flash-EXL3-TP2-2x-DGX-Spark" \
      org.opencontainers.image.revision="70c4e4338c0c41fcf3d8c7a905c2155b2e4cd19f" \
      org.opencontainers.image.licenses="MIT AND AGPL-3.0-only" \
      ai.vonkforge.runtime-interface="v1"

ARG EXL3_REPO=https://github.com/vcruz305/exllamav3.git
ARG EXL3_REV=fdd77361971144e8d175e09e7b50b7100739fd95
ARG PLUGIN_REPO=https://github.com/vcruz305/vllm-exl3.git
ARG PLUGIN_REV=41fdb7d99a441d37938c2a1bd600ac963c18a3cb

# sm_121 (GB10). MAX_JOBS is capped for the GB10's unified memory (vcruz's Dockerfile).
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY patches /opt/m31-patches
RUN set -eux; \
    if ! command -v git >/dev/null 2>&1; then \
        apt-get update; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    git clone "$EXL3_REPO" /opt/exllamav3-local; \
    git -C /opt/exllamav3-local checkout "$EXL3_REV"; \
    test "$(git -C /opt/exllamav3-local rev-parse HEAD)" = "$EXL3_REV"; \
    git clone "$PLUGIN_REPO" /opt/vllm-exl3-local; \
    git -C /opt/vllm-exl3-local checkout "$PLUGIN_REV"; \
    test "$(git -C /opt/vllm-exl3-local rev-parse HEAD)" = "$PLUGIN_REV"; \
    git -C /opt/vllm-exl3-local apply /opt/m31-patches/tp2-minimax-indexer.patch; \
    git -C /opt/vllm-exl3-local apply /opt/m31-patches/tp2-minimax-lmhead.patch; \
    PATCHER=/opt/vllm-exl3-local/tools/patch_exllamav3_aarch64.py; \
    sed -i 's|raise SystemExit(f"expected one x86 pause block in {path}")|print(f"already guarded (x86 pause): {path}")|' "$PATCHER"; \
    grep -q 'already guarded (x86 pause)' "$PATCHER"; \
    python3 "$PATCHER" /opt/exllamav3-local/exllamav3/exllamav3_ext; \
    CUSPARSE_DIR="$(find /usr/local/lib/python3*/dist-packages/nvidia -name cusparse.h -exec dirname {} \; 2>/dev/null | head -1)"; \
    if [ -z "$CUSPARSE_DIR" ]; then echo "ERROR: cusparse.h not found under NVIDIA Python packages" >&2; exit 2; fi; \
    export CPATH="$CUSPARSE_DIR${CPATH:+:$CPATH}"; \
    cd /opt/exllamav3-local; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .; \
    export EXL3_EXT_INCLUDE=/opt/exllamav3-local/exllamav3/exllamav3_ext; \
    cd /opt/vllm-exl3-local; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .; \
    python3 -c "import torch, exllamav3_ext, vllm_exl3; print('exl3 extensions import')"; \
    rm -rf /opt/exllamav3-local /opt/vllm-exl3-local /opt/m31-patches /root/.cache/pip

# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or
# sm_121a; every file with device code needs a loadable image). Covered: the exllamav3 CUDA extension and the
# vllm-exl3 compiled module installed above, both required to carry device code.
COPY scan-sm121.py /opt/scan-sm121.py
RUN python3 /opt/scan-sm121.py \
      $(python3 -c "import exllamav3_ext; print('--require', exllamav3_ext.__file__)") \
      $(python3 -c "import vllm_exl3_c; print('--require', vllm_exl3_c.__file__)") \
 && rm /opt/scan-sm121.py

COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN { test -x /usr/local/bin/vllm || test -x /opt/vllm/.venv/bin/vllm; } \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --mode=0755 /inputs \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton /outputs/cache/nv-compute /outputs/cache/vllm

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    CUDA_CACHE_PATH=/outputs/cache/nv-compute

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
