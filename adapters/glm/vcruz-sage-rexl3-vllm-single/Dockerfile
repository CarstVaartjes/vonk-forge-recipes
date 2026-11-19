# vcruz305's one-Spark GLM-5.3-Flash SAGE-REXL3 2.48bpw + DFlash2 lane on vLLM, Vonk Forge packaging.
# https://github.com/vcruz305/GLM-5.3-Flash-SAGE-REXL3-DGX-Spark-recipe at f1d5dfa1e7bcd8465bf635b618cea220002a82d8
#
# The creator publishes no image and does not say how his local image m31-flash-exl3-spark:dev-fdd7736 was built. It is
# the image of his sibling MiniMax lane (https://github.com/vcruz305/MiniMax-M3.1-Flash-EXL3-Spark-recipe at
# d7445a4317c5894828aaa34a702bdc84200cc456, whose scripts/env.sh names the "GB10-measured image dev-fdd7736-a5e91ed"): the
# vLLM project's published arm64 image vllm/vllm-openai:minimax-m31-flash (main snapshot 2026-10-22) pinned by digest, his
# exllamav3 fork (vcruz305/exllamav3 fdd77361) and his vllm-exl3 plugin at a5e91ed5, the older plugin that image bakes.
# That plugin revision is inferred from the image tag and confirmed against the creator's own boot patch: every anchor
# of its exl3.py edits matches a5e91ed5 and none of them matches the newer 41fdb7d9. Both trees are cloned here at their
# pinned commits and compiled for sm_121, as the MiniMax adapter does.
# The creator's in-container patches (patches/) are applied at build time, because the platform's root file system is
# read-only; his launcher applied them at every boot. They are applied active, as the proven spec server runs them
# (GV_DUMMY_SHRINK, GV_DRAFTER_SLABS, GV_INDEXER_WS). His gv_boot_patch.py ends in a self-test of the overlay config; the
# build runs the part before it, and the launcher runs the self-test against the overlay it builds (make_overlay.py: symlinks to the shards, config.json, a split-tensor safetensors file and its index).
FROM docker.io/vllm/vllm-openai@sha256:5d76347628973fb80519a919a2f2740b6645b62cba3fe2bd87b25f111bd0e948

LABEL org.opencontainers.image.source="https://github.com/vcruz305/GLM-5.3-Flash-SAGE-REXL3-DGX-Spark-recipe" \
      org.opencontainers.image.revision="f1d5dfa1e7bcd8465bf635b618cea220002a82d8" \
      org.opencontainers.image.licenses="AGPL-3.0-only" \
      ai.vonkforge.runtime-interface="v1"

ARG EXLLAMAV3_REPO=https://github.com/vcruz305/exllamav3.git
ARG EXLLAMAV3_REV=fdd77361971144e8d175e09e7b50b7100739fd95
ARG PLUGIN_REPO=https://github.com/vcruz305/vllm-exl3.git
ARG PLUGIN_REV=a5e91ed5a13880efff276ef5e96c9303f17bdc90

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

# The creator's in-container patches (his launcher mounted them at /recipe-py and applied them at every boot).
COPY patches/ /recipe-py/
RUN set -eux; \
    cd /recipe-py; \
    sed '/^# --- self-test/,$d' gv_boot_patch.py > /tmp/gv_boot_patch_build.py; \
    sed -n '/^# --- self-test/,$p' gv_boot_patch.py | sed 's|^MODEL = .*|import os; MODEL = os.environ["VLLM_EXL3_MODEL_DIR"]|' > /opt/vonk-selftest.py; \
    grep -q 'GV_BF16_LOOKUP_SELFTEST_OK' /opt/vonk-selftest.py; \
    GV_DUMMY_SHRINK=1 GV_DRAFTER_SLABS=1 GV_INDEXER_WS=1 python3 /tmp/gv_boot_patch_build.py; \
    GV_DUMMY_SHRINK=1 GV_DRAFTER_SLABS=1 GV_INDEXER_WS=1 python3 /recipe-py/gv_dummy_shrink_patch.py; \
    GV_DUMMY_SHRINK=1 GV_DRAFTER_SLABS=1 GV_INDEXER_WS=1 python3 /recipe-py/gv_drafter_slabs.py; \
    GV_DUMMY_SHRINK=1 GV_DRAFTER_SLABS=1 GV_INDEXER_WS=1 python3 /recipe-py/gv_indexer_ws.py; \
    rm /tmp/gv_boot_patch_build.py; \
    cp /recipe-py/zz_recipe_hooks.pth "$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/zz_recipe_hooks.pth"; \
    chmod -R a+rX /recipe-py /opt/vonk-selftest.py

COPY --chmod=0755 vllm-launcher.sh /opt/vonk/bin/vllm
COPY make_overlay.py /opt/vonk/make_overlay.py
COPY LICENSE /opt/vonk/LICENSE.glm53-sage-rexl3
COPY THIRD_PARTY_NOTICES.md /opt/vonk/THIRD_PARTY_NOTICES.glm53-sage-rexl3.md

# /mem is where the creator's launcher mounted a scratch directory; gv_drafter_slabs.py appends its KV-planner dump there.
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache \
    && ln -s /outputs/cache/mem /mem \
    && chmod -R a+rX /opt/vonk

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
