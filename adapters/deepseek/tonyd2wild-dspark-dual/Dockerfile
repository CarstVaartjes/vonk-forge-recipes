FROM ghcr.io/bjk110/vllm-spark@sha256:d8492e7677cf1b9aaa3344e0e6865efc468454013eee5ebabac85be90af027be

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark" \
      org.opencontainers.image.revision="43201d1c2f281d238b8fd65f0f7168118f83169c" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# Upstream builds its runtime as a chain: this exact base image (tag
# unholy-fusion-prod-ready, pinned by digest above), recipe/Dockerfile.dspark-runtime-overlay,
# then the three NVFP4 stage Dockerfiles, and its vision-exp branch adds a port on top.
# Reproduce the chain here without editing upstream's files. The overlay tree is copied
# over the vLLM package exactly as its Dockerfile does (Patch 3 scheduler fix, Patch 4
# draft shared-expert loader fix, native non-uniform-batch handling). Each NVFP4 stage
# file keeps its anchor patch as the single quoted-PY heredoc that upstream's own sparkrun
# recipe extracts, so do the same. The vision port (ViT, aligner, multimodal registry
# alias) is applied unconditionally; upstream states that every port patch is guarded on
# vision_n_layers, so the text-only 0731 checkpoint is served by the same files.
COPY recipe /opt/vonk/build/recipe
COPY vision-exp/port /opt/vonk/build/port
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.tonyd2wild-dspark

RUN set -eu; \
    V=/opt/env/lib/python3.12/site-packages/vllm; \
    B=/opt/vonk/build; \
    cp -a "$B/recipe/overlay/vllm/." "$V/"; \
    awk '$1 == "COPY" { print $3 }' "$B/recipe/Dockerfile.dspark-runtime-overlay" | xargs /opt/env/bin/python -m py_compile; \
    awk '/[<][<].PY.$/ { f = 1; next } /^PY$/ { f = 0 } f' "$B/recipe/Dockerfile.dspark-runtime-overlay" | /opt/env/bin/python -; \
    for stage in a b c; do \
      awk '/[<][<].PY.$/ { f = 1; next } /^PY$/ { f = 0 } f' "$B/recipe/nvfp4/Dockerfile.stage-$stage" | /opt/env/bin/python -; \
    done; \
    M="$V/models/deepseek_v4/nvidia"; \
    /opt/env/bin/python "$B/port/patch_vision.py" "$M/model.py"; \
    /opt/env/bin/python "$B/port/patch_registry.py" "$V/model_executor/models/registry.py"; \
    cp "$B/port/ds4v_vision.py" "$B/port/ds4v_mm.py" "$M/"; \
    find "$V" -name __pycache__ -type d -prune -exec rm -rf {} + ; \
    /opt/env/bin/python "$B/verify-runtime.py"; \
    rm -rf "$B"

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

# The parent image points every cache at /cache/huggingface; move them under the
# platform-owned writable volume.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_XET=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/state \
    HF_HOME=/state/huggingface \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    DG_JIT_CACHE_DIR=/outputs/cache/deepgemm \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache/flashinfer \
    TILELANG_CACHE_DIR=/outputs/cache/tilelang \
    TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
