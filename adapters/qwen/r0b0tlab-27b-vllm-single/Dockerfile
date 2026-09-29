FROM ghcr.io/r0b0tlab/qwen38-27b-nvfp4-sm121@sha256:5bd3f329c531da4cd5f41f2c32d4ed9527b2131b88032b737364fb582e6b775f

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-vllm" \
      org.opencontainers.image.revision="a96186f8973023365bddb34b6b34fa889d4c40ae" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Base: r0b0tlab's published vLLM 0.27.2rc0 SM121 wheel image (v0.27.2rc0-sm121).
# Overlay: upstream docker/Dockerfile.dflash2 at the pinned commit (DFlash2 hunks
# from vllm-project/vllm#52816 @ 19c93519 plus r0b0tlab's SM121 fixes), then its
# fail-closed audit.
COPY overlay/vllm/config/vllm.py \
     /usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py
COPY overlay/vllm/model_executor/models/registry.py \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py
COPY overlay/vllm/model_executor/models/qwen3_dflash.py \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash.py
COPY overlay/vllm/model_executor/models/qwen3_dflash2.py \
     /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_dflash2.py
COPY overlay/vllm/v1/worker/gpu/spec_decode/__init__.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/__init__.py
COPY overlay/vllm/v1/worker/gpu/spec_decode/dflash2/ \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/dflash2/
COPY audit_dflash2_image.py /opt/audit_dflash2_image.py
RUN python3 /opt/audit_dflash2_image.py

COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
