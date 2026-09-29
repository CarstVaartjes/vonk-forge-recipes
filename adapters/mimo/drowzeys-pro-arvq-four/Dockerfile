FROM ghcr.io/drowzeys/mimo-v26-pro-arvq-spark@sha256:6c6b6aed088da63452ccc280cc0b88e3931ae7de5c27126053d0332b196769cf

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-MiMo-V2.6-Pro-MOPD-C3-Jarrelscy-ARVQ-4-DGX-Sparks-1M-Context" \
      org.opencontainers.image.revision="d8a10997f2c6b1b6140b204fc5dddddd247d894b" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# Base: upstream's prebuilt ARVQ runtime (tag 63430f7-sm121-v4, built from
# serve/image/Dockerfile at the pinned commit; Jarrelscy's vLLM fork
# 88c94233 for sm_121a plus the Keys MTP, prefill and tool-call patches).
# Upstream's entrypoint links the base vLLM package into the fork tree on every
# start; do that once here so the root filesystem can stay read-only.
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN /opt/arvq/entrypoint.sh link-only \
    && test -x /opt/vonk/bin/vllm \
    && test -x /opt/arvq/.venv/bin/python \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
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
