FROM ghcr.io/r0b0tlab/sm121-vllm-nvfp4@sha256:0f4ede2569e9f5cbf72ecc47a0cd05d532e3db713a4a5e2b09335db5891b69a1

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4" \
      org.opencontainers.image.revision="d0343b430a2853f0422665c93aac0fe9984a7a79" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Base: r0b0tlab's published production image (source-built vLLM v0.25.1 with the native
# calibrated W4A4 reroute for SM121, pinned by the immutable digest in upstream's
# sparkrun recipe), used unchanged; only the launcher and user are ours.
USER root
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
