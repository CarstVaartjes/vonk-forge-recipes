FROM ghcr.io/r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm@sha256:3361f68b966ae84940c068aa7fd522078fd7f3cb087c506edee93041ce031892

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm" \
      org.opencontainers.image.revision="b5c2dd2ff156040d696b93d7b0f6ed9eeefe7496" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Base: r0b0tlab's published SM121 vLLM 0.24.1-dev image (immutable manifest recorded in
# upstream's docs/NEW_SESSION.md), used unchanged; only the launcher and user are ours.
USER root
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 vllm-api /opt/vonk/bin/vllm
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
