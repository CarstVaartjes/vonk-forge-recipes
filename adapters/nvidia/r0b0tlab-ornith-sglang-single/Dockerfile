FROM ghcr.io/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang@sha256:3da6bf4220b511b309818d3af1560e8eabc4e91e008cef4347b4dfdff76ccbd3

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang" \
      org.opencontainers.image.revision="6d0a25d9cc8b8963161cde085a83768b6bc545e0" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Base: r0b0tlab's published GB10 image (pinned venv with SGLang main and pre-warmed
# FlashInfer/Triton JIT caches, image built 2026-08-21), used unchanged. Upstream warns
# a cold FlashInfer CUTLASS compile with the model resident can run out of memory, so the
# pre-warmed caches are kept aside and seeded into the platform cache on first start.
USER root
RUN mkdir -p /opt/vonk/prewarm \
    && cp -a /home/sglang/.cache/sglang /opt/vonk/prewarm/sglang \
    && if [ -d /home/sglang/.cache/triton ]; then cp -a /home/sglang/.cache/triton /opt/vonk/prewarm/triton; fi
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    HF_HOME=/outputs/cache/huggingface \
    TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
