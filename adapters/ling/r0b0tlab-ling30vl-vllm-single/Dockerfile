FROM ghcr.io/r0b0tlab/ling30vl-nvfp4-mp-sm121@sha256:912f416de254f0a20562a1e33d2d6b20a379cc38884764839df09d66fbf945ee

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/ling30vl-nvfp4-mp-sm121" \
      org.opencontainers.image.revision="739081ce45a3128afb136d8249566fb013d4e0cd" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
