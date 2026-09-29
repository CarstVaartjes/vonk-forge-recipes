FROM docker.io/lmsysorg/sglang@sha256:a04d9a1a7ffe371b05230aecab001d4ba2bfa0e5c137bc56409ecc4cbc3ac864

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/nemotron-3.5-lightning-sglang-dflash-sm121" \
      org.opencontainers.image.revision="3baf136cb09830656e6aeba19a20d39c3f02e81b" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
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
