FROM docker.io/lmsysorg/sglang@sha256:6fb82d4fa57425bc0df5b35fba328b08bda950ecf78ce60ea63d8830fc271eb5

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/nemotron3-super-120b-a12b-nvfp4-gb10-native-mtp" \
      org.opencontainers.image.revision="19ab5365aaff58bc3c04ea4a9c9181c9c6482860" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

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
