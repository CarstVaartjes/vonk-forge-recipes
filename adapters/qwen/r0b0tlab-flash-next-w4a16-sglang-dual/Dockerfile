FROM ghcr.io/r0b0tlab/qwen38-flash-next-w4a16-sm121-sglang@sha256:9d7406a4f2b5d829c2646e8408ce0d02ccd20d7b98f5d7a3198af3c6a81854e0

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/qwen38-flash-next-w4a16-sm121-sglang" \
      org.opencontainers.image.revision="7bfacc510506e955a1392f94b87213b798e45e51" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Base: r0b0tlab's published production full-context image (SGLang source 7d2e8fcb,
# runtime image ID sha256:49d06b35...), used unchanged; it is the image upstream's
# click-run pulls by this digest.
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.r0b0tlab-flash-next-w4a16
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor \
    TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
