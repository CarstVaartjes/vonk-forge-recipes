FROM ghcr.io/r0b0tlab/qwen38-flash-next-nvidia-nvfp4-sm121-sglang@sha256:de34276bf6e00e86b80625d2e2b365a7109fa7d934cb5691bf3ff87f19302962

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/qwen38-flash-next-nvidia-nvfp4-sm121-sglang" \
      org.opencontainers.image.revision="1dc3c2f0796945b175b541076f036f7f21a145d8" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Base: r0b0tlab's published v1.0.1-sm121-nextn image (patched SGLang 0.5.19 SM121,
# profiles, locks and the profile compiler entrypoint). Upstream's runtime expects a
# writable /cache; the platform provides /outputs/cache, so /cache points there.
USER root
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.r0b0tlab-flash-next
COPY --chmod=0755 flash-next-serve /opt/vonk/bin/flash-next-serve
RUN rm -rf /cache \
    && ln -s /outputs/cache /cache \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

WORKDIR /opt/r0b0tlab
USER 10001:10001
ENTRYPOINT []
