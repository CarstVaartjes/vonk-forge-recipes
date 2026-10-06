# r0b0tlab's published image of his nemotron-3.5-lightning-sglang-dflash-sm121 kit (3baf136c, MIT), tag
# v0.1.0-sm121-dflash: the kit's thin provenance layer (its entrypoint and ready-gate scripts) over the day-0
# lmsysorg/sglang:dev-nemotron3-5-lightning image (arm64 manifest sha256:6da24ff7...). The kit publishes it (scripts/
# click_run_dflash.sh pulls it); this recipe runs that image as published, pinned by digest (the tag is mutable; it was
# resolved once to the digest below), and does not rebuild it. This layer only adds the Controller placement wrapper at
# the platform's launcher path, the licence text and the non-root user; the kit's own entrypoint is replaced by the
# recipe's launch arguments. Base: the linux/arm64 image ghcr.io/r0b0tlab/nemotron-3.5-lightning-sglang-dflash-sm121.
FROM ghcr.io/r0b0tlab/nemotron-3.5-lightning-sglang-dflash-sm121:v0.1.0-sm121-dflash@sha256:b868b726ce08fcb771501336f1c096a3820da575cfc04c8ee40cacf7adfabe61

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/nemotron-3.5-lightning-sglang-dflash-sm121" \
      org.opencontainers.image.revision="3baf136cb09830656e6aeba19a20d39c3f02e81b" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
RUN python3 -c "import importlib.util as u, sys; sys.exit(u.find_spec('sglang') is None)" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
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
