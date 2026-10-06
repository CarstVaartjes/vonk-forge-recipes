# syntax=docker/dockerfile:1.7
# 0xSero's published image of his glm-5.3-flash-sglang-sm121 kit (dfb1bb7e): the digest-pinned
# lmsysorg/sglang GLM-5.3-Flash image with the kit's six GLM/SGLang SM121 patch files, chat
# template, launch script and SM121 environment already baked in. The kit pins it by digest
# (manifest.json image.published); this recipe runs that image as published and does not
# rebuild it from the kit's sources. This layer only adds the Controller placement wrapper at
# the platform's launcher path (it reproduces the kit's `exec python3 -m sglang.launch_server`),
# the offline settings and the non-root user. Base: the linux/arm64 image
# ghcr.io/0xsero/glm-5.3-flash-sglang-sm121, by digest.
FROM ghcr.io/0xsero/glm-5.3-flash-sglang-sm121@sha256:7aff51ea7050480dc47137055b5201b73e23e7803d9439233742cab65e3e5609

LABEL org.opencontainers.image.source="https://github.com/0xSero/glm-5.3-flash-sglang-sm121" \
      org.opencontainers.image.revision="dfb1bb7e45c20058d37df7a39cceda45a9d216a8" \
      ai.vonkforge.runtime-interface="v1"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve

RUN python3 -c "import importlib.util as u, sys; sys.exit(u.find_spec('sglang') is None)" \
    && test -r /opt/glm53/chat-template-mm.jinja \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache \
    && test -x /opt/vonk/bin/sglang-serve

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
