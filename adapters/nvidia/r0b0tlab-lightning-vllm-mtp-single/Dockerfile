# r0b0tlab's published runtime image of his nemotron-3.5-lightning-sm121-nvfp4 kit (024bb77b, MIT), tag sm121-mtp1: the
# NVIDIA vllm/vllm-openai image the kit pins (sha256:3af90144..., vLLM 0.23.1rc1, PyTorch 2.11 cu129, FlashInfer 0.6.14,
# arm64) plus the fastokens 0.3.1 package its guide profile needs (the kit's runtime/Dockerfile, one RUN layer). The kit's
# launcher (runtime/launch.sh) runs this image; this recipe runs it as published, pinned by digest (the tag is mutable; it
# was resolved once to the digest below, which is the registry manifest digest the kit records), and does not rebuild it.
# This layer only adds the Controller placement wrapper at the platform's launcher path, the licence text and the
# non-root user. Base: the linux/arm64 image ghcr.io/r0b0tlab/nemotron-lightning-repro-runtime.
FROM ghcr.io/r0b0tlab/nemotron-lightning-repro-runtime:sm121-mtp1@sha256:9fcb3b11818b035605783471c10cd169aaa3ffd62eafd9f1c647e9dce40d2dda

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/nemotron-3.5-lightning-sm121-nvfp4" \
      org.opencontainers.image.revision="024bb77b60482ef833e85376260ccb5d4b767225" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
RUN { test -x /usr/local/bin/vllm || test -x /opt/vllm/.venv/bin/vllm; } \
    && python3 -c 'import fastokens' \
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
