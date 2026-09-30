FROM ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121@sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/hy3-295b-nvfp4-gb10-benchmark" \
      org.opencontainers.image.revision="c134975aa465140064fee138e56d7dd221b28c2c" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's vLLM v0.25.0 CUDA 13 arm64 SM121 image (tag v0.25.0-cu130-sm121-arm64-702f4814-r2,
# vLLM 702f4814, image ID sha256:f734ce99...), pinned here by registry digest. This layer only adds
# the placement wrapper and the non-root user.
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN test -x "$(command -v vllm)" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
