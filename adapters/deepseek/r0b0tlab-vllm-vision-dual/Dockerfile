FROM ghcr.io/r0b0tlab/dsv4-flash-vision-exp-vllm-sm121@sha256:78eb0a105a3b4cfc9e28da477d1c5285ffc76e3fb1e621d26799906976c8fbcc

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/dsv4-flash-vision-exp-vllm-sm121" \
      org.opencontainers.image.revision="d956795219d7eac63b3c3f2e30b7c36c9ed0b685" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's measured image vision-54631-fi512b-k5adapt (arm64, image ID sha256:f8b73f96...).
# docker/Dockerfile.overlay-54631, Dockerfile.fi512 and Dockerfile.sm121-adaptive in this directory
# are upstream's overlay chain on vllm/vllm-openai:deepseekv4-flash-vision-arm64-cu130, already baked in.
# This layer only adds the placement wrapper and the non-root user.
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
