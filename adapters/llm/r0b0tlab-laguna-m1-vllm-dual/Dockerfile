FROM ghcr.io/r0b0tlab/vllm-laguna-m1-nvfp4-sm121@sha256:6e2dfa4c48ac73965890a864dc1510017f42981e743b3f2834b4b189ad7712a8

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/laguna-m1-nvfp4-sm121-vllm" \
      org.opencontainers.image.revision="3446af5bf5cdaffb37eb51487c6d570bc84a7f7f" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's published image tag cu130-sm121-arm64-dda4668b (arm64, CUDA 13.0, torch 2.11.0+cu130,
# vLLM dda4668b), pinned by registry digest. This layer only adds the placement wrapper (Ray head/worker)
# and the non-root user.
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
