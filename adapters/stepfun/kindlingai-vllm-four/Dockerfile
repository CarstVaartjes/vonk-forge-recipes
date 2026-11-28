# KindlingAI's Step-5 Preview NVFP4 recipe for four DGX Sparks (https://github.com/kindlingai/recipes/tree/13c24ac483eaf8b2fcb828a62ae2604bcc038b92/step-5-preview/dgx-spark/4x/nvfp4-vllm, MIT)
# runs the vLLM Step-5 Preview image as the kit pins it (vllm/vllm-openai:step-5-preview, an arm64+amd64 index) with the
# creator's read-only FP8 KV overlay on PYTHONPATH. Nothing is compiled. This layer only adds the overlay (vendored
# byte for byte, its overlay.sha256 checked at build time as the creator's entrypoint checks it at start), the
# Controller placement launcher, the licence texts and the non-root user. The creator's NFS shard split and mount
# scripts are left out: the platform owns the model mounts and every node gets the full Model.
# Base: the linux/arm64 manifest of vllm/vllm-openai:step-5-preview, by digest.
FROM docker.io/vllm/vllm-openai@sha256:784f5812c03fe18b11e33a01ea04e063b31fc14bc0fb1a1109f908adcc0619d1

LABEL org.opencontainers.image.source="https://github.com/kindlingai/recipes" \
      org.opencontainers.image.revision="13c24ac483eaf8b2fcb828a62ae2604bcc038b92" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

COPY fp8-kv /opt/step5-fp8kv
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 launcher /opt/vonk/bin/vllm
RUN cd /opt/step5-fp8kv && sha256sum -c overlay.sha256 \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
