FROM ghcr.io/r0b0tlab/vllm-step37-flash-nvfp4-sm121@sha256:c218aa40a162889959f83d055e0b69e92c1f1bacd53a13df2cfa7d8c3c44e52e

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/step37-flash-nvfp4-sm121-vllm-docker" \
      org.opencontainers.image.revision="e79e192dbe1120302406abed97bf26be4c189484" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's published image tag cu130-v0.1.dev16944-arm64-tp2 (vLLM 0.1.dev16944+ge9c8946e7,
# Ray 2.55.1, FlashInfer 0.6.11.post2, arm64, CUDA 13.0), pinned by registry digest;
# docker/Dockerfile.publish in this directory is upstream's recipe for it. This layer only adds the
# placement wrapper (Ray head/worker) and the non-root user.
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN test -x "$(command -v vllm)" && test -x "$(command -v ray)" \
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
