FROM ghcr.io/0xsero/step-5-preview-spark@sha256:ddc5d0b88594d3267f5d0e8cdaeb21187ccb6d0b2032e301694473f198a3cf96
LABEL org.opencontainers.image.source="https://github.com/0xSero/Step-5-Preview-Four-Sparks/tree/662949a6c1244c1e8a5c44ed62cedb89a207cbcc" \
      org.opencontainers.image.revision="662949a6c1244c1e8a5c44ed62cedb89a207cbcc"
RUN install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs
COPY --chmod=0755 launcher /opt/vonk/bin/vllm
ENV HOME=/state HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
