# vLLM v0.32.0 CUDA 13, linux/arm64. Tencent plain AR serving path.
FROM docker.io/vllm/vllm-openai@sha256:6ff5ff0ddf3658d2c2d677dac5ca2d4726ce0156b4371166d918181b210bd4bc
LABEL org.opencontainers.image.source="https://github.com/Tencent-Hunyuan/HunyuanOCR" \
      org.opencontainers.image.revision="1ef4179e53cf860f7e6fd8276a292e5d05b3e927"
RUN install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs
COPY --chmod=0755 launcher /opt/vonk/bin/vllm
ENV HOME=/state HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
