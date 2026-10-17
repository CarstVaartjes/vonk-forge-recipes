FROM docker.io/vllm/vllm-openai@sha256:1ebc27c3f077ef80c64db7032fb2793c2bac96bc88cb74532145dbbc556ccd1e
LABEL org.opencontainers.image.source="https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-NVFP4/tree/fc6908f53c6882d4ec48fca5ba144e41d21e1f92" \
      org.opencontainers.image.revision="fc6908f53c6882d4ec48fca5ba144e41d21e1f92"
RUN install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs
COPY --chmod=0755 launcher /opt/vonk/bin/vllm
ENV HOME=/state HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
