FROM docker.io/vllm/vllm-openai@sha256:2cc49b81319f7a66a33dd8bd63a7bfddae079122b33ce51989b6828a1f038c37

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Gemma-4-26B-A4B-DGX-Spark-18-concurrencies" \
      org.opencontainers.image.revision="39e94192bd6f114a222d434cf2bf4dcf173248a5" \
      org.opencontainers.image.licenses="Apache-2.0"

# Upstream bind-mounts a patched gemma4.py over the image's copy (ModelOpt NVFP4
# lm_head tying). verify-runtime.py installs the same byte-identical file.
COPY gemma4.py /opt/vonk/build/gemma4.py
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py
# vLLM v0.25.1 carries the #43957 regression that breaks Gemma 4 MTP on MoE
# targets (issue #47794); apply the upstream fix (#47953, released in v0.27.0).
COPY patch-vllm-pr47953.py /opt/vonk/build/patch-vllm-pr47953.py
COPY --chmod=0755 vllm-wrapper.sh /opt/vonk/bin/vllm

RUN python3 /opt/vonk/build/verify-runtime.py \
    && python3 /opt/vonk/build/patch-vllm-pr47953.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
