# linux/arm64 manifest of docker.io/vllm/vllm-openai:v0.24.0 (the image upstream's start.sh runs).
FROM docker.io/vllm/vllm-openai@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Nemotron-Labs-3-Puzzle-75B-DGX-Spark" \
      org.opencontainers.image.revision="c77d26681086df72aadb6bf2879745408537e0b0" \
      ai.vonkforge.runtime-interface="v1"

# Upstream's start.sh applies the aarch64 NCCL and FlashInfer patches inside the
# container at every launch; here the same commands run once at build time so
# the image is read-only at run time.
COPY patch-aarch64.sh verify-runtime.py /opt/vonk/build/

RUN bash /opt/vonk/build/patch-aarch64.sh \
    && install --directory --mode=0755 /opt/vonk/bin \
    && ln -s "$(command -v vllm)" /opt/vonk/bin/vllm \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

# CUDA 13 / GB10 library environment as set by upstream's docker run.
ENV LD_PRELOAD=/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcudart.so.13 \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib \
    PATH=/usr/local/cuda-13/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
