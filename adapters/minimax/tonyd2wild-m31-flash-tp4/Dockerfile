# MiniMax-M3.1-Flash TP4 on four DGX Sparks (tonyd2wild kit, 7b46c7f5): the vLLM project's published arm64 image
# vllm/vllm-openai:minimax-m31-flash-arm64-cu134 (vLLM 0.1.dev23341+g8744dd2fb, CUDA 13.4, torch 2.14.0+cu130,
# FlashInfer 0.7.2), which the kit's node_run.sh runs. This recipe runs that image, pinned by digest, and does not
# rebuild it. This layer adds the Controller placement wrapper at the platform's launcher path, the CUDA 13.4
# forward-compat libcuda ahead of the host's 13.0 one (kit README gotcha 1, node_run.sh LD_LIBRARY_PATH), and the
# non-root user. Base: the linux/arm64 image docker.io/vllm/vllm-openai, by digest.
FROM docker.io/vllm/vllm-openai:minimax-m31-flash-arm64-cu134@sha256:5d76347628973fb80519a919a2f2740b6645b62cba3fe2bd87b25f111bd0e948

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/MiniMax-M3.1-Flash-TP4-DGX-Sparks" \
      org.opencontainers.image.revision="7b46c7f5d55328d1c8ee97880c50d83528a34d90" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN { test -x /usr/local/bin/vllm || test -x /opt/vllm/.venv/bin/vllm; } \
    && test -d /usr/local/cuda-13.4/compat \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --mode=0755 /inputs \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV LD_LIBRARY_PATH=/usr/local/cuda-13.4/compat:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/nvidia/lib \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
