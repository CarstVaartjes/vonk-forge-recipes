FROM docker.io/vllm/vllm-openai@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.6-35B-A3B-NVFP4-vLLM" \
      org.opencontainers.image.revision="5a23ee406d403569fc789bb70eebb7ff20508fe2"

# The base is the stock vllm/vllm-openai:v0.24.0 image (linux/arm64 manifest
# pinned by digest) that upstream's start.sh runs. Upstream bind-mounts
# chat_template.jinja at run time; it is baked in here instead so the image
# stays read-only.
COPY chat_template.jinja /opt/vonk/chat_template.jinja
COPY --chmod=0755 vllm /opt/vonk/bin/vllm
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache

ENV HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache/xdg \
    XDG_CONFIG_HOME=/outputs/cache/config \
    HF_HOME=/outputs/cache/huggingface \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_HOME=/outputs/cache/torch \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    UV_CACHE_DIR=/outputs/cache/uv

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
