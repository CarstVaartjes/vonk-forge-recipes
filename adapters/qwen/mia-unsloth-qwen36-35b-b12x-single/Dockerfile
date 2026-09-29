FROM ghcr.io/miaai-lab/mia-vllm-gb10-linear-b12x@sha256:19627342e1da2607f4db50745dca30e57d7dd0ebff06062f03fd69b43a252931

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Unsloth-Qwen3.6-35b-NVFP4-DGX-Spark" \
      org.opencontainers.image.revision="79c9e6f359f6101cdacd0dfd6fe9861ae2493a4d"

# The base is MiaAI-Lab's published mia-vllm-gb10-linear-b12x image (linux/arm64,
# built 2026-07-29 from timothystewart6/vllm-gb10 v0.26.0-gb10.2 plus the
# docker/ soft-fallback patch) that upstream's start.sh pulls as :latest, pinned
# by digest. Upstream's own docker/ sources are vendored beside this file for
# provenance; the image is not rebuilt because its floating base tag has since
# moved on (v0.30.0-gb10.3).
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
