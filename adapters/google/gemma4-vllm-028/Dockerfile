FROM docker.io/vllm/vllm-openai@sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4

LABEL org.opencontainers.image.source="https://huggingface.co/google/gemma-4-26B-A4B-it" \
      org.opencontainers.image.revision="4d7ae4984b7db7de8f8457170b3f1a419ee76d52" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Offline vLLM 0.28 ARM64 image-chat adapter for Gemma 4 26B-A4B IT" \
      io.vonk.vllm.build-commit="2cf0a6915ce544dc493a0990f2ea38d81601128a"

COPY model-interface-smoke.py /tmp/model-interface-smoke.py
COPY --chmod=0755 vllm-wrapper.sh /opt/vonk/bin/vllm
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache \
    && python /tmp/model-interface-smoke.py \
    && rm /tmp/model-interface-smoke.py

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
