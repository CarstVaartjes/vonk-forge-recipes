FROM docker.io/vllm/vllm-openai@sha256:41b54fb42c66a670a8b27e613ebef05898f24b9ab1bdab28bd00c877bd4935f4

LABEL org.opencontainers.image.source="https://github.com/Tencent/UI-Mate" \
      org.opencontainers.image.revision="d185dc9d74cfcab3a890d7ffb2bb011ecdd64c64" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.vllm.build-commit="2cf0a6915ce544dc493a0990f2ea38d81601128a" \
      io.vonk.ui-mate.upstream-archive-sha256="12046e80b390539417bfc803d1effcd7b867a4bb95ac8a20a5631ce60db9ab4d"

COPY agents /opt/vonk/ui-mate/agents
COPY LICENSE.upstream NOTICE /opt/vonk/ui-mate/
COPY ui-mate-step.py vllm-wrapper.sh /opt/vonk/bin/

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache \
    && chmod 0755 /opt/vonk/bin/ui-mate-step /opt/vonk/bin/vllm-wrapper.sh \
    && ln -s /opt/vonk/bin/vllm-wrapper.sh /opt/vonk/bin/vllm \
    && PYTHONPATH=/opt/vonk/ui-mate python -c \
      'import openai, PIL, requests; from agents.ui_mate_agent import UIMateAgent, parse_response; assert UIMateAgent and parse_response'

ENV PYTHONPATH=/opt/vonk/ui-mate \
    OPENAI_BASE_URL=http://127.0.0.1:8000/v1 \
    OPENAI_API_KEY=EMPTY \
    HOME=/outputs/cache/home \
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
