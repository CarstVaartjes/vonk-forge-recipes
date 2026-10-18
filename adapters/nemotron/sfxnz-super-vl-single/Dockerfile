FROM vllm/vllm-openai@sha256:fd33bec1578f312a01547cc6b9691a5154c3e8bcfb76d2b35acb657bfc4eee8f
LABEL org.opencontainers.image.source="https://github.com/sfxnz/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4-DGX-SPARK" \
      org.opencontainers.image.revision="1747d21344b1a7f3ab30149edfe8b0fe093c0537"
COPY LICENSE /opt/vonk/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper /opt/vonk/bin/vllm
RUN python -c "import vllm" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
