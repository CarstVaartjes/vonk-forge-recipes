FROM ghcr.io/miaai-lab/hy3-dual-dgx-spark@sha256:9017669d03d30831973a44dd4ae330a5ddb710397b08eee3177fd077e1b5598f

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Hy3-Dual-DGX-Spark" \
      org.opencontainers.image.revision="d7510b52b234569b66e6263c5ead982a4eb62756"

# The base is MiaAI-Lab's published vLLM image (ARM64, CUDA 13.2, torch 2.11,
# system Python 3.12) that upstream's start.sh runs, pinned by digest. Upstream
# patches the hy_v3 parsers inside the container at every serve; here the same
# sed commands run once at build time so the image is read-only at run time.
COPY patch-opensource-tokens.sh verify-runtime.py /opt/vonk/build/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN bash /opt/vonk/build/patch-opensource-tokens.sh \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    RAY_memory_usage_threshold=0.99 \
    RAY_memory_monitor_refresh_ms=0 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
