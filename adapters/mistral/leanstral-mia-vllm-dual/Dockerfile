FROM docker.io/vllm/vllm-openai@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Leanstral-1.5-119B-A6B_2x_DGX_Sparks_vLLM" \
      org.opencontainers.image.revision="94d729896226db3eaf7f584e79b7c1f0028bab1d" \
      org.opencontainers.image.licenses="Apache-2.0"

# The pinned vLLM v0.24.0 image has no Ray CLI. Upstream runs
# `pip install 'ray[default]'` in both containers at start-up; install a pinned
# Ray here so nothing is downloaded at run time.
RUN python3 -m pip install --no-cache-dir "ray[default]==2.58.0"

COPY verify-runtime.py /opt/vonk/build/verify-runtime.py
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    RAY_memory_usage_threshold=0.99 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
