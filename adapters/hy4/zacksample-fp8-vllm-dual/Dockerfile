FROM docker.io/vllm/vllm-openai@sha256:5e2c1badbca60595d6591c60feb4b8547f59da96e7a91710d40239250e721aba

LABEL org.opencontainers.image.source="https://github.com/ZackSample/hy4-flash-dgx-sparks" \
      org.opencontainers.image.revision="a0e88d769cdb1f9f4bbfdc06e1a621a6a1ffe169" \
      org.opencontainers.image.licenses="Apache-2.0"

# The base is Tencent's official vllm/vllm-openai:hy4-flash image (linux/arm64,
# vLLM 77766db1) that upstream's start.sh runs, pinned by digest. Upstream lays
# four patched files over the image's copies with bind mounts at every start;
# here the same files are copied in once at build time so the image is read-only
# at run time. They add the GB10 (sm_12x) attention path and the on-disk Engram
# reader.
ARG V=/usr/local/lib/python3.12/dist-packages/vllm/models
COPY patches/engram.py ${V}/hy_v4_flash/nvidia/engram.py
COPY patches/attention.py ${V}/hy_v4_flash/nvidia/attention.py
COPY patches/sm120.py ${V}/hy_v4_flash/nvidia/sm120.py
COPY patches/hy4_attention.py ${V}/hy_v4/nvidia/attention.py
COPY verify-runtime.py /opt/vonk/build/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN python3 -m compileall -q \
        ${V}/hy_v4_flash/nvidia/engram.py ${V}/hy_v4_flash/nvidia/attention.py \
        ${V}/hy_v4_flash/nvidia/sm120.py ${V}/hy_v4/nvidia/attention.py \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
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
