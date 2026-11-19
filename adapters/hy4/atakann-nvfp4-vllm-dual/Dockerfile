FROM docker.io/vllm/vllm-openai@sha256:5e2c1badbca60595d6591c60feb4b8547f59da96e7a91710d40239250e721aba

LABEL org.opencontainers.image.source="https://github.com/atakann/dgx-spark-recipes" \
      org.opencontainers.image.revision="80320614203b8f2e40036e676c575c03de60f34a" \
      org.opencontainers.image.licenses="MIT"

# The base is Tencent's official vllm/vllm-openai:hy4-flash image (linux/arm64,
# vLLM 77766db1) that upstream's images/vllm-hy4-flash/Dockerfile builds on,
# pinned by digest. Its six Python files are copied in exactly as upstream
# does: the GB10 (sm_12x) attention sink on FlashInfer's SM120 sparse-MLA
# kernel, fp8_ds_mla KV on sliding-window MLA, ModelOpt NVFP4 expert loading
# and a DFly drafter that accepts an NVFP4 target.
ARG V=/usr/local/lib/python3.12/dist-packages/vllm
COPY images/vllm-hy4-flash/sm120.py images/vllm-hy4-flash/attention.py images/vllm-hy4-flash/model.py ${V}/models/hy_v4_flash/nvidia/
COPY images/vllm-hy4-flash/hy_v4_attention.py ${V}/models/hy_v4/nvidia/attention.py
COPY images/vllm-hy4-flash/dfly_config.py ${V}/models/hy_v4/dfly_config.py
COPY images/vllm-hy4-flash/mla_attention.py ${V}/model_executor/layers/attention/mla_attention.py
COPY verify-runtime.py /opt/vonk/build/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN python3 -m compileall -q \
        ${V}/models/hy_v4_flash/nvidia ${V}/models/hy_v4/nvidia/attention.py \
        ${V}/models/hy_v4/dfly_config.py ${V}/model_executor/layers/attention/mla_attention.py \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV CUTE_DSL_ARCH=sm_121a \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
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
