FROM ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/MiMo-V2.6-Flash-DGX-Spark-Recipe" \
      org.opencontainers.image.revision="13621bb3cc6fd30a94d53609320599d1f1134686" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# The stock image does not serve MiMo-V2.6-Flash-RL correctly on GB10. The
# upstream recipe bind-mounts three patched vLLM files over the image's copies
# (fused fp8 QKV loading, DFlash on the omni class, fp8 KV that actually applies).
# Bake the same exact files in, and add the audio libraries the image lacks
# (soundfile, PyAV), pinned by wheel hash.
COPY patches/mimo_v2.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2.py
COPY patches/mimo_v2_omni.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/mimo_v2_omni.py
COPY patches/triton_attn_diffkv.py /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/triton_attn_diffkv.py
COPY requirements-audio.txt /opt/vonk/requirements-audio.txt
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY verify-runtime.py /opt/vonk/verify-runtime.py

RUN pip install --no-cache-dir --no-deps --require-hashes -r /opt/vonk/requirements-audio.txt \
    && python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    VLLM_USE_DEEP_GEMM=0 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
