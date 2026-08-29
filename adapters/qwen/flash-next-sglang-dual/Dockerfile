FROM docker.io/lmsysorg/sglang@sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks" \
      org.opencontainers.image.revision="344f9d0d5e9523d8398fa2804d5a3e123fd3d21a" \
      org.opencontainers.image.licenses="MIT"

COPY sm121_varlen.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/sm121_varlen.py
COPY qsa_nvfp4_kv.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa_nvfp4_kv.py
COPY patch-qsa.py /opt/vonk/build/patch-qsa.py
COPY apply_nvfp4_patches.py install-nccl.py verify-runtime.py /opt/vonk/build/
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve

RUN python3 /opt/vonk/build/patch-qsa.py \
    && python3 /opt/vonk/build/apply_nvfp4_patches.py \
    && python3 /opt/vonk/build/install-nccl.py \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    LD_PRELOAD=/opt/vonk/lib/libnccl.so.2

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
