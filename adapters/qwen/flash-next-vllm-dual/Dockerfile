FROM docker.io/vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks" \
      org.opencontainers.image.revision="c2325b22602b51a5faf55fc2bebccc34f3f80b9f" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later AND Apache-2.0"

COPY patch_ple_layer.py patch_modelopt_mxfp8.py patch_modelopt_fp8_block_moe.py patch_qsa_fp8_kv.py patch_checkpoint_config.py detect_ple_dtype.py apply_patches.py /opt/vonk/build/
COPY --chmod=0755 qwen38-vllm-wrapper.py /opt/vonk/bin/qwen38-vllm-serve
COPY LICENSE.upstream NOTICE.upstream /opt/vonk/

RUN python3 /opt/vonk/build/apply_patches.py \
    && install --directory --mode=0755 /outputs \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && chown -R 10001:10001 /outputs /opt/vonk

ENV VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600

WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
