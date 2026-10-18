# Published TensorFold v0.6.0 with Mia patches fb91a0019fdc; thin Controller launcher layer.
FROM ghcr.io/miaai-lab/qwen3.8-27b-dgx-spark-tensorfold@sha256:043d192b9e1725f55c5439fc9672055de891433e5b31448ed8963918e3c460e9

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold" \
      org.opencontainers.image.revision="c11ff138871cf43d12898d79dbfa4dc952d17ca4" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN test -x "$(command -v tensorfold)" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# First start JIT-compiles the CUDA kernels for GB10 (and the RoCE all-gather) into the platform's writable /outputs/cache.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
