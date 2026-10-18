# Native TensorFold at db28187; creator-published image, thin placement wrapper.
FROM ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold@sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold" \
      org.opencontainers.image.revision="6730319ed948cc88b525bbbdc9d309451a863834" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN test -x "$(command -v tensorfold-native)" \
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
