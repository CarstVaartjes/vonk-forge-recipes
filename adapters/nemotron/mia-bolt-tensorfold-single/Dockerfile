# Native Bolt runtime with creator patches 5e11cf1ed295; thin Controller wrapper.
# The kit has moved on to TensorFold v1.1.6; its image is not published yet, so this stays on the published 46a08c1 build.
FROM ghcr.io/miaai-lab/nemotron-3.5-bolt-tensorfold-dgx-spark@sha256:9d39df6d1526ede3d0afb983d122643284ae4a92f50c07ee544c47e2b7befa1e

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Nemotron-3.5-Bolt-TensorFold-DGX-Spark" \
      org.opencontainers.image.revision="6b064a31dd47e4f959e9041d6129eb2bad830481" \
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
