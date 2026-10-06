# MiaAI-Lab's published image of her Qwen3.8 Flash Next single-Spark TensorFold kit (4c0dea8e, Apache-2.0): TensorFold v0.6.1
# (Apache-2.0, commit 17c73e18) plus her patches (video and many images, prompt rows, copy drafts, and the optional draft
# languages) on NVIDIA's PyTorch container 26.07 (arm64), tag v0.6.1-f8a0b4cb702b: the kit's <TF_VERSION>-<patches hash>
# tag for patches/*.patch plus patches/languages/*.patch (label tf.patches=f8a0b4cb702b), the same patch set this recipe
# used to rebuild. This recipe runs that image as published, by digest, and does not rebuild it from the kit's sources.
# This layer only adds the Controller launcher wrapper at the platform's path, the licence texts and the non-root user.
# Base: the linux/arm64 image ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold, by digest.
FROM ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold@sha256:a503c88acdcfaab6231cf0089d747dac794cb5a7d1b121e566b180f35f8b02c6

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold" \
      org.opencontainers.image.revision="4c0dea8ebe93a1d445d6a716b5a38995ff6e9362" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN test -x "$(command -v tensorfold)" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# First start JIT-compiles the CUDA kernels for GB10 into the platform's writable /outputs/cache.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
