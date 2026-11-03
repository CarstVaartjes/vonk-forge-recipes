# MiaAI-Lab's published image of her Qwen3.8 Flash Next single-Spark TensorFold kit (20e15e41, MIT; TensorFold is Apache-2.0):
# TensorFold's Zig engine v1.1.4 (commit eab85828) plus her patches 0001 to 0017 (the NVIDIA FP4 vision tower, fp8 KV cache, prompt and decode
# speed work, sampled drafts, structured output) on NVIDIA's PyTorch container 26.07 (arm64), tag v1.1.4-eab8582-4baf16137ecd: the kit's
# <TF_TAG>-<patches hash> tag for patches/*.patch (label tf.patches=4baf16137ecd, tf.kernels=embedded), the digest the kit's scripts/config.sh
# pins in IMAGE_DIGEST. This recipe runs that image as published, by digest, and does not rebuild it from the kit's sources.
# This layer only adds the Controller launcher wrapper at the platform's path, the licence texts and the non-root user.
# Base: the linux/arm64 image ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold, by digest.
FROM ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold@sha256:5c6c42b9747318c1cd8b6cb85cb69493fff2e82cb23b61e49b9cf54cd09ac28e

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold" \
      org.opencontainers.image.revision="20e15e41f2f514a0d287710f4665ee1b5009ba41" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN test -x "$(command -v tensorfold-native)" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# The engine's CUDA kernels are embedded in the image (tf.kernels=embedded); only the driver's compute cache is written, under the platform's /outputs/cache.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
