# MiaAI-Lab's published image of her GLM-5.3-Flash EXL3 TensorFold kit v1.7.1 (68ebd67b, Apache-2.0): TensorFold v0.6.0
# (Apache-2.0, commit c4646171) plus her 76 patches on NVIDIA's PyTorch container 26.07 (arm64), tag v0.6.0-1692d2df78d2.
# The kit pins it by digest (scripts/config.sh IMAGE_DIGEST); this recipe runs that image as published and does not
# rebuild it from the kit's sources. This layer only adds the Controller placement wrapper at the platform's launcher
# path, the licence texts and the non-root user. Base: the linux/arm64 image
# ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold, by digest.
FROM ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:a8067cd7e14c14fa83d1dbed60261428f6d1737cec4554445573354af040dd7c

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold" \
      org.opencontainers.image.revision="68ebd67b5326974b8004009e202268b1fa7c551d" \
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
