# MiaAI-Lab's published image of her Ornith-1.5-122B TensorFold kit (0c8fd6a0, MIT; TensorFold is Apache-2.0): TensorFold's Zig engine
# (tensorfold-native, commit d1018b31) plus her patches 0001 to 0027 (EXL3 for the Qwen3.5-MoE CUDA engine, the DFlash2 and MTP drafters,
# the prompt and decode kernels) on NVIDIA's PyTorch container 26.07 (arm64), tag zig-d1018b3-f5683f4c5651: the kit's <TF_TAG>-<patches hash>
# tag for patches/*.patch (label tf.patches=f5683f4c5651). This recipe runs that image as published, by digest, and does not rebuild it from the kit's sources.
# This layer only adds the Controller launcher wrapper at the platform's path, the licence texts and the non-root user.
# Base: the linux/arm64 image ghcr.io/miaai-lab/ornith-1.5-122b-tensorfold, by digest.
FROM ghcr.io/miaai-lab/ornith-1.5-122b-tensorfold@sha256:56c064d22327a6d0ce09d0de3930084b11841918072e9467a2dd9039b3d92b82

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Ornith-1.5-122B-TensorFold" \
      org.opencontainers.image.revision="0c8fd6a0d1e2ece7557978a7d8c3d96d137eb075" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN test -x "$(command -v tensorfold-native)" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# The engine's own (native) kernels are in the image (TENSORFOLD_CUDA_KERNELS=native); only the driver's compute cache is written, under the platform's /outputs/cache.
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
