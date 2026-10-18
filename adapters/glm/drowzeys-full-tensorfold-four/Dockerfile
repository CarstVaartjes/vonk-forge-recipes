# Creator-published Python TensorFold full-GLM TP4 image; thin placement wrapper.
FROM ghcr.io/drowzeys/keys-tensorfold-glm53-tp4-dgx-spark@sha256:77da91d4fd80fe1e1f7ca533b47f8977b2454f0587cb15f9ff53be9c43c5293b

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-TensorFold-GLM-5.3-TP4-4x-DGX-Spark" \
      org.opencontainers.image.revision="ea05ec696d1c408268a3d1de39b090b154c1ebff" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN python3 -c "import tensorfold.cli" \
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
