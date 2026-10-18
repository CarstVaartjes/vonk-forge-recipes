# Creator fork preserves DeepSeek V4.1 support absent from native TensorFold.
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

LABEL org.opencontainers.image.source="https://github.com/sfxnz/TensorFold" \
      org.opencontainers.image.revision="19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.recipe.source="https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-TensorFold-2x-DGX-Spark/tree/f7a2ad22db3a3ba20ecc109953e205e31d02364d"

# Install only the pinned engine; the creator verifies its dependencies are already in this base.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

COPY vendor/tensorfold-19f5478ba778531ffb9a1ebb9c1ca7e06b8ceacb.tar.gz /tmp/tensorfold.tar.gz
RUN echo "3d73f08395628a12f640895885cbfe1a5dbe6db93057ce2306ac09217f4db84d  /tmp/tensorfold.tar.gz" | sha256sum -c - \
    && mkdir /tmp/tensorfold-src \
    && tar -xzf /tmp/tensorfold.tar.gz -C /tmp/tensorfold-src --strip-components=1 \
    && python -c "import setuptools; assert int(setuptools.__version__.split('.')[0]) >= 69, setuptools.__version__" \
    && PIP_CONSTRAINT= python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/tensorfold-src \
    && python -c "import tensorfold.cuda.server, tensorfold.families.deepseek_v41" \
    && tensorfold --version \
    && rm -rf /tmp/tensorfold.tar.gz /tmp/tensorfold-src

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# First start JIT-compiles the CUDA kernels for GB10 into the platform's writable /outputs/cache.
# TORCH_CUDA_ARCH_LIST=12.1 is taussoe's docker/tensorfold-spark/Dockerfile.local: the container's default list also
# builds for GPUs without the FP8 MMA the prompt kernels need, and takes longer.
ENV TORCH_CUDA_ARCH_LIST=12.1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
