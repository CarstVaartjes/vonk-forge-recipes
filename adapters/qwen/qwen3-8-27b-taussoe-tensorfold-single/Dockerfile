# taussoe/TensorFold branch glm-long-context at c32abe92 (MIT; TensorFold 0.3.6.3 plus the creator's long-context work) with the Qwen3.8-27B CUDA engine and DFlash2 drafting, on NVIDIA's PyTorch container (arm64).
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

LABEL org.opencontainers.image.source="https://github.com/taussoe/TensorFold" \
      org.opencontainers.image.revision="c32abe92411eced43565a63f5be7fe3595f8e22c" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.recipe.source="https://github.com/taussoe/tensorfold-recipes/tree/d8a7c09c893eb41be5e93dbd684ce7d4f127d7ac"

# NGC exports PIP_CONSTRAINT for its own stack; the hash-pinned lock below is resolved without it and leaves torch and numpy alone.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

COPY vendor/tensorfold-c32abe92411eced43565a63f5be7fe3595f8e22c.tar.gz /tmp/tensorfold.tar.gz
COPY requirements.lock /tmp/requirements.lock
RUN echo "8bf12b9a2e44cbc78e669eb08b3d02aba8ab1e4dae0d66dfa1c5e856f7f836d4  /tmp/tensorfold.tar.gz" | sha256sum -c - \
    && mkdir /tmp/tensorfold-src \
    && tar -xzf /tmp/tensorfold.tar.gz -C /tmp/tensorfold-src --strip-components=1 \
    && PIP_CONSTRAINT= python -m pip install --ignore-installed --no-deps --no-cache-dir --require-hashes --requirement /tmp/requirements.lock \
    && python -c "import setuptools; assert int(setuptools.__version__.split('.')[0]) >= 69, setuptools.__version__" \
    && PIP_CONSTRAINT= python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/tensorfold-src \
    && python -c "import tensorfold.cuda.server, tensorfold.families.qwen3_5" \
    && tensorfold --version \
    && rm -rf /tmp/tensorfold.tar.gz /tmp/tensorfold-src /tmp/requirements.lock

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
