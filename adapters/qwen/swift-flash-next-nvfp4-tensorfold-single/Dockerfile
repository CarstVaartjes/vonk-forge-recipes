# TensorFold v0.3.6.3 (MIT, commit 19118807) with the NVFP4 Flash Next reader, on NVIDIA's PyTorch container (arm64).
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

LABEL org.opencontainers.image.source="https://github.com/ashhart/TensorFold" \
      org.opencontainers.image.revision="191188075bca56a7c71074a79375eb4c1cb22e1c" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.recipe.source="https://github.com/tournierjc/Swift-1.5-TensorFold-Single-DGX-Spark/tree/b695b88f786977428246dcc3e0d97293caca2c34"

# NGC exports PIP_CONSTRAINT for its own stack; the hash-pinned lock below is resolved without it and leaves torch and numpy alone.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

COPY vendor/tensorfold-191188075bca56a7c71074a79375eb4c1cb22e1c.tar.gz /tmp/tensorfold.tar.gz
COPY requirements.lock /tmp/requirements.lock
RUN echo "d2e8665e2adb80d8d98fdd6b5d4c8923e562732b0c6f388ca56a559956979ddf  /tmp/tensorfold.tar.gz" | sha256sum -c - \
    && mkdir /tmp/tensorfold-src \
    && tar -xzf /tmp/tensorfold.tar.gz -C /tmp/tensorfold-src --strip-components=1 \
    && PIP_CONSTRAINT= python -m pip install --ignore-installed --no-deps --no-cache-dir --require-hashes --requirement /tmp/requirements.lock \
    && python -c "import setuptools; assert int(setuptools.__version__.split('.')[0]) >= 69, setuptools.__version__" \
    && PIP_CONSTRAINT= python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/tensorfold-src \
    && python -c "import tensorfold.cuda.server; from tensorfold.families.qwen4_exp.cuda import nvfp4, nvfp4_moe" \
    && tensorfold --version \
    && rm -rf /tmp/tensorfold.tar.gz /tmp/tensorfold-src /tmp/requirements.lock

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
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
