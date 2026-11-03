# Tonyd2wild bundled multi-stream TensorFold engine; upstream local image is not published.
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/GLM-5.3-Flash-TensorFold-DGX-Spark" \
      org.opencontainers.image.revision="1758bc2dab8ae56bf9592b74b312b398577850cc" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.recipe.source="https://github.com/tonyd2wild/GLM-5.3-Flash-TensorFold-DGX-Spark/tree/1758bc2dab8ae56bf9592b74b312b398577850cc"

# Install the exact bundled engine and hash-locked Python dependencies on the pinned CUDA base.
# engine-multi/pyproject.toml's package-data for tensorfold.cuda lists *.cu, *.cuh and *.cpp but not *.c, so the
# install drops roce_proxy.c, which TF_GLM_COMM=roce compiles at start; the creator runs from the source tree.
# Copy it in and build the proxy once here to prove the source, gcc and libibverbs are all in the image.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

COPY vendor/creator-1758bc2dab8ae56bf9592b74b312b398577850cc.tar.gz /tmp/tensorfold.tar.gz
COPY requirements.lock /tmp/requirements.lock
RUN echo "15e417b5f06136f86c53455214455bdda5d17e9147b0f19a40fb9c8da2c276ca  /tmp/tensorfold.tar.gz" | sha256sum -c - \
    && mkdir /tmp/tensorfold-src \
    && tar -xzf /tmp/tensorfold.tar.gz -C /tmp/tensorfold-src --strip-components=1 \
    && PIP_CONSTRAINT= python -m pip install --ignore-installed --no-deps --no-cache-dir --require-hashes --requirement /tmp/requirements.lock \
    && python -c "import setuptools; assert int(setuptools.__version__.split('.')[0]) >= 69, setuptools.__version__" \
    && PIP_CONSTRAINT= python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/tensorfold-src/engine-multi \
    && cp /tmp/tensorfold-src/engine-multi/src/tensorfold/cuda/roce_proxy.c "$(python -c 'import os, tensorfold.cuda as m; print(os.path.dirname(m.__file__))')/" \
    && python -c "import tensorfold.cuda.server, tensorfold.families.glm5_next.cuda.engine" \
    && TORCH_EXTENSIONS_DIR=/tmp/roce-smoke python -c "from tensorfold.cuda.roce import proxy_library; proxy_library()" \
    && rm -rf /tmp/roce-smoke \
    && tensorfold --version \
    && rm -rf /tmp/tensorfold.tar.gz /tmp/tensorfold-src

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# First start JIT-compiles the CUDA kernels for GB10 (and the RoCE all-gather) into the platform's writable /outputs/cache.
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
