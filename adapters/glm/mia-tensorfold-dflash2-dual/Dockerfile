# TensorFold v0.6.0 (Apache-2.0, commit c4646171) plus MiaAI-Lab's 70-patch GLM-5.3-Flash EXL3 set on NVIDIA's PyTorch container (arm64).
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

LABEL org.opencontainers.image.source="https://github.com/ashhart/TensorFold" \
      org.opencontainers.image.revision="c4646171139ee8a3c38103eaa1699dad226ec12b" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.recipe-patches.source="https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/tree/a15282478bd0e58ea80c512c330df3f61af85f74"

# NGC exports PIP_CONSTRAINT for its own stack; the hash-pinned lock below is resolved without it and leaves torch and numpy alone.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

COPY vendor/tensorfold-c4646171139ee8a3c38103eaa1699dad226ec12b.tar.gz /tmp/tensorfold.tar.gz
COPY requirements.lock /tmp/requirements.lock
COPY patches /opt/tf-patches
# Mia's patches are unified diffs against the installed package (paths such as tensorfold/cuda/server.py), applied with -p0 in filename order.
RUN echo "bfaec2d02944f8ddf576ef0fed67f5cca23dbc252caeb0b558154a968d001641  /tmp/tensorfold.tar.gz" | sha256sum -c - \
    && mkdir /tmp/tensorfold-src \
    && tar -xzf /tmp/tensorfold.tar.gz -C /tmp/tensorfold-src --strip-components=1 \
    && PIP_CONSTRAINT= python -m pip install --ignore-installed --no-deps --no-cache-dir --require-hashes --requirement /tmp/requirements.lock \
    && python -c "import setuptools; assert int(setuptools.__version__.split('.')[0]) >= 69, setuptools.__version__" \
    && PIP_CONSTRAINT= python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/tensorfold-src \
    && cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" \
    && for p in /opt/tf-patches/*.patch; do echo "applying $p"; patch --batch --fuzz=0 --forward -p0 < "$p" || exit 1; done \
    && python -c "import tensorfold.cuda.server, tensorfold.families.glm5_next.cuda.engine, tensorfold.vision.glm, av, xgrammar" \
    && tensorfold --version \
    && rm -rf /tmp/tensorfold.tar.gz /tmp/tensorfold-src /tmp/requirements.lock

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
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
