# TensorFold v0.3.7 (MIT) plus MiaAI-Lab's Flash Next patch set on NVIDIA's PyTorch container (arm64).
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

LABEL org.opencontainers.image.source="https://github.com/ashhart/TensorFold" \
      org.opencontainers.image.revision="02b726f3fb5e67723cb6f4e0315adf85e4b91033" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.recipe-patches.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold/tree/856bb6be4b58ce6a6727e6d071fb1c52f3f80e6e"

# NGC exports PIP_CONSTRAINT for its own stack; the hash-pinned lock below is resolved without it and leaves torch and numpy alone.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

COPY vendor/tensorfold-02b726f3fb5e67723cb6f4e0315adf85e4b91033.tar.gz /tmp/tensorfold.tar.gz
COPY requirements.lock /tmp/requirements.lock
COPY patches /opt/tf-patches
RUN echo "b97ac73797e82d1169e6325ea05808e2c0ddb30f3d3d5169202ad11a4720a7b8  /tmp/tensorfold.tar.gz" | sha256sum -c - \
    && mkdir /tmp/tensorfold-src \
    && tar -xzf /tmp/tensorfold.tar.gz -C /tmp/tensorfold-src --strip-components=1 \
    && PIP_CONSTRAINT= python -m pip install --no-cache-dir --require-hashes --requirement /tmp/requirements.lock \
    && python -c "import setuptools; assert int(setuptools.__version__.split('.')[0]) >= 69, setuptools.__version__" \
    && PIP_CONSTRAINT= python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/tensorfold-src \
    && cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" \
    && for p in /opt/tf-patches/*.patch; do echo "applying $p"; patch -p0 --forward < "$p" || exit 1; done \
    && python -c "import tensorfold.cuda.server, tensorfold.vision.qwen_cuda, tensorfold.vision.videos, av; from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel" \
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
