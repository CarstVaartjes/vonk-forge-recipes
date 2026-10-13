# tournierjc/TensorFold branch integration/0.6.4 at 53ceb126 (Apache-2.0; TensorFold v0.6.4 plus the rig's changes, among them the 8-bit decode faces and the 80,014-id MTP draft vocabulary) with the NVFP4 Flash Next reader and vision, on NVIDIA's PyTorch container (arm64).
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

LABEL org.opencontainers.image.source="https://github.com/tournierjc/TensorFold" \
      org.opencontainers.image.revision="53ceb126d5d54e9193245bcc74130fc1d81cf46f" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.recipe.source="https://github.com/tournierjc/Swift-1.5-TensorFold-Single-DGX-Spark/tree/70c561394e6e955e369b135b5edfd70c1dfe9efb"

# NGC exports PIP_CONSTRAINT for its own stack; the hash-pinned lock below is resolved without it and leaves torch and numpy alone.
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1

COPY vendor/tensorfold-53ceb126d5d54e9193245bcc74130fc1d81cf46f.tar.gz /tmp/tensorfold.tar.gz
COPY requirements.lock /tmp/requirements.lock
RUN echo "79b670c2f3a2a2063eeb7d3659a266146b8e95d0b1493c47eb74b60bcf4770b5  /tmp/tensorfold.tar.gz" | sha256sum -c - \
    && mkdir /tmp/tensorfold-src \
    && tar -xzf /tmp/tensorfold.tar.gz -C /tmp/tensorfold-src --strip-components=1 \
    && PIP_CONSTRAINT= python -m pip install --ignore-installed --no-deps --no-cache-dir --require-hashes --requirement /tmp/requirements.lock \
    && python -c "import setuptools; assert int(setuptools.__version__.split('.')[0]) >= 69, setuptools.__version__" \
    && PIP_CONSTRAINT= python -m pip install --no-cache-dir --no-deps --no-build-isolation /tmp/tensorfold-src \
    && python -c "import tensorfold.cuda.server; from tensorfold.families.qwen4_exp.cuda import nvfp4, nvfp4_moe" \
    && python -c "from tensorfold.vision.videos import load_videos; from tensorfold.families.qwen4_exp.cuda.bf16 import faces_8bit; from tensorfold.vision.qwen_cuda import QwenCudaVision; from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel; from PIL import Image; import av" \
    && python -c "import os, pathlib, tensorfold; p = pathlib.Path(os.path.dirname(tensorfold.__file__), 'families/qwen4_exp/cuda/draft_vocab.txt'); ids = [int(x) for x in p.read_text().split()]; assert ids == sorted(set(ids)) and len(ids) == 80014, len(ids)" \
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
