FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6

LABEL org.opencontainers.image.source="https://github.com/knapcio/MiniMax-H3-2x-DGX-Spark-TP2" \
      org.opencontainers.image.revision="2845006edea296b1e1f6f9d76a5a1c13a7c1e133" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.vllm-omni.revision="a8576ccb725c4e21cd13c3eb5f9a546b21149d2b" \
      io.vonk.vllm.version="0.30.0"

# The creator's stack, rebuilt from pinned sources (README "Second runtime: vLLM-Omni v0.30.0"):
# Python 3.12, vllm==0.30.0 from PyPI with the cu130 torch index, vLLM-Omni v0.30.0 and the
# creator's h3_tp2 package, both installed with --no-deps from sha256-checked tarballs.
# A devel base: Triton and the CuTe DSL JIT-compile kernels at run time and need nvcc-side
# headers, a host compiler and Python.h. CUDA 13.0 matches the cu130 wheels and the
# creator's driver generation (580).
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        build-essential \
        ca-certificates \
        curl \
        ibverbs-providers \
        libgl1 \
        libibverbs1 \
        libglib2.0-0t64 \
        libsm6 \
        libxext6 \
        python3 \
        python3-dev \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Every wheel is hash-locked. openai-whisper (a vLLM-Omni dependency) publishes only a source
# distribution: it is hash-locked too and built without build isolation, from the locked setuptools
# installed by the first pip run, so no unpinned build requirement is fetched.
COPY requirements.lock requirements-setuptools.lock /opt/vonk/build/
RUN python3 -m venv /opt/vonk/venv \
    && /opt/vonk/venv/bin/python -m pip install --no-cache-dir \
        --require-hashes \
        --no-deps \
        --only-binary=:all: \
        --requirement /opt/vonk/build/requirements-setuptools.lock \
    && /opt/vonk/venv/bin/python -m pip install --no-cache-dir \
        --require-hashes \
        --only-binary=:all: \
        --no-binary=openai-whisper \
        --no-build-isolation \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --requirement /opt/vonk/build/requirements.lock

RUN install --directory /opt/vonk/build/vllm-omni /opt/vonk/build/h3-tp2 \
    && curl --fail --silent --show-error --location \
        --output /opt/vonk/build/vllm-omni.tar.gz \
        https://codeload.github.com/vllm-project/vllm-omni/tar.gz/a8576ccb725c4e21cd13c3eb5f9a546b21149d2b \
    && echo "c56ba8f28cd40dbf621e1260dc9b626c360a64af156aa133279f5cf45da9c054  /opt/vonk/build/vllm-omni.tar.gz" | sha256sum --check - \
    && tar --extract --gzip --strip-components=1 --directory /opt/vonk/build/vllm-omni --file /opt/vonk/build/vllm-omni.tar.gz \
    && curl --fail --silent --show-error --location \
        --output /opt/vonk/build/h3-tp2.tar.gz \
        https://codeload.github.com/knapcio/MiniMax-H3-2x-DGX-Spark-TP2/tar.gz/2845006edea296b1e1f6f9d76a5a1c13a7c1e133 \
    && echo "9693a6df2d494245224fac93c9e2368aaca7214f5f8f6827021a5579836e3c2f  /opt/vonk/build/h3-tp2.tar.gz" | sha256sum --check - \
    && tar --extract --gzip --strip-components=1 --directory /opt/vonk/build/h3-tp2 --file /opt/vonk/build/h3-tp2.tar.gz

# The tarball has no git metadata for setuptools-scm: state the release version, as v0.30.0.
RUN VLLM_OMNI_VERSION_OVERRIDE=0.30.0 VLLM_OMNI_TARGET_DEVICE=cuda \
    /opt/vonk/venv/bin/python -m pip install --no-cache-dir --no-deps --no-build-isolation \
        /opt/vonk/build/vllm-omni \
        /opt/vonk/build/h3-tp2

COPY NOTICE LICENSE MODEL-LICENSE.md /opt/vonk/source/
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py
COPY --chmod=0755 vllm_omni_job.py /opt/vonk/bin/vllm-omni-job

RUN /opt/vonk/venv/bin/python /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && useradd --uid 10001 --create-home --shell /usr/sbin/nologin vonk

USER 10001:10001
ENTRYPOINT []
