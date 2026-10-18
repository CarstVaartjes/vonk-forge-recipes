FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
LABEL org.opencontainers.image.source="https://github.com/vcruz305/GLM-5.3-EXL3-DGX-Spark-recipe" \
      org.opencontainers.image.revision="ed2f31d65c00b1ebfc3f32039d1d37764fcf4e03"
ENV DEBIAN_FRONTEND=noninteractive CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1 \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} MAX_JOBS=2
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates build-essential libibverbs-dev python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 --require-hashes --requirement /tmp/requirements.lock
COPY vendor/tensorfold.tar.gz /tmp/tensorfold.tar.gz
RUN echo "10570c84b768c3a5cbe1cf5321a52286481b8786fab29d277ef80acae425b0c4  /tmp/tensorfold.tar.gz" | sha256sum -c - \
    && mkdir /tmp/tensorfold \
    && tar -xzf /tmp/tensorfold.tar.gz -C /tmp/tensorfold --strip-components=1 \
    && pip install --no-cache-dir --no-deps --no-build-isolation /tmp/tensorfold \
    && rm -rf /tmp/tensorfold /tmp/tensorfold.tar.gz
COPY vendor/b12x.tar.gz /tmp/b12x.tar.gz
RUN echo "8cfd2d8bf09169d00a669f72f179c54bdfc49af35b05214a5e5f0fdb1d5779d8  /tmp/b12x.tar.gz" | sha256sum -c - \
    && mkdir /tmp/b12x \
    && tar -xzf /tmp/b12x.tar.gz -C /tmp/b12x --strip-components=1 \
    && pip install --no-cache-dir --no-deps --no-build-isolation --target /opt/b12x/site /tmp/b12x \
    && rm -rf /tmp/b12x /tmp/b12x.tar.gz
COPY upstream /opt/vonk/runtime
COPY prepare_view.py /opt/vonk/runtime/prepare_view.py
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
COPY licenses /opt/vonk/licenses
ENV PYTHONPATH=/opt/b12x/site HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions TRITON_CACHE_DIR=/outputs/cache/triton
RUN python -c "import tensorfold, b12x; from tensorfold.families import glm_moe_dsa" \
    && groupadd --gid 10001 vonk \
    && useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
