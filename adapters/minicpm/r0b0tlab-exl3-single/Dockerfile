FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/minicpm-v47-exl3-exllamav3-gb10" \
      org.opencontainers.image.revision="b8859be8b257ddb7602f68ce7b85a7f0fd03f7b4" \
      io.vonk.exllamav3.source="https://github.com/r0b0tlab/exllamav3-r0b0t" \
      io.vonk.exllamav3.revision="5ddae7d124482aa9e705c3a4aedf02cf8d09911b"
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST=12.1 CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} MAX_JOBS=2
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates build-essential python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 --require-hashes --requirement /tmp/requirements.lock
COPY vendor/exllamav3.tar.gz /tmp/exllamav3.tar.gz
RUN echo "b2ba24077400d8b310d6638d3a5a8ccd132d3af4004b86a18c78d0f5ee8d96a8  /tmp/exllamav3.tar.gz" | sha256sum -c - \
    && mkdir /tmp/exllamav3 \
    && tar -xzf /tmp/exllamav3.tar.gz -C /tmp/exllamav3 --strip-components=1 \
    && pip install --no-cache-dir --no-deps --no-build-isolation /tmp/exllamav3 \
    && rm -rf /tmp/exllamav3 /tmp/exllamav3.tar.gz /tmp/requirements.lock
COPY scripts /opt/vonk/kit/scripts
COPY config /opt/vonk/kit/config
COPY --chmod=0755 exllamav3-serve /opt/vonk/bin/exllamav3-serve
COPY licenses /opt/vonk/licenses
RUN python -c "import exllamav3; from exllamav3 import Model" \
    && groupadd --gid 10001 vonk \
    && useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
