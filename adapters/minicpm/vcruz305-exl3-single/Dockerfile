FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
LABEL org.opencontainers.image.source="https://github.com/vcruz305/MiniCPM-V-4.7-35B-A3B-EXL3-DGX-Spark-recipe" \
      org.opencontainers.image.revision="7d0bff70e089ad78f57764fe44889a033b18a948" \
      io.vonk.exllamav3.source="https://github.com/vcruz305/exllamav3" \
      io.vonk.exllamav3.revision="475461e9a1a288e7c10eb472056239d3fb78b625"
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST=12.1 CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} MAX_JOBS=2
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates build-essential python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 --require-hashes --requirement /tmp/requirements.lock
COPY vendor/exllamav3.tar.gz /tmp/exllamav3.tar.gz
RUN echo "2fee9bd3f5f5475c4fed1ba74c1820e217e7d4698e3e89025eb38d06d98bdc37  /tmp/exllamav3.tar.gz" | sha256sum -c - \
    && mkdir /tmp/exllamav3 \
    && tar -xzf /tmp/exllamav3.tar.gz -C /tmp/exllamav3 --strip-components=1 \
    && pip install --no-cache-dir --no-deps --no-build-isolation /tmp/exllamav3 \
    && rm -rf /tmp/exllamav3 /tmp/exllamav3.tar.gz /tmp/requirements.lock
COPY vendor/tabby.tar.gz /tmp/tabby.tar.gz
RUN echo "85d83013e319e49ffb97608ba3fff0d93b8357139b6f893a8e73023f49eaf74e  /tmp/tabby.tar.gz" | sha256sum -c - \
    && mkdir /opt/tabbyAPI \
    && tar -xzf /tmp/tabby.tar.gz -C /opt/tabbyAPI --strip-components=1 \
    && pip install --no-cache-dir --no-deps --no-build-isolation /opt/tabbyAPI \
    && rm /tmp/tabby.tar.gz
COPY tools /opt/vonk/kit/tools
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
