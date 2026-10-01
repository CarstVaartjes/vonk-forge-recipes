# Qwen3.8-Flash-Next EXL3 on one DGX Spark (GB10 / SM121): TabbyAPI on the
# vcruz305/exllamav3 fork runtime.
#
# Upstream (vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe, exllamav3-tabby/)
# ships no container: setup.sh builds a venv on the host, installs torch 2.13.0
# from the cu130 index, compiles the fork's CUDA extension for sm_121 and
# installs the latest TabbyAPI main. This image performs the same installation
# on a digest-pinned CUDA 13.4.1 devel base, with the fork and TabbyAPI pinned
# by commit (upstream tracks TabbyAPI main). The upstream launch scripts are not
# run; tabby-serve replaces serve.sh and renders the vendored tabby-config.yml
# the same way.
FROM nvcr.io/nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:d44d6dc249c2c8330d0b380c595a514f0e86137fcc717fe7a4389a25720ade37

LABEL org.opencontainers.image.source="https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe" \
      org.opencontainers.image.revision="6fbc0a2f1a0a8ee3d94830bb607ec36aa370ea49" \
      org.opencontainers.image.licenses="MIT AND AGPL-3.0-only" \
      io.vonk.exllamav3.source="https://github.com/vcruz305/exllamav3" \
      io.vonk.exllamav3.revision="94ba01d50a13fa9ff672473f2d0eef8b51a71e99" \
      io.vonk.tabbyapi.source="https://github.com/theroyallab/tabbyAPI" \
      io.vonk.tabbyapi.revision="be74bf0a00bcb3a518e6feb7606f150c189be637"

ARG EXLLAMAV3_COMMIT=94ba01d50a13fa9ff672473f2d0eef8b51a71e99
ARG TABBYAPI_COMMIT=be74bf0a00bcb3a518e6feb7606f150c189be637
ARG TORCH_SPEC=torch==2.13.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=12.1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl build-essential ninja-build python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

# setup.sh steps 1-2: venv, build tools, torch with CUDA, and Triton (the fork's
# gated-delta-net kernels are Triton and the aarch64 torch wheel does not bring it).
RUN test "$(uname -m)" = aarch64 \
    && python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip setuptools wheel ninja packaging \
    && pip install --no-cache-dir "${TORCH_SPEC}" --index-url "${TORCH_INDEX_URL}" \
    && pip install --no-cache-dir triton

# setup.sh step 3: the fork at the pinned commit, compiled for sm_121 (about 15
# minutes), built from the commit tree rather than a clone.
RUN set -eux; \
    mkdir -p /tmp/exllamav3; \
    curl -fsSL "https://codeload.github.com/vcruz305/exllamav3/tar.gz/${EXLLAMAV3_COMMIT}" \
      | tar -xz -C /tmp/exllamav3 --strip-components=1; \
    cd /tmp/exllamav3; \
    MAX_JOBS="$(nproc)" pip install --no-cache-dir --no-build-isolation .; \
    rm -rf /tmp/exllamav3

# setup.sh step 4: TabbyAPI without its GPU extras (they pull stock exllamav3
# and torch wheels), plus uvloop, which main.py imports but its pyproject skips
# on aarch64. A stock exllamav3 pulled in by TabbyAPI would shadow the fork.
RUN set -eux; \
    mkdir -p /opt/tabbyAPI; \
    curl -fsSL "https://codeload.github.com/theroyallab/tabbyAPI/tar.gz/${TABBYAPI_COMMIT}" \
      | tar -xz -C /opt/tabbyAPI --strip-components=1; \
    cd /opt/tabbyAPI; \
    pip install --no-cache-dir .; \
    pip install --no-cache-dir uvloop; \
    pip show exllamav3 | grep -q '^Version:'

COPY tabby-config.yml /opt/vonk/tabby-config.yml
COPY --chmod=0755 tabby-serve /opt/vonk/bin/exllamav3-serve
COPY verify-runtime.py /opt/vonk/verify-runtime.py

RUN python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache/xdg \
    XDG_CONFIG_HOME=/outputs/cache/config \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_HOME=/outputs/cache/torch \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    CUDA_CACHE_PATH=/outputs/cache/cuda

WORKDIR /opt/tabbyAPI
USER 10001:10001
ENTRYPOINT []
