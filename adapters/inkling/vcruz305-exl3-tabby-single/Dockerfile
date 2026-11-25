# Inkling 2 Mini SAGE REXL3 4.23 bpw on one DGX Spark (GB10 / SM121): TabbyAPI on the
# vcruz305/exllamav3 fork runtime.
#
# Upstream (vcruz305/Inkling-2-Mini-SAGE-EXL3-recipe, dgx-spark/) ships no
# container: setup.sh builds a venv on the host, installs torch 2.13.0 from the
# cu130 index with the CUDA 13.0 toolkit, installs the fork's requirements,
# builds the fork's CUDA extension for the card and installs vcruz305/tabbyAPI.
# This image performs the same installation on a digest-pinned CUDA 13.0.2
# devel base, with the engine and TabbyAPI forks locked to the commits the kit
# pins in scripts/env.sh. The extension is compiled for sm_121 at build time
# instead of at first import. The upstream launch scripts are not run;
# tabby-serve replaces serve.sh and renders the vendored tabby-config.yml the
# same way. The DFlash2 drafter is a separate model, mounted at /models/draft.
# Kit ed64091c (2026-11-25): engine c702887b adds the DFlash2 hookup.
FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6

LABEL org.opencontainers.image.source="https://github.com/vcruz305/Inkling-2-Mini-SAGE-EXL3-recipe" \
      org.opencontainers.image.revision="ed64091c4d5020b8617b0d7d18ce1296af2337c0" \
      org.opencontainers.image.licenses="MIT AND AGPL-3.0-only" \
      io.vonk.exllamav3.source="https://github.com/vcruz305/exllamav3" \
      io.vonk.exllamav3.revision="c702887ba7be368e1714339dd755a7c3c9b263d5" \
      io.vonk.tabbyapi.source="https://github.com/vcruz305/tabbyAPI" \
      io.vonk.tabbyapi.revision="a2898191eed710b55066d699ce829a976b4b3272"

ARG TORCH_SPEC=torch==2.13.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST=12.1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl build-essential ninja-build python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY kit-lock.json /opt/vonk/kit-lock.json

# setup.sh: venv, build tools, torch with CUDA.
RUN test "$(uname -m)" = aarch64 \
    && python3 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip wheel "setuptools<81" \
    && pip install --no-cache-dir "${TORCH_SPEC}" --index-url "${TORCH_INDEX_URL}"

# setup.sh: the fork at the pinned commit, built from the commit tree rather
# than a clone. Its requirements first, then the fork itself without
# dependencies, so nothing replaces torch; the extension is compiled for sm_121
# here (about 15 minutes).
RUN set -eux; \
    EXLLAMAV3_COMMIT="$(python3 -c 'import json; print(json.load(open("/opt/vonk/kit-lock.json"))["dependencies"]["engine"]["commit"])')"; \
    mkdir -p /tmp/exllamav3; \
    curl -fsSL "https://codeload.github.com/vcruz305/exllamav3/tar.gz/${EXLLAMAV3_COMMIT}" \
      | tar -xz -C /tmp/exllamav3 --strip-components=1; \
    cd /tmp/exllamav3; \
    pip install --no-cache-dir -r requirements.txt; \
    MAX_JOBS="$(nproc)" pip install --no-cache-dir --no-build-isolation --no-deps .; \
    rm -rf /tmp/exllamav3

# setup.sh: TabbyAPI base dependencies only. Its cu12 extra would replace torch
# and install a stock exllamav3, which has neither the architecture nor the
# draft head. uvloop is imported by main.py.
RUN set -eux; \
    TABBYAPI_COMMIT="$(python3 -c 'import json; print(json.load(open("/opt/vonk/kit-lock.json"))["dependencies"]["server"]["commit"])')"; \
    mkdir -p /opt/tabbyAPI; \
    curl -fsSL "https://codeload.github.com/vcruz305/tabbyAPI/tar.gz/${TABBYAPI_COMMIT}" \
      | tar -xz -C /opt/tabbyAPI --strip-components=1; \
    cd /opt/tabbyAPI; \
    pip install --no-cache-dir .; \
    pip install --no-cache-dir uvloop

COPY tabby-config.yml /opt/vonk/tabby-config.yml
COPY --chmod=0755 tabby-serve /opt/vonk/bin/exllamav3-serve
COPY verify-runtime.py /opt/vonk/verify-runtime.py
COPY LICENSE /opt/vonk/licenses/kit-LICENSE

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
