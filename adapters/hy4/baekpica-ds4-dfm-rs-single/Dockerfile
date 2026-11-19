# Baekpica/ds4-dfm-rs (Rust-host continuation of antirez ds4) for the Hy4 flash MQ-Q5-Q8 GGUF on DGX Spark.
#
# The creator's card (Baekpica/Hy4-flash-Mixed-Quant-GGUF @ 7c44df46, "Serving on DGX Spark") says to build
# the engine from commit 25bf4b8ff712b1ad5e4298b9e87ddfa031881ac1 (upstream PR #159, merged).
# That commit is pinned by full sha and by the sha256 of its codeload archive; the build fails closed if
# either differs. Build follows the commit's docker/Dockerfile: CUDA 13.0 devel, rustup with the toolchain of
# rust-toolchain.toml (1.100.0), then `make cuda-spark` (sm_121). The kit declares nothing machine-readable,
# so the pins live here and in the Model.
ARG DS4_COMMIT=25bf4b8ff712b1ad5e4298b9e87ddfa031881ac1
ARG DS4_ARCHIVE_SHA256=9f3052b793c09a38eb1fbc8730f7546682d372de3a6a45651cf2e23ce1b754f7

FROM docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:38c2598ded0fd3ed2b0b13351543fde493faba43ae580f81f0a9ab58d76fdd2d AS build
ARG DS4_COMMIT
ARG DS4_ARCHIVE_SHA256
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/src
RUN test "$(uname -m)" = aarch64 \
    && curl -fsSL "https://codeload.github.com/Baekpica/ds4-dfm-rs/tar.gz/${DS4_COMMIT}" -o ds4.tar.gz \
    && echo "${DS4_ARCHIVE_SHA256}  ds4.tar.gz" | sha256sum -c - \
    && mkdir ds4 \
    && tar -xzf ds4.tar.gz -C ds4 --strip-components=1 \
    && rm ds4.tar.gz
WORKDIR /opt/src/ds4
ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --default-toolchain none \
    && rm /tmp/rustup-init.sh \
    && rustup show active-toolchain
RUN make -j"$(nproc)" cuda-spark \
    && test -x ds4-server

FROM docker.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:532381a7bf759bbe4d9c783468bc2cc4766cf656da5814fb04e049dc58fe268b
ARG DS4_COMMIT
ARG DS4_ARCHIVE_SHA256

LABEL org.opencontainers.image.source="https://github.com/Baekpica/ds4-dfm-rs" \
      org.opencontainers.image.revision="${DS4_COMMIT}" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.source-archive-sha256="${DS4_ARCHIVE_SHA256}"

RUN groupadd --gid 10001 vonk \
    && useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home \
        --shell /usr/sbin/nologin vonk \
    && install --directory --owner 10001 --group 10001 /opt/vonk/bin /state
COPY --from=build --chmod=0755 /opt/src/ds4/ds4-server /opt/vonk/bin/ds4-serve
COPY --from=build /opt/src/ds4/LICENSE /opt/vonk/LICENSE.ds4

ENV HOME=/state \
    HF_HUB_OFFLINE=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
