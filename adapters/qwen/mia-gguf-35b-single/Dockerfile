# llama.cpp llama-server (CUDA, sm_121a) for DGX Spark, built from a pinned
# llama.cpp commit (release tag b11261) on digest-pinned CUDA base images.
#
# The vendored upstream start.sh/stop.sh/README.md (MiaAI-Lab) are kept for
# provenance; they expect a host-built llama-server and are not run in the image.
# The recipe launches /opt/vonk/bin/llama-server with the same flags start.sh uses.
ARG LLAMA_CPP_COMMIT=b5cf8ce02aaf95c15375e2ed97f273fa7e274f5c
ARG UBUNTU_SNAPSHOT=20260929T000000Z

FROM nvcr.io/nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:d44d6dc249c2c8330d0b380c595a514f0e86137fcc717fe7a4389a25720ade37 AS build
ARG LLAMA_CPP_COMMIT
ARG UBUNTU_SNAPSHOT
# Resolve build tools only from the fixed Ubuntu ARM64 archive snapshot. CUDA
# packages already come from the digest-pinned NGC base, so exclude its live apt
# source while resolving this closure.
RUN test -f /etc/apt/sources.list.d/ubuntu.sources \
    && sed -i "s|^URIs: .*|URIs: https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/|" /etc/apt/sources.list.d/ubuntu.sources \
    && find /etc/apt/sources.list.d -type f \
        \( -name '*.list' -o -name '*.sources' \) ! -name ubuntu.sources -delete \
    && apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates=20260601~24.04.1 cmake=3.28.3-1build7 curl=8.5.0-2ubuntu10.15 ninja-build=1.11.1-2 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/src
# Source by immutable commit (codeload serves the commit tree). The pinned commit
# must know the MTP speculative type that upstream start.sh passes.
RUN test "$(uname -m)" = aarch64 \
    && curl -fsSL "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/${LLAMA_CPP_COMMIT}" -o llama.tar.gz \
    && mkdir llama.cpp \
    && tar -xzf llama.tar.gz -C llama.cpp --strip-components=1 \
    && rm llama.tar.gz \
    && grep -q '"draft-mtp"' llama.cpp/common/speculative.cpp
# No web UI (its assets come from a Hugging Face bucket at build time): the OpenAI-compatible API and /health are served headless.
RUN cmake -S llama.cpp -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=121a-real \
        -DGGML_NATIVE=OFF \
        -DGGML_OPENMP=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_NUMBER=11261 \
        -DLLAMA_BUILD_COMMIT=${LLAMA_CPP_COMMIT} \
    && cmake --build build --target llama-server -j"$(nproc)" \
    && test -x build/bin/llama-server

FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962
ARG LLAMA_CPP_COMMIT

LABEL org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      org.opencontainers.image.revision="${LLAMA_CPP_COMMIT}" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd --gid 10001 vonk \
    && useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home \
        --shell /usr/sbin/nologin vonk \
    && install --directory --owner 10001 --group 10001 /opt/vonk/bin /state
COPY --from=build --chmod=0755 /opt/src/build/bin/llama-server /opt/vonk/bin/llama-server
COPY --from=build /opt/src/llama.cpp/LICENSE /opt/vonk/LICENSE.llama.cpp

ENV HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
