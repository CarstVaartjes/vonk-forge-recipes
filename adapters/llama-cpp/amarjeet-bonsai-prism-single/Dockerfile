# PrismML llama.cpp fork (CUDA, sm_121a) for DGX Spark, built from the pinned
# commit behind the release prism-b10687-5d80cff on digest-pinned CUDA base images.
#
# Upstream (amarjeet/dgx-spark-recipes, Ternary-Bonsai-2-27B-GGUF-llamacpp-prism)
# is a native recipe: its build.sh clones the fork and compiles it on the host.
# This image performs the same build with the same CMake settings, without the
# embedded web UI. The ternary PQ2_0/PTQ1_0 types and the Hadamard activation
# transform exist only in the fork; stock llama.cpp loads the related Q2_0 band
# without an error and emits nonsense, so the source check below fails closed.
ARG PRISM_COMMIT=5d80cff0b8cb9f2bf823cfc4e71e3abb97f290d6

FROM nvcr.io/nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:d44d6dc249c2c8330d0b380c595a514f0e86137fcc717fe7a4389a25720ade37 AS build
ARG PRISM_COMMIT
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake curl ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/src
# Source by immutable commit (codeload serves the commit tree).
RUN test "$(uname -m)" = aarch64 \
    && curl -fsSL "https://codeload.github.com/PrismML-Eng/llama.cpp/tar.gz/${PRISM_COMMIT}" -o prism.tar.gz \
    && mkdir llama.cpp \
    && tar -xzf prism.tar.gz -C llama.cpp --strip-components=1 \
    && rm prism.tar.gz \
    && grep -q 'GGML_TYPE_PQ2_0' llama.cpp/ggml/include/ggml.h
# 121a-real: sm_120a is not forwards-compatible to GB10's sm_121 and carries no
# PTX to JIT from, so the fork's own scripts/build_cuda_linux.sh (which fixes
# the architecture list to 80;86;89;90;100;120a) would produce a binary that
# cannot run here. The build number is the fork tag's (prism-b10687).
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
        -DLLAMA_BUILD_NUMBER=10687 \
        -DLLAMA_BUILD_COMMIT=${PRISM_COMMIT} \
    && cmake --build build --target llama-server -j"$(nproc)" \
    && test -x build/bin/llama-server

FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962
ARG PRISM_COMMIT

LABEL org.opencontainers.image.source="https://github.com/PrismML-Eng/llama.cpp" \
      org.opencontainers.image.revision="${PRISM_COMMIT}" \
      org.opencontainers.image.version="prism-b10687-5d80cff" \
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
