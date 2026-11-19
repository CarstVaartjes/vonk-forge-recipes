# llama.cpp with Tencent's Hy4 Flash Spark patch (CUDA 13.0, sm_121a) for DGX Spark.
#
# Upstream (AngelSlim/Hy4-flash-GGUF, README_SPARK-4Bit.md) builds ggml-org/llama.cpp
# at the commit below, applies patch/hy4-flash-spark.patch from the Hugging Face
# repository, and compiles llama-server for sm_121a. This image does the same with the
# creator's CMake flags, on the creator's CUDA 13.0 toolkit. The commit archive and
# the patch are revision-pinned and digest-checked; the build fails closed on a
# different byte. The creator applies the patch with git apply; a codeload archive is
# not a git checkout, so patch -p1 --fuzz=0 applies the same diff without fuzz.
ARG LLAMA_CPP_COMMIT=2cbca389f515df455412dafa07e2b1359cc72a6d
ARG LLAMA_CPP_SHA256=5ae980bfae6e215e4e953cf06d97f7849d8227c6bf80f28f360749ca018a519c
ARG HY4_REVISION=65acb8ae7f59847fe03a17291f1b72b89b141a9e
ARG HY4_PATCH_SHA256=6ec50722c8b181222a928e0a66cda0159e8dff3e72f8193334f6296f6213dfff

FROM docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:38c2598ded0fd3ed2b0b13351543fde493faba43ae580f81f0a9ab58d76fdd2d AS build
ARG LLAMA_CPP_COMMIT
ARG LLAMA_CPP_SHA256
ARG HY4_REVISION
ARG HY4_PATCH_SHA256
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake curl ninja-build patch \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/src
RUN test "$(uname -m)" = aarch64 \
    && curl -fsSL "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/${LLAMA_CPP_COMMIT}" -o llama.tar.gz \
    && echo "${LLAMA_CPP_SHA256}  llama.tar.gz" | sha256sum -c - \
    && mkdir llama.cpp \
    && tar -xzf llama.tar.gz -C llama.cpp --strip-components=1 \
    && rm llama.tar.gz \
    && curl -fsSL "https://huggingface.co/AngelSlim/Hy4-flash-GGUF/resolve/${HY4_REVISION}/patch/hy4-flash-spark.patch" -o hy4-flash-spark.patch \
    && echo "${HY4_PATCH_SHA256}  hy4-flash-spark.patch" | sha256sum -c - \
    && patch -d llama.cpp -p1 --fuzz=0 < hy4-flash-spark.patch \
    && rm hy4-flash-spark.patch
RUN cmake -S llama.cpp -B build -G Ninja \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=121a \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_UI=OFF \
        -DLLAMA_USE_PREBUILT_UI=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_COMMIT=${LLAMA_CPP_COMMIT} \
    && cmake --build build --target llama-server -j"$(nproc)" \
    && test -x build/bin/llama-server

FROM docker.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:532381a7bf759bbe4d9c783468bc2cc4766cf656da5814fb04e049dc58fe268b
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
