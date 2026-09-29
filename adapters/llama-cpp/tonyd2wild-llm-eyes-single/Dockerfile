# llama.cpp (MIT) at ggml-org's b11243 (commit fc07d781, the revision of the ggml-org/llama.cpp server-cuda image
# published on 2026-09-29) built for GB10 on NVIDIA's PyTorch container (arm64, CUDA 13 toolchain).
# Upstream (tonyd2wild/2Wild-Vision-Mode) tells the user to build llama-server with CUDA on the Spark and does
# not pin it; the pin below replaces that step.
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/2Wild-Vision-Mode" \
      org.opencontainers.image.revision="c9ca903c760bbc83bd050bae7953d5f8dc030e86" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.recipe-engine.source="https://github.com/ggml-org/llama.cpp/tree/fc07d781e61f0d23764394e902b88d26a974e202"

RUN set -eu; \
    git init -q /tmp/llama.cpp; cd /tmp/llama.cpp; \
    git remote add origin https://github.com/ggml-org/llama.cpp; \
    git fetch -q --depth 1 origin fc07d781e61f0d23764394e902b88d26a974e202; \
    git checkout -q FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "fc07d781e61f0d23764394e902b88d26a974e202"; \
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real \
          -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF; \
    cmake --build build --target llama-server -j"$(nproc)"; \
    install -D build/bin/llama-server /opt/llama.cpp/bin/llama-server; \
    cd /; rm -rf /tmp/llama.cpp

COPY --chmod=0755 llama-server /opt/vonk/bin/llama-server
COPY verify-runtime.py /opt/vonk/verify-runtime.py
RUN python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

ENV HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    LLAMA_CACHE=/outputs/cache/llama.cpp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
