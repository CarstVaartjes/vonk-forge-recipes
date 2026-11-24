# ajensenwaud's InkSpark kit, Inkling-2-Small on two DGX Sparks with TensorFold (BSD-3-Clause; patches Apache-2.0):
# https://github.com/ajensenwaud/inkspark-2dgx/tree/1c15d6464ff0adb32d580b7b37106988a248cf1c
# The creator publishes no image. scripts/prepare.sh builds docker/Dockerfile.build (the kit's BASE_IMAGE, Mia's GLM-5.3
# TensorFold image by digest: NVIDIA PyTorch 26.07, CUDA 13.3, NCCL 2.30.7; plus Zig 0.17.0 and the libjpeg-turbo 3.1.4.1
# and libwebp 1.6.0 decoders the server loads for pictures), applies the kit's patches-zig/ 0001 to 0046 to TensorFold
# v1.2.7 and compiles the server and the family's check tools in that image for GB10 (sm_121). This file is that
# build, with the kit and engine commits from kit-lock.json (tools/kit-pins). The final stage is the same base with the
# build and the two decoders on the loader's path, as the kit runs it, plus the thin Controller layer: the launcher
# wrapper, the licence texts and the non-root user.
FROM ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529 AS codecs
# docker/Dockerfile.build's codecs stage: only the two shared libraries and their licences leave it.
WORKDIR /src
RUN curl -fsSL --retry 3 -o jpeg.tar.gz https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz \
 && curl -fsSL --retry 3 -o webp.tar.gz https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz \
 && printf '%s  jpeg.tar.gz\n%s  webp.tar.gz\n' ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022 e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564 | sha256sum -c - \
 && tar -xzf jpeg.tar.gz && tar -xzf webp.tar.gz \
 && cmake -S libjpeg-turbo-3.1.4.1 -B jpeg -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/codecs \
      -DCMAKE_INSTALL_LIBDIR=lib -DENABLE_STATIC=OFF -DWITH_TOOLS=OFF -DWITH_TESTS=OFF \
 && cmake --build jpeg -j"$(nproc)" && cmake --install jpeg \
 && cmake -S libwebp-1.6.0 -B webp -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/codecs \
      -DCMAKE_INSTALL_LIBDIR=lib -DBUILD_SHARED_LIBS=ON -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF \
      -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF \
      -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_LIBWEBPMUX=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF \
 && cmake --build webp -j"$(nproc)" && cmake --install webp \
 && mkdir -p /out/lib /out/share/doc/libjpeg-turbo /out/share/doc/libwebp \
 && cp -a /opt/codecs/lib/libturbojpeg.so.* /opt/codecs/lib/libwebpdecoder.so.* /out/lib/ \
 && cp libjpeg-turbo-3.1.4.1/LICENSE.md libjpeg-turbo-3.1.4.1/README.ijg /out/share/doc/libjpeg-turbo/ \
 && cp libwebp-1.6.0/COPYING libwebp-1.6.0/PATENTS /out/share/doc/libwebp/ \
 && grep -aq "libjpeg-turbo version 3.1.4.1 " /out/lib/libturbojpeg.so.0 \
 && test -e /out/lib/libwebpdecoder.so.3

FROM ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529 AS build
RUN curl -fsSL --retry 3 -o /tmp/zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz \
 && echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz \
 && test "$(/opt/zig/zig version)" = 0.17.0
# The kit and TensorFold commits come from kit-lock.json, each checked by commit after the fetch.
COPY kit-lock.json /opt/vonk-build/kit-lock.json
RUN set -- $(python3 -c 'import json; d = json.load(open("/opt/vonk-build/kit-lock.json")); k, e = d["kit"], d["dependencies"]["engine"]; print(k["repo"], k["commit"], e["repo"], e["commit"])') \
 && fetch() { git init -q "$1" && git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3" \
      && git -C "$1" checkout -q --detach FETCH_HEAD && test "$(git -C "$1" rev-parse HEAD)" = "$3"; } \
 && fetch /kit "$1" "$2" && fetch /tensorfold "$3" "$4"
# scripts/prepare.sh: patches-zig/*.patch in file name order on the pinned release, then the server and the family's
# tools for sm_121 with the image's nvcc, offline.
RUN cd /tensorfold \
 && for p in /kit/patches-zig/*.patch; do git apply --check "$p" && git apply "$p" || exit 1; done \
 && NVCC=$(command -v nvcc) && ZIG_GLOBAL_CACHE_DIR=/tmp/zig-cache /opt/zig/zig build native tf-inkling2-engine tf-inkling2-vision \
      -Dnvcc="$NVCC" -Dtp-nvcc="$NVCC" -Dsm=121 -Doptimize=ReleaseFast --prefix /opt/tensorfold \
 && for b in native/bin/tensorfold-native bin/tf-inkling2-engine bin/tf-inkling2-vision; do test -x "/opt/tensorfold/$b" || exit 1; done \
 && mkdir -p /opt/tensorfold/share/doc/tensorfold \
 && for f in LICENSE NOTICE LICENSES THIRD_PARTY_NOTICES.md; do if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done \
 && rm -rf /tmp/zig-cache

FROM ghcr.io/miaai-lab/glm-5.3-flash-exl3-2x-dgx-sparks-tensorfold@sha256:14f15591eae5d6a540f09218d3852068962fe5381371bbfefe0e9194cd834529
LABEL org.opencontainers.image.source="https://github.com/ajensenwaud/inkspark-2dgx" \
      org.opencontainers.image.revision="1c15d6464ff0adb32d580b7b37106988a248cf1c" \
      org.opencontainers.image.licenses="BSD-3-Clause AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"
# The server looks for libturbojpeg.so.0 and libwebpdecoder.so.3 beside its bin/, then on the loader's path (Dockerfile.build).
COPY --from=codecs /out/ /usr/local/
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN ldconfig && ldconfig -p | grep -E 'libturbojpeg\.so\.0 |libwebpdecoder\.so\.3 ' \
 && ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native \
 && ! ldd /opt/tensorfold/native/bin/tensorfold-native | grep -v -E 'libcuda|libnvidia' | grep "not found" \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
