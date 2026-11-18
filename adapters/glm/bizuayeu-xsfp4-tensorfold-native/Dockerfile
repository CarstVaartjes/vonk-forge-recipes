# Bizuayeu's GLM-5.3-Flash BIZ 3.x image (kit v3.15.0, 542aacc4; its image is 3.12.0's): TensorFold's native engine
# (tensorfold-native, upstream TensorFold v1.2.7 itself, commit 93a2e93e, cloned from Bizuayeu/TensorFold) compiled with Zig
# 0.17.0 and CUDA 13.3.1's nvcc for sm_121, libtfgrammar from xgrammar v0.2.8 (structured output), libtfvideo over a static
# FFmpeg 9.0.2 and the TurboJPEG/WebP decoders the engine checks by version (images and videos), the TP=3 wire's mailbox kernels,
# on the CUDA 13.3.1 runtime image with NCCL 2.30.7. The creator publishes no image; this is its v3/docker/Dockerfile with
# the changes the recipe build policy asks for: every FROM carries its digest, the engine pins live here instead of in
# v3/config/engine.lock.json (the same values), the kit's serve.sh/cluster.sh/tp*.env are replaced by tensorfold-serve (the
# Controller supplies rank, rendezvous and RoCE rails), the creator's host-side tokenizers/pip tools are left out, a
# build-time sm_121 kernel scan covers the files that carry device code, and the final user is numeric. One image serves
# both checkpoints and every TP size.
#
# The engine's zig build embeds its kernels as raw module bytes, so cuobjdump sees no device code in the binaries; the scan
# requires the mailbox fatbin to carry a loadable image and passes the binaries when they carry none.

# The media libraries, beside the engine's build: shared TurboJPEG and WebP decoder, and a static PIC FFmpeg with the
# engine's tools/zig/build_ffmpeg.sh configuration (decoders only, no GPL parts: LGPL-2.1), linked into libtfvideo below.
FROM docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87 AS media
ARG JPEG_TURBO_URL=https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz
ARG JPEG_TURBO_SHA256=ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022
ARG WEBP_URL=https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz
ARG WEBP_SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
ARG FFMPEG_URL=https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz
ARG FFMPEG_SHA256=8c3850283eb25fa026482078a04051e0be17347b09ef81a0849bec15a96e002e
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake make xz-utils curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp/media
RUN curl -fsSL -o jpeg.tar.gz "$JPEG_TURBO_URL" && echo "$JPEG_TURBO_SHA256  jpeg.tar.gz" | sha256sum -c - \
 && mkdir jpeg && tar -xzf jpeg.tar.gz -C jpeg --strip-components=1 \
 && cmake -S jpeg -B jpeg/b -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/jpeg -DENABLE_STATIC=OFF \
      -DWITH_TURBOJPEG=ON \
 && cmake --build jpeg/b -j"$(nproc)" && cmake --install jpeg/b
RUN curl -fsSL -o webp.tar.gz "$WEBP_URL" && echo "$WEBP_SHA256  webp.tar.gz" | sha256sum -c - \
 && mkdir webp && tar -xzf webp.tar.gz -C webp --strip-components=1 \
 && cd webp && ./configure --prefix=/opt/webp --disable-static --enable-libwebpdecoder --disable-libwebpmux \
      --disable-libwebpdemux --disable-png --disable-jpeg --disable-tiff --disable-gif --disable-wic \
 && make -j"$(nproc)" && make install
RUN curl -fsSL -o ffmpeg.tar.xz "$FFMPEG_URL" && echo "$FFMPEG_SHA256  ffmpeg.tar.xz" | sha256sum -c - \
 && mkdir ffmpeg && tar -xJf ffmpeg.tar.xz -C ffmpeg --strip-components=1 \
 && cd ffmpeg && ./configure --prefix=/opt/ffmpeg --enable-static --disable-shared --enable-pic \
      --disable-programs --disable-doc --disable-network --disable-autodetect --disable-everything \
      --disable-avdevice --disable-avfilter --disable-swresample \
      --enable-avcodec --enable-avformat --enable-swscale \
      --enable-decoder=h264,hevc,vp8,vp9,mpeg4 \
      --enable-parser=h264,hevc,vp8,vp9,mpeg4video \
      --enable-demuxer=mov,matroska \
      --enable-bsf=vp9_superframe_split \
 && make -j"$(nproc)" && make install
# The libraries the engine opens (libturbojpeg.so.0, libwebpdecoder.so.3) and the three projects' notices.
RUN mkdir -p /out/lib /out/LICENSES \
 && cp -P /opt/jpeg/lib*/libturbojpeg.so.0* /opt/webp/lib/libwebpdecoder.so.3* /out/lib/ \
 && cp jpeg/LICENSE.md /out/LICENSES/libjpeg-turbo-LICENSE.md \
 && cp jpeg/README.ijg /out/LICENSES/libjpeg-turbo-README.ijg \
 && cp webp/COPYING /out/LICENSES/libwebp-COPYING.txt && cp webp/PATENTS /out/LICENSES/libwebp-PATENTS.txt \
 && cp ffmpeg/COPYING.LGPLv2.1 /out/LICENSES/FFmpeg-LGPL-2.1.txt && cp ffmpeg/LICENSE.md /out/LICENSES/FFmpeg-LICENSE.md

FROM docker.io/nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87 AS build
# The engine and its compiler as the kit's v3/config/engine.lock.json pins them.
ARG ENGINE_REPO=https://github.com/Bizuayeu/TensorFold.git
ARG ENGINE_REF=93a2e93ecebff8aa52b88e58c5dd222516e69c5f
ARG ZIG_URL=https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz
ARG ZIG_SHA256=9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8
ARG BUILD_FLAGS="-Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121"
# Structured output's library (libtfgrammar: the engine's C ABI over xgrammar's C++), from the xgrammar release the
# engine pins, checked by commit.
ARG XGRAMMAR_REPO=https://github.com/mlc-ai/xgrammar.git
ARG XGRAMMAR_TAG=v0.2.8
ARG XGRAMMAR_REF=97787376faee5ed8466cfad57c99855e4ce2f6aa
RUN apt-get update \
 && apt-get install -y --no-install-recommends git xz-utils curl ca-certificates python3 \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL -o /tmp/zig.tar.xz "$ZIG_URL" \
 && echo "$ZIG_SHA256  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz
RUN git clone "$ENGINE_REPO" /src \
 && git -C /src checkout --detach "$ENGINE_REF" \
 && test "$(git -C /src rev-parse HEAD)" = "$ENGINE_REF"
# The plain build installs tf-cuda-test in zig-out/bin; `native` puts tensorfold-native in zig-out/native/bin.
WORKDIR /src
RUN /opt/zig/zig build $BUILD_FLAGS \
 && /opt/zig/zig build native $BUILD_FLAGS \
 && mkdir /out && cp zig-out/native/bin/tensorfold-native zig-out/bin/tf-cuda-test /out/ \
 && cd /out && printf "engine %s\nref %s\nzig %s\nflags %s\n" "$ENGINE_REPO" "$ENGINE_REF" "$ZIG_URL" "$BUILD_FLAGS" > ENGINE \
 && sha256sum tensorfold-native tf-cuda-test >> ENGINE
# `zig build grammar` puts libtfgrammar.so beside the native server, in zig-out/native/lib; xgrammar's and dlpack's
# notices go with it.
RUN git clone --branch "$XGRAMMAR_TAG" --recurse-submodules "$XGRAMMAR_REPO" /xgrammar \
 && test "$(git -C /xgrammar rev-parse HEAD)" = "$XGRAMMAR_REF" \
 && /opt/zig/zig build grammar -Dxgrammar=/xgrammar $BUILD_FLAGS \
 && cp zig-out/native/lib/libtfgrammar.so /out/ \
 && mkdir /out/LICENSES \
 && cp /xgrammar/LICENSE /out/LICENSES/xgrammar-Apache-2.0.txt \
 && cp /xgrammar/NOTICE /out/LICENSES/xgrammar-NOTICE.txt \
 && cp /xgrammar/3rdparty/dlpack/LICENSE /out/LICENSES/dlpack-Apache-2.0.txt \
 && cd /out && printf "xgrammar %s %s %s\n" "$XGRAMMAR_REPO" "$XGRAMMAR_TAG" "$XGRAMMAR_REF" >> ENGINE \
 && sha256sum libtfgrammar.so >> ENGINE
# Video input's libtfvideo (`zig build video` puts it in zig-out/native/lib) over the media stage's static FFmpeg; the
# media libraries' versions and every library's SHA-256 go into ENGINE.
ARG JPEG_TURBO_URL=https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz
ARG WEBP_URL=https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz
ARG FFMPEG_URL=https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz
COPY --from=media /opt/ffmpeg /opt/ffmpeg
COPY --from=media /out/lib/ /out/media/
RUN /opt/zig/zig build video -Dffmpeg=/opt/ffmpeg \
 && cp zig-out/native/lib/libtfvideo.so /out/media/ \
 && cd /out && printf "media %s\nmedia %s\nmedia %s\n" "$JPEG_TURBO_URL" "$WEBP_URL" "$FFMPEG_URL" >> ENGINE \
 && cd /out/media && sha256sum libtfvideo.so libturbojpeg.so.0.* libwebpdecoder.so.3.* >> /out/ENGINE
# The mailbox kernels of the wire at TP=3 (TF_GLM_ROCE_MULTI), which tensorfold-serve names in TF_TP_KERNEL_IMAGE.
RUN /usr/local/cuda/bin/nvcc -fatbin -O3 -std=c++17 -gencode=arch=compute_120,code=sm_120 \
      -gencode=arch=compute_121,code=sm_121 -gencode=arch=compute_90,code=compute_90 \
      -o /out/tp_mailbox.fatbin zig/src/fabric/tp/mailbox.cu \
 && cd /out && sha256sum tp_mailbox.fatbin >> ENGINE
# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or sm_121a;
# every --require file needs a loadable image; a file with no device code passes). The mailbox fatbin is required; the
# engine's binaries and libraries embed their kernels as module bytes and carry none of their own.
COPY scan-sm121.py /opt/scan-sm121.py
RUN python3 /opt/scan-sm121.py --require /out/tp_mailbox.fatbin \
      /out/tensorfold-native /out/tf-cuda-test /out/libtfgrammar.so /out/media/libtfvideo.so
# The engine's own notices, which its binaries carry (a step of its own, after the builds).
RUN mkdir /out/engine \
 && cp -r /src/LICENSE /src/NOTICE /src/THIRD_PARTY_NOTICES.md /src/LICENSES /out/engine/

FROM docker.io/nvidia/cuda:13.3.1-runtime-ubuntu24.04@sha256:63da350831208559df18c7b8f3e0d5d1c984eaa815cdde690aacd6606cd0cb11
LABEL org.opencontainers.image.source="https://github.com/Bizuayeu/GLM-5.3-Flash-DGX-Sparks-BIZ" \
      org.opencontainers.image.revision="542aacc4d4ff6b8e318dbced1ee19630776ea314" \
      org.opencontainers.image.licenses="Apache-2.0"
# NCCL at the version the creator's host pairs measured (2.30.7), held there; rdma-core's verbs and their providers so NCCL
# takes RoCE (NET/IB) rather than sockets; python3 for the Controller launcher. The creator's HTTP-check tools (pip,
# tokenizers) are not installed. The pin is not named NCCL_VERSION: the CUDA runtime image sets ENV NCCL_VERSION (its own
# bare 2.30.7-1), which overrides an ARG of that name inside RUN, so the apt pin lost its +cuda13.3 suffix and was not found.
ARG NCCL_PIN=2.30.7-1+cuda13.3
RUN apt-get update \
 && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages \
      "libnccl2=${NCCL_PIN}" libibverbs1 ibverbs-providers rdma-core ibverbs-utils \
      python3-minimal procps util-linux ca-certificates \
 && apt-mark hold libnccl2 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/tensorfold-native /out/tf-cuda-test /usr/local/bin/
COPY --from=build /out/ENGINE /usr/share/doc/tf3/ENGINE
# The server looks in ../lib beside its binary for the grammar library,
COPY --from=build /out/libtfgrammar.so /usr/local/lib/
# and in the loader's path for the media libraries (JPEG, WebP, video), which it opens at the first such input.
COPY --from=build /out/media/ /usr/local/lib/
RUN ldconfig
COPY --from=build /out/LICENSES/ /usr/share/doc/tf3/LICENSES/
COPY --from=media /out/LICENSES/ /usr/share/doc/tf3/LICENSES/
COPY --from=build /out/engine/ /usr/share/doc/tf3/engine/
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /usr/share/doc/tf3/
COPY LICENSES/ /usr/share/doc/tf3/LICENSES/
COPY --from=build /out/tp_mailbox.fatbin /opt/glm53-tf3/tp_mailbox.fatbin
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN /usr/local/bin/tensorfold-native --version \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
