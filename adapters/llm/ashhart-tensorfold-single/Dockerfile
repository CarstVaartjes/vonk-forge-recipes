# TensorFold v1.2.7 (Apache-2.0): Ash Hart's native Zig engine as released, the linux-aarch64 archive
# installed as the creator's Linux instructions install it (README "Linux with an NVIDIA GPU": download, sha256sum -c,
# unpack, keep its directories together). CUDA fatbins (sm_80, sm_86, sm_120, sm_121) are embedded in the binary, so
# nothing is compiled at start; lib/ holds libtfgrammar for structured output and required tool calls. The NVIDIA
# driver (R580 or newer) supplies libcuda.so.1; the base supplies the CUDA 13 cuBLASLt that --vision loads.
# The archive ships no image decoders: as the creator's docs say (packaging/RUNTIME.md, docs/vision.md), JPEG needs a
# separately built libjpeg-turbo 3.1.4.1 (libturbojpeg) and WebP libwebp 1.6.0 (libwebpdecoder), each placed in lib/
# beside the server, where the engine finds them and checks their exact versions. The base's Ubuntu packages are other
# releases, so both are built here from their projects' release tarballs, pinned by sha256.
# This layer also adds the Controller launcher wrapper at the platform's path and the non-root user.
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

LABEL org.opencontainers.image.source="https://github.com/ashhart/TensorFold" \
      org.opencontainers.image.revision="93a2e93ecebff8aa52b88e58c5dd222516e69c5f" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

RUN curl -fsSL -o /tmp/tensorfold.tar.gz https://github.com/ashhart/TensorFold/releases/download/v1.2.7/tensorfold-1.2.7-linux-aarch64.tar.gz \
    && echo "250b7cd3d1aa2685cbf011f501c4f10bbff44c6ccd876dc63a7a4bbf337ed01b  /tmp/tensorfold.tar.gz" | sha256sum -c - \
    && mkdir /opt/tensorfold \
    && tar -xzf /tmp/tensorfold.tar.gz -C /opt/tensorfold --strip-components=1 \
    && rm /tmp/tensorfold.tar.gz \
    && test "$(cat /opt/tensorfold/VERSION)" = 1.2.7 \
    && /opt/tensorfold/bin/tensorfold-native --version

RUN curl -fsSL -o /tmp/jpeg.tar.gz https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz \
    && echo "ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022  /tmp/jpeg.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/jpeg.tar.gz -C /tmp && cd /tmp/libjpeg-turbo-3.1.4.1 \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_STATIC=0 -DWITH_TURBOJPEG=1 >/dev/null \
    && cmake --build build -j"$(nproc)" --target turbojpeg >/dev/null \
    && cp -P build/libturbojpeg.so.0* /opt/tensorfold/lib/ \
    && curl -fsSL -o /tmp/webp.tar.gz https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz \
    && echo "e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564  /tmp/webp.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/webp.tar.gz -C /tmp && cd /tmp/libwebp-1.6.0 \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF \
       -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF \
       -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF >/dev/null \
    && cmake --build build -j"$(nproc)" --target webpdecoder >/dev/null \
    && cp -P build/libwebpdecoder.so.3* /opt/tensorfold/lib/ \
    && cd / && rm -rf /tmp/jpeg.tar.gz /tmp/libjpeg-turbo-3.1.4.1 /tmp/webp.tar.gz /tmp/libwebp-1.6.0 \
    && caps="$(/opt/tensorfold/bin/tensorfold-native capabilities --json)" \
    && echo "$caps" | grep -q '"soname":"libcublasLt.so.13","found":true' \
    && echo "$caps" | grep -q '"soname":"libturbojpeg.so.0","found":true,[^}]*"usable":true,"problem":null' \
    && echo "$caps" | grep -q '"soname":"libwebpdecoder.so.3","found":true,[^}]*"usable":true,"problem":null'

COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# The engine writes its measured window costs under $HOME/.cache/tensorfold and the driver its compute cache, both under
# the platform's writable /outputs/cache.
ENV HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
