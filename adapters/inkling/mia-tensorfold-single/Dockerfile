# Mia AI Lab's Inkling-2-Mini / one DGX Spark / TensorFold kit (MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold 9e3cf20b).
# The creator publishes no image: scripts/prepare.sh builds a local tag (tensorfold-inkling2:c4-dc2ca8a) with an inline Dockerfile.
# This file is that build, pinned: jschmied's Inkling 2 Mini CUDA engine (TensorFold PR #4441, commit dc2ca8a1, locked in
# kit-lock.json) plus the kit's patches 0001 to 0018 (vendored), Zig 0.17.0, libtfgrammar built from xgrammar v0.2.8 (commit
# checked), and the JPEG and WebP decoders for image input (libjpeg-turbo 3.1.4.1, libwebp 1.6.0), on the
# nvcr.io/nvidia/pytorch:26.07-py3 the kit builds on, with the kit's -Dnative-fp-decode default and a build-time sm_121 kernel
# scan after. The thin Controller layer on top is the launcher wrapper, the licence texts and the non-root user.
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3 AS codecs
# PNG is built into the engine; JPEG and WebP go through these two decoders, which the engine loads from <binary dir>/../lib.
RUN mkdir -p /opt/codecs/lib /opt/codecs/doc \
 && curl -fsSL --retry 3 -o /tmp/libjpeg-turbo.tar.gz https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz \
 && echo "ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022  /tmp/libjpeg-turbo.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/libjpeg-turbo.tar.gz -C /tmp \
 && cmake -S /tmp/libjpeg-turbo-3.1.4.1 -B /tmp/build-jpeg -DCMAKE_BUILD_TYPE=Release -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON \
 && cmake --build /tmp/build-jpeg --target turbojpeg -j"$(nproc)" \
 && cp -a /tmp/build-jpeg/libturbojpeg.so.0* /opt/codecs/lib/ \
 && cp /tmp/libjpeg-turbo-3.1.4.1/LICENSE.md /opt/codecs/doc/libjpeg-turbo-LICENSE.md \
 && curl -fsSL --retry 3 -o /tmp/libwebp.tar.gz https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz \
 && echo "e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564  /tmp/libwebp.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/libwebp.tar.gz -C /tmp \
 && cmake -S /tmp/libwebp-1.6.0 -B /tmp/build-webp -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF -DWEBP_BUILD_EXTRAS=OFF -DWEBP_BUILD_LIBWEBPMUX=OFF \
 && cmake --build /tmp/build-webp --target webpdecoder -j"$(nproc)" \
 && cp -a /tmp/build-webp/libwebpdecoder.so.3* /opt/codecs/lib/ \
 && cp /tmp/libwebp-1.6.0/COPYING /opt/codecs/doc/libwebp-BSD-3-Clause.txt \
 && rm -rf /tmp/libjpeg-turbo* /tmp/build-jpeg /tmp/libwebp* /tmp/build-webp

FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3 AS build
RUN curl -fsSL -o /tmp/zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz \
 && echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig \
 && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
 && rm /tmp/zig.tar.xz \
 && /opt/zig/zig version
RUN curl -fsSL https://codeload.github.com/jschmied/TensorFold/tar.gz/dc2ca8a1785df6ee5c8489ee7cd18e29cf7b4568 -o /tmp/tensorfold.tar.gz \
 && echo "ef8cd867e425dbbffca9283112e42f8a65363a52bd09ce4d402ed056dfc15665  /tmp/tensorfold.tar.gz" | sha256sum -c - \
 && mkdir /tensorfold \
 && tar -xzf /tmp/tensorfold.tar.gz --strip-components=1 -C /tensorfold \
 && rm /tmp/tensorfold.tar.gz
# patches/*.patch, in file name order, applied in the engine's root (scripts/prepare.sh).
COPY patches /opt/tf-patches
RUN cd /tensorfold \
 && for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
# fatbins and native only: the kernels are embedded in tensorfold-native (scripts/config.sh NATIVE_FP_DECODE=1: sm_121a's native FP4/FP8 decode)
RUN cd /tensorfold \
 && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast -Dnative-fp-decode=true --prefix /opt/tensorfold -j"$(nproc)" fatbins native \
 && test -x /opt/tensorfold/native/bin/tensorfold-native
# libtfgrammar.so (structured output, required tool calls) from the patched tree over xgrammar v0.2.8, checked by commit
# (97787376faee5ed8466cfad57c99855e4ce2f6aa); the engine finds it in <binary dir>/../lib, i.e. /opt/tensorfold/native/lib
RUN git clone -q --depth 1 --branch v0.2.8 --recurse-submodules --shallow-submodules https://github.com/mlc-ai/xgrammar /opt/xgrammar \
 && test "$(git -C /opt/xgrammar rev-parse HEAD)" = 97787376faee5ed8466cfad57c99855e4ce2f6aa \
 && cd /tensorfold \
 && /opt/zig/zig build grammar -Dxgrammar=/opt/xgrammar -Doptimize=ReleaseFast --prefix /opt/tensorfold -j"$(nproc)" \
 && test -s /opt/tensorfold/native/lib/libtfgrammar.so \
 && mkdir -p /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar \
 && cp /opt/xgrammar/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar/xgrammar-Apache-2.0.txt \
 && cp /opt/xgrammar/NOTICE /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar/xgrammar-NOTICE.txt \
 && cp /opt/xgrammar/3rdparty/dlpack/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar/dlpack-Apache-2.0.txt
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold \
 && for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done
# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or sm_121a;
# every file with device code needs a loadable image; each file's listing is echoed). Covered: tensorfold-native, whose kernels the
# Zig build embeds as fatbins (the `fatbins` install step above installs those exact images to /opt/tensorfold/fatbin, and every one
# is required to carry a loadable image), and the other shared libraries of the install. Not covered: a runtime JIT of kernels the
# binaries do not embed.
COPY scan-sm121.py /opt/scan-sm121.py
RUN ls /opt/tensorfold/fatbin/*.fatbin >/dev/null \
 && python3 /opt/scan-sm121.py $(for f in /opt/tensorfold/fatbin/*.fatbin; do echo --require $f; done) \
      $(find /opt/tensorfold/native -type f \( -name '*.so*' -o -name tensorfold-native \))

FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Inkling-2-Mini-DGX-Spark-TensorFold" \
      org.opencontainers.image.revision="6061fa89a34d2a245bc4a230f04948f312347a5e" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1" \
      tf.kernels=embedded
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=codecs /opt/codecs/lib/ /opt/tensorfold/native/lib/
COPY --from=codecs /opt/codecs/doc/ /opt/tensorfold/share/doc/tensorfold/LICENSES-codecs/
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN test -x "$(command -v tensorfold-native)" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# The engine's kernels are embedded in the binary; only the driver's compute cache is written, under the platform's /outputs/cache.
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
