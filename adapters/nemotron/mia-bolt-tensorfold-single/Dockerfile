# Native Bolt runtime: the creator's exact engine, built from the kit's own recipe (scripts/prepare.sh at the pinned
# kit commit) because the creator has published no image for kit 6b064a31 (TensorFold eea9dc9c, tag
# zig-eea9dc9-4425be0c2259 is not on GHCR). Same TensorFold commit, patches, libbolt sources, Zig 0.17.0, CUDA
# toolkit and GB10 arch list (-Dsm default 121) as the creator's build; the thin Controller wrapper is on top.
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3 AS build
RUN curl -fsSL https://codeload.github.com/ashhart/TensorFold/tar.gz/eea9dc9c9c8b459c0a03de7f6fe0c6f4835d69ab -o /tmp/tensorfold.tar.gz \
 && echo "92bf97b0e74eb10b5fa061d9566dface8f1cad85e16ddaddcb5c632935e7792f  /tmp/tensorfold.tar.gz" | sha256sum -c - \
 && mkdir /tensorfold && tar -xzf /tmp/tensorfold.tar.gz --strip-components=1 -C /tensorfold && rm /tmp/tensorfold.tar.gz
RUN curl -fsSL -o /tmp/zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz \
 && echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz \
 && /opt/zig/zig version
# the decoders libtfvideo needs (FFmpeg 9.0.2, upstream's script) and the JPEG library the pictures need (libjpeg-turbo 3.1.4.1)
RUN cd /tensorfold && tools/zig/build_ffmpeg.sh /opt/ffmpeg >/tmp/ffmpeg.log 2>&1 \
 && curl -fsSL -o /tmp/jt.tgz https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz \
 && echo "ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022  /tmp/jt.tgz" | sha256sum -c - \
 && tar -xzf /tmp/jt.tgz -C /tmp && cd /tmp/libjpeg-turbo-3.1.4.1 \
 && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_STATIC=0 -DWITH_TURBOJPEG=1 -DCMAKE_INSTALL_PREFIX=/opt/jpeg >/dev/null \
 && cmake --build build -j"$(nproc)" >/dev/null && cmake --install build >/dev/null && rm -rf /tmp/jt.tgz /tmp/libjpeg-turbo-3.1.4.1
# structured output loads libtfgrammar, built from the xgrammar release the pinned tree names (grammar_library.zig);
# the tag is checked against its commit (v0.2.8 = 97787376faee5ed8466cfad57c99855e4ce2f6aa)
RUN tag="$(sed -n 's/^pub const pinned = "\(.*\)";/\1/p' /tensorfold/zig/src/platform/grammar_library.zig)" && test "$tag" = v0.2.8 \
 && git clone -q --branch "$tag" --recurse-submodules https://github.com/mlc-ai/xgrammar /opt/xgrammar \
 && test "$(git -C /opt/xgrammar rev-parse HEAD)" = 97787376faee5ed8466cfad57c99855e4ce2f6aa
COPY patches /opt/tf-patches
COPY engine /opt/bolt-engine
RUN cd /tensorfold \
 && for p in /opt/tf-patches/*.patch; do [ -e "$p" ] || continue; echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done \
 && cp /opt/bolt-engine/bolt_native.zig /opt/bolt-engine/bolt_api.zig zig/src/families/nemotron/
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dbolt-src=/opt/bolt-engine -Dffmpeg=/opt/ffmpeg \
      -Dxgrammar=/opt/xgrammar -Doptimize=fast --prefix /opt/tensorfold -j"$(nproc)" install native bolt video grammar fatbins \
 && cp -a /opt/jpeg/lib/libturbojpeg.so.0* /opt/tensorfold/native/lib/ \
 && test -x /opt/tensorfold/native/bin/tensorfold-native \
 && test -f /opt/tensorfold/native/lib/libbolt.so && test -f /opt/tensorfold/native/lib/libtfvideo.so \
 && test -f /opt/tensorfold/native/lib/libturbojpeg.so.0 && test -f /opt/tensorfold/native/lib/libtfgrammar.so
# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or sm_121a;
# every file with device code needs a loadable image; each file's listing is echoed). Covered: libbolt.so and the other
# shared libraries of the install, and tensorfold-native's kernels, which the Zig build embeds as fatbins
# (zig/build/cuda_runtime.zig): the `fatbins` install step of the same zig build above installs those exact images to
# /opt/tensorfold/fatbin, and every one is required to carry a loadable image (then the directory is removed again).
# Not covered: a runtime JIT of kernels the binaries do not embed.
COPY scan-sm121.py /opt/scan-sm121.py
RUN ls /opt/tensorfold/fatbin/*.fatbin >/dev/null \
 && python3 /opt/scan-sm121.py $(for f in /opt/tensorfold/fatbin/*.fatbin; do echo --require $f; done) \
      --require /opt/tensorfold/native/lib/libbolt.so \
      $(find /opt/tensorfold/native -type f \( -name '*.so*' -o -name tensorfold-native \) ! -name libbolt.so) \
 && rm -rf /opt/tensorfold/fatbin
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold \
 && for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done \
 && mkdir -p /opt/tensorfold/share/doc/tensorfold/LICENSES \
 && cp /opt/xgrammar/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-Apache-2.0.txt \
 && cp /opt/xgrammar/NOTICE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-NOTICE.txt \
 && cp /opt/xgrammar/3rdparty/dlpack/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES/dlpack-Apache-2.0.txt

FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Nemotron-3.5-Bolt-TensorFold-DGX-Spark" \
      org.opencontainers.image.revision="6b064a31dd47e4f959e9041d6129eb2bad830481" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1" \
      tf.gate=bolt
COPY --from=build /opt/tensorfold /opt/tensorfold
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN test -x "$(command -v tensorfold-native)" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# First start JIT-compiles the CUDA kernels for GB10 (and the RoCE all-gather) into the platform's writable /outputs/cache.
ENV LD_LIBRARY_PATH=/opt/tensorfold/native/lib \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
