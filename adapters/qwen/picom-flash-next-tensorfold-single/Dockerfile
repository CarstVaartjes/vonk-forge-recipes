# Picom-code's pico-dgx-qwen38 (Apache-2.0): Qwen3.8-Flash-Next on one DGX Spark, NVIDIA's FP4-Local checkpoint, bf16 KV cache, on TensorFold
# with MiaAI-Lab's patch set (patches 0001-0017) plus the creator's patches 0101-0129, both fetched at the commits in kit-lock.json.
# The creator publishes no image of the patch set (recipe.sh sets PULL=0 and builds locally), so this is the Dockerfile that
# MiaAI-Lab's scripts/prepare.sh builds, stage for stage (codecs, grammar, build, final), with the Controller's static source policy
# applied (digest-pinned base, literal https hosts, no heredoc). Every version and checksum comes from the kit's scripts/config.sh at
# the locked commit, read as data by kit-config; the build fails when the kit no longer has the lines it relies on. The base is the
# nvcr.io/nvidia/pytorch:26.07-py3 the creator pins in recipe.sh (its linux/arm64 manifest). A build-time sm_121 kernel scan follows.
# Thin layer on top: the Controller launcher wrapper, the two-stage draft head's inputs, the licence texts and the non-root user.
FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3 AS sources
COPY kit-lock.json /opt/vonk-build/kit-lock.json
COPY --chmod=0755 kit-config /opt/vonk-build/kit-config
RUN set -- $(python3 -c 'import json; d = json.load(open("/opt/vonk-build/kit-lock.json")); k, m, e = d["kit"], d["dependencies"]["mia-kit"], d["dependencies"]["engine"]; print(k["repo"], k["commit"], m["repo"], m["commit"], e["repo"], e["commit"])') \
 && fetch() { git init -q "$1" && git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3" \
      && git -C "$1" checkout -q --detach FETCH_HEAD && test "$(git -C "$1" rev-parse HEAD)" = "$3"; } \
 && fetch /picom "$1" "$2" && fetch /kit "$3" "$4" && fetch /tensorfold "$5" "$6" \
 && cp /kit/scripts/config.sh /opt/vonk-build/config.sh \
 && test "$(/opt/vonk-build/kit-config TF_REF)" = "$6" \
 && mkdir -p /opt/tf-patches && cp /kit/patches/*.patch /picom/tensorfold/patches/*.patch /opt/tf-patches/

FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3 AS codecs
COPY --from=sources /opt/vonk-build/config.sh /opt/vonk-build/config.sh
COPY --from=sources /opt/vonk-build/kit-config /opt/vonk-build/kit-config
RUN mkdir -p /src /opt/codecs && cd /src \
 && v=$(/opt/vonk-build/kit-config LIBJPEG_TURBO_VERSION) && sha=$(/opt/vonk-build/kit-config LIBJPEG_TURBO_SHA256) \
 && curl -fsSL -o jt.tar.gz "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/$v/libjpeg-turbo-$v.tar.gz" \
 && echo "$sha  jt.tar.gz" | sha256sum -c - \
 && tar -xzf jt.tar.gz && cmake -S libjpeg-turbo-$v -B jt-build -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON -DCMAKE_INSTALL_PREFIX=/opt/codecs/jpeg -DCMAKE_INSTALL_LIBDIR=lib \
 && cmake --build jt-build -j"$(nproc)" && cmake --install jt-build \
 && test -e /opt/codecs/jpeg/lib/libturbojpeg.so.0
RUN cd /src \
 && v=$(/opt/vonk-build/kit-config LIBWEBP_VERSION) && sha=$(/opt/vonk-build/kit-config LIBWEBP_SHA256) \
 && curl -fsSL -o webp.tar.gz "https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-$v.tar.gz" \
 && echo "$sha  webp.tar.gz" | sha256sum -c - \
 && tar -xzf webp.tar.gz && cd libwebp-$v \
 && ./configure --prefix=/opt/codecs/webp --enable-libwebpdecoder --disable-static --disable-gl --disable-sdl --disable-png --disable-jpeg --disable-tiff --disable-gif --disable-wic \
 && make -j"$(nproc)" && make install \
 && test -e /opt/codecs/webp/lib/libwebpdecoder.so.3

FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3 AS grammar
COPY --from=sources /opt/vonk-build/config.sh /opt/vonk-build/config.sh
COPY --from=sources /opt/vonk-build/kit-config /opt/vonk-build/kit-config
RUN mkdir -p /opt/tfrel && cd /tmp \
 && v=$(/opt/vonk-build/kit-config TF_RELEASE_VERSION) && sha=$(/opt/vonk-build/kit-config TF_RELEASE_AARCH64_SHA256) \
 && curl -fsSL -o tf-rel.tar.gz "https://github.com/ashhart/TensorFold/releases/download/v$v/tensorfold-$v-linux-aarch64.tar.gz" \
 && echo "$sha  tf-rel.tar.gz" | sha256sum -c - \
 && tar -xzf tf-rel.tar.gz -C /opt/tfrel --strip-components=1 \
      tensorfold-$v-linux-aarch64/lib/libtfgrammar.so \
      tensorfold-$v-linux-aarch64/LICENSES/xgrammar-Apache-2.0.txt \
      tensorfold-$v-linux-aarch64/LICENSES/xgrammar-NOTICE.txt \
      tensorfold-$v-linux-aarch64/LICENSES/dlpack-Apache-2.0.txt \
 && test -s /opt/tfrel/lib/libtfgrammar.so

FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3 AS build
COPY --from=sources /opt/vonk-build/config.sh /opt/vonk-build/config.sh
COPY --from=sources /opt/vonk-build/kit-config /opt/vonk-build/kit-config
RUN v=$(/opt/vonk-build/kit-config ZIG_VERSION) && sha=$(/opt/vonk-build/kit-config ZIG_SHA256) \
 && curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/$v/zig-aarch64-linux-$v.tar.xz" \
 && echo "$sha  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz \
 && test "$(/opt/zig/zig version)" = "$v"
COPY --from=sources /tensorfold /tensorfold
COPY --from=sources /opt/tf-patches /opt/tf-patches
# The kit's patches 0001-0017, then the creator's 0101-0129, in file name order, applied in the engine's root (prepare.sh).
RUN cd /tensorfold \
 && for p in /opt/tf-patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
# fatbins + native only (prepare.sh: the default install also builds tf-cuda-test, which does not compile on 1.1.4)
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins native \
 && test -x /opt/tensorfold/native/bin/tensorfold-native
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold \
 && for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done
# Build-time sm_121 kernel scan (scan-sm121.py; rules in docs/recipe-authoring.md). Every kernel image the engine embeds (the `fatbins`
# install of this same build) must carry a loadable image, and so must any device code in the server binary.
COPY scan-sm121.py /opt/vonk-build/scan-sm121.py
RUN ls /opt/tensorfold/fatbin/*.fatbin >/dev/null \
 && python3 /opt/vonk-build/scan-sm121.py $(for f in /opt/tensorfold/fatbin/*.fatbin; do echo --require $f; done) \
      /opt/tensorfold/native/bin/tensorfold-native \
 && rm -rf /opt/tensorfold/fatbin

FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/Picom-code/pico-dgx-qwen38" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      ai.vonkforge.runtime-interface="v1" \
      tf.kernels="embedded"
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=codecs /opt/codecs/jpeg/lib/libturbojpeg.so* /opt/tensorfold/lib/
COPY --from=codecs /opt/codecs/webp/lib/libwebpdecoder.so* /opt/tensorfold/lib/
# the engine looks for libtfgrammar.so in <binary dir>/../lib, i.e. next to native/bin
COPY --from=grammar /opt/tfrel/lib/libtfgrammar.so /opt/tensorfold/native/lib/libtfgrammar.so
COPY --from=grammar /opt/tfrel/LICENSES /opt/tensorfold/share/doc/tensorfold/LICENSES-grammar
# The creator's two-stage draft head (HEAD2, patch 0115): mkhead2.py, its draft vocabulary and its rotation from the creator's kit.
COPY --from=sources /picom/tensorfold/head2/mkhead2.py /picom/tensorfold/head2/draft_vocab_default.txt /picom/tensorfold/head2/head2_rot_r9.rot /opt/vonk/head2/
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve

# The engine's CUDA kernels are embedded in the image (tf.kernels=embedded); only the driver's compute cache is written, under the platform's /outputs/cache.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TMPDIR=/outputs/cache/tmp \
    TENSORFOLD_LIBTURBOJPEG=/opt/tensorfold/lib/libturbojpeg.so.0 \
    TENSORFOLD_LIBWEBP=/opt/tensorfold/lib/libwebpdecoder.so.3
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
