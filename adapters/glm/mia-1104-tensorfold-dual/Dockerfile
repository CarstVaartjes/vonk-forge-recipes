# MiaAI-Lab's GLM-5.3-Flash DGX Sparks kit 1104 (commit 02dac387, Apache-2.0): TensorFold's native Zig/CUDA server, upstream
# commit f350cd3e (1.1.9) plus the kit's 108 engine patches, built for sm_121 (GB10) on CUDA 13.3.1.
# The creator publishes no image of this kit (scripts/config.sh: IMAGE is empty, start.sh builds tensorfold-glm53-spark:<hash>
# locally), so this is the kit's own docker/Dockerfile, stage for stage, with the Controller's static source policy applied:
# digest-pinned bases, literal https hosts, no BuildKit cache mount, and the WITH_VIDEO and NCCL-wheel stages dropped (the kit's
# defaults WITH_VIDEO=0 and NCCL_WHEEL= leave them empty; the runtime image's NCCL is checked below). TOOLS=0 builds the server
# and the grammar library only: the CUDA dev CLI and the tf-*-test tools are not part of serving. ZIG_CPU=baseline, not the
# kit's `native`: the kit builds on the Spark that runs it, this image is built on a hosted arm64 runner and must run on a GB10.
# Thin layer on top: the Controller placement wrapper, the licence texts and the non-root user.
FROM nvcr.io/nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:afc92f9727f96346f2912934f11e5a43aa0581aca4a826bd5c8fd4ab6b152bcc AS build
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git xz-utils cmake nasm libibverbs-dev \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL --retry 3 -o /tmp/zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz \
 && echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig \
 && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
 && rm /tmp/zig.tar.xz \
 && ln -s /opt/zig/zig /usr/local/bin/zig \
 && test "$(zig version)" = 0.17.0

# Codecs: the image decoders --vision needs: the server refuses JPEG and WebP input unless it finds exactly libjpeg-turbo 3.1.4.1 and libwebp 1.6.0.
WORKDIR /build
RUN curl -fsSL --retry 3 -o jpeg.tar.gz https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-3.1.4.1.tar.gz \
 && echo "ecae8008e2cc9ade2f2c1bb9d5e6d4fb73e7c433866a056bd82980741571a022  jpeg.tar.gz" | sha256sum -c - \
 && mkdir jpeg && tar -xzf jpeg.tar.gz -C jpeg --strip-components=1 \
 && cmake -S jpeg -B jpeg/out -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/codecs -DCMAKE_INSTALL_LIBDIR=lib \
      -DENABLE_SHARED=ON -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON -DWITH_JAVA=OFF -DWITH_TOOLS=OFF -DWITH_TESTS=OFF \
 && cmake --build jpeg/out -j"$(nproc)" --target turbojpeg \
 && install -D -m 0755 "$(readlink -f jpeg/out/libturbojpeg.so.0)" /codecs/lib/libturbojpeg.so.0 \
 && install -D -m 0644 jpeg/LICENSE.md /codecs/LICENSES/libjpeg-turbo-LICENSE.md
RUN curl -fsSL --retry 3 -o webp.tar.gz https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz \
 && echo "e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564  webp.tar.gz" | sha256sum -c - \
 && mkdir webp && tar -xzf webp.tar.gz -C webp --strip-components=1 \
 && cmake -S webp -B webp/out -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
      -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_GIF2WEBP=OFF \
      -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF -DWEBP_BUILD_WEBPMUX=OFF \
      -DWEBP_BUILD_EXTRAS=OFF -DWEBP_BUILD_LIBWEBPMUX=OFF \
 && cmake --build webp/out -j"$(nproc)" --target webpdecoder \
 && install -D -m 0755 "$(readlink -f webp/out/libwebpdecoder.so.3)" /codecs/lib/libwebpdecoder.so.3 \
 && install -D -m 0644 webp/COPYING /codecs/LICENSES/libwebp-COPYING \
 && ldd /codecs/lib/libturbojpeg.so.0 /codecs/lib/libwebpdecoder.so.3

# Engine: TensorFold f350cd3e (the kit's TF_REF, by its codeload archive) with patches/engine/*.patch applied in filename order.
RUN git clone -q --depth 1 --branch v0.2.8 --recurse-submodules https://github.com/mlc-ai/xgrammar.git /src/xgrammar \
 && test "$(git -C /src/xgrammar rev-parse HEAD)" = 97787376faee5ed8466cfad57c99855e4ce2f6aa \
 && test "$(git -C /src/xgrammar describe --tags --exact-match)" = v0.2.8
WORKDIR /src/tensorfold
RUN curl -fsSL --retry 3 -o /tmp/tensorfold.tar.gz https://codeload.github.com/ashhart/TensorFold/tar.gz/f350cd3ea253a042e205746928271f9d8f71a82d \
 && echo "d40afea2621fdd7aad03c4b41220998c2cfca778154b8ec79c2323581c1ac601  /tmp/tensorfold.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/tensorfold.tar.gz --strip-components=1 -C /src/tensorfold \
 && rm /tmp/tensorfold.tar.gz
COPY patches/engine/ /opt/tf-patches/engine/
RUN shopt -s nullglob \
 && patches=(/opt/tf-patches/engine/*.patch) \
 && for p in "${patches[@]}"; do \
      echo "apply ${p##*/}"; git apply --check "${p}" && git apply "${p}" || exit 1; \
    done \
 && echo "${#patches[@]} engine patch(es) applied" \
 && test "${#patches[@]}" = 108
RUN test "$(cat .zig-version)" = 0.17.0 \
 && zig build native grammar --prefix /opt/tensorfold --summary failures \
      -Doptimize=fast -Dcpu=baseline -Dstrip=false \
      -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Dxgrammar=/src/xgrammar \
 && /opt/tensorfold/native/bin/tensorfold-native --version
RUN shopt -s nullglob \
 && mkdir -p /opt/tensorfold/LICENSES/tensorfold \
 && cp -r /codecs/lib/. /opt/tensorfold/native/lib/ \
 && cp -r /codecs/LICENSES/. /opt/tensorfold/LICENSES/ \
 && cp -r LICENSE NOTICE THIRD_PARTY_NOTICES.md LICENSES /opt/tensorfold/LICENSES/tensorfold/ \
 && cp /src/xgrammar/LICENSE /opt/tensorfold/LICENSES/xgrammar-Apache-2.0.txt \
 && cp /src/xgrammar/NOTICE /opt/tensorfold/LICENSES/xgrammar-NOTICE.txt \
 && cp /src/xgrammar/3rdparty/dlpack/LICENSE /opt/tensorfold/LICENSES/dlpack-Apache-2.0.txt

# The CUDA runtime image carries libcublasLt, libnccl.so.2 (the engine needs 2.30.0 or newer) and, with rdma-core below, the
# libibverbs and mlx5 provider the engine's RoCE all-gather dlopens. libcuda.so.1 comes from the host driver.
FROM nvcr.io/nvidia/cuda:13.3.1-runtime-ubuntu24.04@sha256:585155539dd4ff2ab653ee59d812aae81a9406a5804596d516c06e3a5719eb26
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104" \
      org.opencontainers.image.revision="02dac3879852b8c5713318d3d64a5a791002e954" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1" \
      tf.engine="zig" \
      tf.ref="f350cd3ea253a042e205746928271f9d8f71a82d" \
      tf.sm="121" \
      tf.zig="0.17.0"
RUN apt-get update \
 && apt-get install -y --no-install-recommends libibverbs1 ibverbs-providers rdma-core \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/tensorfold/ /opt/tensorfold/
RUN so=$(find /usr/lib /lib -name 'libnccl.so.2*' | head -1) \
 && test -n "${so}" \
 && ver=$( { grep -aoE '[0-9]+\.[0-9]+\.[0-9]+' "${so}" || true; } | { grep -xE '2\.[0-9]+\.[0-9]+' || true; } \
           | sort -u -t. -k1,1n -k2,2n -k3,3n | tail -1 ) \
 && echo "libnccl: ${so} (${ver:-unknown}), engine minimum 2.30.0" \
 && if [ -n "${ver}" ]; then \
      printf '%s\n%s\n' 2.30.0 "${ver}" | sort -C -t. -k1,1n -k2,2n -k3,3n \
        || { echo "libnccl ${ver} is older than 2.30.0" >&2; exit 1; }; \
    fi \
 && ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native \
 && tensorfold-native --version \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY licenses /opt/vonk/licenses
COPY chat_template.jinja /opt/vonk/templates/chat_template.jinja
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
ENV HF_HUB_OFFLINE=1 HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
