# sfxnz's GLM-5.3-Flash EXL3 4bpw recipe (kit 01ffd10b, "v4 / V6-m4"): TensorFold's stock 1.2.10 runtime with its
# tensorfold-native replaced by a source build of the kit's 39 engine patches (engine-patches/, applied in `series` order) on
# ashhart/TensorFold 81de26f6, as the kit's docker/Dockerfile.patched does. The creator publishes no image; its docker/Dockerfile
# builds the runtime from NVIDIA's CUDA 13.0 base, and so does this, with these changes: its three `ADD` downloads are curl steps
# checked against the creator's sha256 (the recipe build policy forbids ADD), python3 is installed for the Controller launcher,
# and the final user is numeric. The binary build follows adapters/glm/mia-1104-tensorfold-dual (same engine base): the engine
# archive is not a git checkout, so the patches go in with `git apply`; two deviations from the kit's `zig build`: -Dcpu=baseline,
# not the host's, because this image is built on a hosted arm64 runner and must run on a GB10, and -Doptimize=fast, Zig 0.17's
# name for the kit's ReleaseFast. The release
# archive carries everything else (codecs' libtfgrammar, the CUDA kernels the unpatched binary embeds).
ARG TF_VERSION=1.2.10
ARG TF_SHA256=c335c44f81d55847a5f81d74a94ac3d09f81bd6688247cff972cf1f63fc18105

# Every FROM pins the digest (the recipe source policy checks each one).
# Build stage: the patched native server, CUDA 13.0 as the kit builds it.
FROM docker.io/nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:1e8ac7a54c184a1af8ef2167f28fa98281892a835c981ebcddb1fad04bdd452d AS build
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git xz-utils python3 \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL --retry 3 -o /tmp/zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz \
 && echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig \
 && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
 && rm /tmp/zig.tar.xz \
 && ln -s /opt/zig/zig /usr/local/bin/zig \
 && test "$(zig version)" = 0.17.0
# Engine: TensorFold 81de26f6 (engine-patches/BASE), by its codeload archive, with engine-patches/series applied in order.
WORKDIR /src/tensorfold
RUN curl -fsSL --retry 3 -o /tmp/tensorfold.tar.gz https://codeload.github.com/ashhart/TensorFold/tar.gz/81de26f649677aa72ae0034d4af6e010cd947038 \
 && echo "28e565c5734459b87862ce580cf663bf686c7afa9ff8b860da1d9c593a5f9b59  /tmp/tensorfold.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/tensorfold.tar.gz --strip-components=1 -C /src/tensorfold \
 && rm /tmp/tensorfold.tar.gz
COPY engine-patches/ /opt/tf-patches/
RUN n=0; while read -r p; do \
      echo "apply ${p}"; git apply --check "/opt/tf-patches/${p}" && git apply "/opt/tf-patches/${p}" || exit 1; n=$((n + 1)); \
    done < /opt/tf-patches/series \
 && echo "${n} engine patch(es) applied"
RUN test "$(cat .zig-version)" = 0.17.0 \
 && zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast -Dcpu=baseline --summary failures \
 && test -x zig-out/native/bin/tensorfold-native
# Build-time sm_121 kernel scan (scan-sm121.py, docs/recipe-authoring.md): every kernel image the engine embeds (the `fatbins`
# output of this same build) and the server itself must carry code a GB10 can load. Not covered: a runtime JIT.
COPY scan-sm121.py /opt/scan-sm121.py
RUN zig build fatbins -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc --prefix /tmp/fat \
 && ls /tmp/fat/fatbin/*.fatbin >/dev/null \
 && python3 /opt/scan-sm121.py $(for f in /tmp/fat/fatbin/*.fatbin; do echo --require $f; done) \
      zig-out/native/bin/tensorfold-native \
 && rm -rf /tmp/fat

# TensorFold refuses any JPEG/WebP decoder whose embedded version is not exactly libjpeg-turbo 3.1.4.1 / libwebp 1.6.0,
# and Ubuntu 24.04 ships older ones.
FROM docker.io/nvidia/cuda:13.0.0-base-ubuntu24.04@sha256:6e43a6b02e5f16e4a715953be8b29c40acbe84aed11a61357ef0fc899967fbd9 AS codecs
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl gcc libc6-dev make && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /dl /out /src/jpeg \
 && curl -fsSL -o /dl/libjpeg-turbo.deb https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/3.1.4.1/libjpeg-turbo-official_3.1.4.1_arm64.deb \
 && curl -fsSL -o /dl/libwebp.tar.gz https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz \
 && printf '%s  %s\n' \
      d3f9d7ddd998758e1850b37153a7904bdcdeb8cf54a782c772780b0bc50dd553 /dl/libjpeg-turbo.deb \
      e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564 /dl/libwebp.tar.gz | sha256sum -c - \
 && dpkg-deb -x /dl/libjpeg-turbo.deb /src/jpeg \
 && cp -L /src/jpeg/opt/libjpeg-turbo/lib64/libturbojpeg.so.0 /out/ \
 && tar -xzf /dl/libwebp.tar.gz -C /src && cd /src/libwebp-1.6.0 \
 && ./configure --enable-libwebpdecoder --disable-static \
 && make -C sharpyuv -j"$(nproc)" && make -C src -j"$(nproc)" \
 && cp -L src/.libs/libwebpdecoder.so.3 /out/

FROM docker.io/nvidia/cuda:13.0.0-base-ubuntu24.04@sha256:6e43a6b02e5f16e4a715953be8b29c40acbe84aed11a61357ef0fc899967fbd9
ARG TF_VERSION
ARG TF_SHA256
LABEL io.vonk.recipe.source="https://github.com/sfxnz/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-Spark/tree/01ffd10bfc800202878833afe62757260fd1470b" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT"
# NCCL 2.30.7 is the version upstream's GLM TP receipts ran; ibverbs-providers carries the mlx5 provider for RoCE gathers.
# curl only fetches the release archive and leaves with the apt lists.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libcublas-13-0=13.1.1.3-1 libnccl2=2.30.7-1+cuda13.3 libibverbs1 ibverbs-providers python3-minimal ca-certificates curl \
 && mkdir /opt/tensorfold \
 && curl -fsSL -o /tmp/tensorfold.tar.gz https://github.com/ashhart/TensorFold/releases/download/v${TF_VERSION}/tensorfold-${TF_VERSION}-linux-aarch64.tar.gz \
 && echo "${TF_SHA256}  /tmp/tensorfold.tar.gz" | sha256sum -c - \
 && tar -xzf /tmp/tensorfold.tar.gz -C /opt/tensorfold --strip-components=1 \
 && cd /opt/tensorfold && sha256sum -c --quiet SHA256SUMS \
 && rm /tmp/tensorfold.tar.gz \
 && apt-get purge -y --auto-remove curl ca-certificates && rm -rf /var/lib/apt/lists/*
# ../lib from the binary is the first place the server looks for libtfgrammar and both codecs.
COPY --from=codecs /out/ /opt/tensorfold/lib/
# The patched binary replaces the release's; its SHA256SUMS line is rewritten as the kit's docker/Dockerfile.patched does.
COPY --from=build /src/tensorfold/zig-out/native/bin/tensorfold-native /opt/tensorfold/bin/tensorfold-native
RUN bin=$(sha256sum /opt/tensorfold/bin/tensorfold-native | cut -d' ' -f1) \
 && cd /opt/tensorfold && sed -i "s|^[0-9a-f]\{64\}  \./bin/tensorfold-native\$|${bin}  ./bin/tensorfold-native|" SHA256SUMS \
 && grep -q "^${bin}  ./bin/tensorfold-native\$" SHA256SUMS \
 && sha256sum -c --quiet SHA256SUMS \
 && /opt/tensorfold/bin/tensorfold-native --version
# libcuda.so.1 comes from the host driver at run time.
RUN ! ldd /opt/tensorfold/bin/tensorfold-native | grep "not found" | grep -v -e libcuda.so -e libnvidia
LABEL tensorfold.version=${TF_VERSION} \
      tensorfold.archive=tensorfold-${TF_VERSION}-linux-aarch64.tar.gz \
      tensorfold.archive.sha256=${TF_SHA256}

COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
