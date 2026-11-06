# sfxnz's GLM-5.3-Flash EXL3 4bpw recipe (kit 9e5933c0): the stock TensorFold 1.2.2 release, no fork and no patches. The
# creator publishes no image; its docker/Dockerfile builds one from NVIDIA's CUDA 13.0 base, and so does this, with three
# changes: its three `ADD` downloads are curl steps checked against the creator's sha256 (the recipe build policy forbids
# ADD), python3 is installed for the Controller launcher, and the final user is numeric. The release archive carries the
# CUDA kernels it embeds.
ARG TF_VERSION=1.2.2
ARG TF_SHA256=17632c30f761e7f9051255744a17fc36f0b9c0d46343d47617bb34e7ec4e4128

# Every FROM pins the digest (the recipe source policy checks each one).
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
LABEL io.vonk.recipe.source="https://github.com/sfxnz/GLM-5.3-Flash-EXL3-4bpw-TensorFold-2x-DGX-Spark/tree/9e5933c085e60361104e2dc7157661bdaa8829fb" \
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
RUN /opt/tensorfold/bin/tensorfold-native --version
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
