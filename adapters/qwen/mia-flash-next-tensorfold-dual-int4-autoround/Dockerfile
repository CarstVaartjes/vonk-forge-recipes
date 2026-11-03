# Native TensorFold at db28187 with the kit's patches 0001-0012, on the creator's published image, thin placement wrapper.
# The creator has published no image of this kit commit: its CHANGELOG says patches 0010-0012 change the engine and
# "setup builds the image locally until one is published for these patches (the v1.1 digest in scripts/config.sh does
# not apply)"; the kit's image_hash for these inputs is 3dfe459421d5, and GHCR has only zig-db28187-f3e1300f4f62 (this
# FROM, kit commit f8287ef4) and zig-db28187-57a020c8d666. Since f8287ef4 the kit changed only Zig engine sources
# (0007's zig/src/families/flashnext/cuda_lanes.zig, the PR #2 revert; 0010-0012's cuda_lanes.zig and cuda_mirror.zig):
# no CUDA kernel source, Triton kernel source, kernels.json or vision Python. So the image keeps the published image's
# CUDA toolkit, sm121 Triton AOT kernel set (/opt/tensorfold/share/tensorfold/cuda/sm121) and vision Python tree, and
# rebuilds only what the kit's `zig build ... fatbins install native` installs (bin/, fatbin/, native/), with the kit's
# own TensorFold commit, patches, Zig 0.17.0, nvcc and arch list (-Dsm default 121), exactly as scripts/prepare.sh does.
FROM ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold@sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c AS build
RUN curl -fsSL -o /tmp/zig.tar.xz https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz \
 && echo "9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz \
 && /opt/zig/zig version
# The kit commit and the TensorFold commit it declares come from kit-lock.json (tools/kit-pins), checked by commit.
COPY kit-lock.json /opt/vonk-build/kit-lock.json
RUN set -- $(python3 -c 'import json; d = json.load(open("/opt/vonk-build/kit-lock.json")); k, e = d["kit"], d["dependencies"]["engine"]; print(k["repo"], k["commit"], e["repo"], e["commit"])') \
 && fetch() { git init -q "$1" && git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3" \
      && git -C "$1" checkout -q --detach FETCH_HEAD && test "$(git -C "$1" rev-parse HEAD)" = "$3"; } \
 && fetch /kit "$1" "$2" && fetch /tensorfold "$3" "$4" \
 && cd /tensorfold \
 && for p in /kit/patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tf-engine \
      -j"$(nproc)" fatbins install native \
 && test -x /opt/tf-engine/native/bin/tensorfold-native \
 && test "$(ls /opt/tf-engine | tr '\n' ' ')" = "bin fatbin native "
# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or
# sm_121a; every file with device code needs a loadable image; each file's listing is echoed). Every rebuilt kernel
# image is required to carry a loadable image: the fatbins the build installs are the exact images tensorfold-native
# embeds (zig/src/cuda/kernels.zig, @embedFile). The rebuilt binaries are scanned too. Not rebuilt, so not scanned: the
# published image's sm121 Triton AOT kernel set.
COPY scan-sm121.py /opt/vonk-build/scan-sm121.py
RUN ls /opt/tf-engine/fatbin/*.fatbin >/dev/null \
 && python3 /opt/vonk-build/scan-sm121.py $(for f in /opt/tf-engine/fatbin/*.fatbin; do echo --require $f; done) \
      /opt/tf-engine/native/bin/tensorfold-native /opt/tf-engine/bin/tensorfold /opt/tf-engine/bin/tf-cuda-test

FROM ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold@sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c

# tf.patches and the description name the base image's patch set (f3e1300f4f62); this image's patches are the kit
# commit's (org.opencontainers.image.revision), recorded with the TensorFold commit in kit-lock.json.
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold" \
      org.opencontainers.image.revision="dc05c912342ec19467b3b7b686888de763f2ada1" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Qwen3.8 Flash Next on two DGX Sparks: TensorFold's Zig engine (db281878ddb8) with the kit's patches" \
      ai.vonkforge.runtime-interface="v1" \
      tf.patches=""

COPY --from=build /opt/tf-engine/ /opt/tensorfold/
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN test -x "$(command -v tensorfold-native)" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# First start JIT-compiles the CUDA kernels for GB10 (and the RoCE all-gather) into the platform's writable /outputs/cache.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
