# Native TensorFold with the kit's patches, built on the creator's published image (see kit-lock.json for the commits).
# The creator has published no image of the pinned kit commit (its CHANGELOG: setup builds the image locally until one
# is published), so this rebuilds the creator's exact engine (P15): the kit's TensorFold commit, patches, Zig, build line
# and arch list (-Dsm default 121), with the published image's own CUDA toolkit. It replaces only what that build
# installs (bin/, fatbin/, native/); the published Triton AOT kernel set and vision Python tree are reused, and the
# build fails if the patched sources they come from differ from the published ones.
FROM ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold@sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c AS build
# The kit and TensorFold commits come from kit-lock.json (tools/kit-pins), checked by commit after the fetch.
COPY kit-lock.json /opt/vonk-build/kit-lock.json
RUN set -- $(python3 -c 'import json; d = json.load(open("/opt/vonk-build/kit-lock.json")); k, e = d["kit"], d["dependencies"]["engine"]; print(k["repo"], k["commit"], e["repo"], e["commit"])') \
 && fetch() { git init -q "$1" && git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3" \
      && git -C "$1" checkout -q --detach FETCH_HEAD && test "$(git -C "$1" rev-parse HEAD)" = "$3"; } \
 && fetch /kit "$1" "$2" && fetch /tensorfold "$3" "$4"
# Zig: the version and sha256 the kit declares in scripts/config.sh, read as data (never sourced).
RUN version=$(sed -n 's/^ZIG_VERSION="\${ZIG_VERSION:-\([0-9][0-9.]*\)}".*/\1/p' /kit/scripts/config.sh) \
 && sha=$(sed -n 's/^ZIG_SHA256="\${ZIG_SHA256:-\([0-9a-f]\{64\}\)}".*/\1/p' /kit/scripts/config.sh) \
 && if [ -z "$version" ] || [ -z "$sha" ] || [ "$(echo "$version" | wc -l)" -ne 1 ] || [ "$(echo "$sha" | wc -l)" -ne 1 ]; then \
      echo "scripts/config.sh declares no single ZIG_VERSION/ZIG_SHA256 (got '$version' '$sha')" >&2; exit 1; fi \
 && curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/$version/zig-aarch64-linux-$version.tar.xz" \
 && echo "$sha  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz \
 && test "$(/opt/zig/zig version)" = "$version"
RUN cd /tensorfold \
 && for p in /kit/patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
# Reuse net: the published vision Python tree is the patched src/tensorfold, and every Triton kernel source the
# published AOT kernel set was compiled from (zig/tests/cuda/flashnext/kernels.json) lies in it, so a kit change to
# those sources fails the build here.
RUN python3 -c 'import json, sys; files = {k["source"]["file"] for k in json.load(open(sys.argv[1]))["kernels"]}; bad = sorted(f for f in files if not f.startswith("tensorfold/")); sys.exit(f"kernel sources outside src/tensorfold: {bad}" if bad else 0)' \
      /tensorfold/zig/tests/cuda/flashnext/kernels.json \
 && diff -r /tensorfold/src/tensorfold /opt/tensorfold/python/tensorfold
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tf-engine \
      -j"$(nproc)" fatbins install native \
 && test -x /opt/tf-engine/native/bin/tensorfold-native \
 && test "$(ls /opt/tf-engine | tr '\n' ' ')" = "bin fatbin native "
# Build-time sm_121 kernel scan (scan-sm121.py; rules in docs/recipe-authoring.md). Every rebuilt fatbin, the exact
# images tensorfold-native embeds (@embedFile), must carry a loadable image; the rebuilt binaries are scanned too.
COPY scan-sm121.py /opt/vonk-build/scan-sm121.py
RUN ls /opt/tf-engine/fatbin/*.fatbin >/dev/null \
 && python3 /opt/vonk-build/scan-sm121.py $(for f in /opt/tf-engine/fatbin/*.fatbin; do echo --require $f; done) \
      /opt/tf-engine/native/bin/tensorfold-native /opt/tf-engine/bin/tensorfold /opt/tf-engine/bin/tf-cuda-test

FROM ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold@sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c

# The base image's tf.patches and description name its own patch set; this image's commits are in kit-lock.json.
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold" \
      org.opencontainers.image.revision="dc05c912342ec19467b3b7b686888de763f2ada1" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Qwen3.8 Flash Next on two DGX Sparks: TensorFold's Zig engine with the kit's patches, both at the commits in kit-lock.json" \
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
