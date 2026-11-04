# Native TensorFold with the kit's patches, rebuilt as the kit's scripts/prepare.sh builds it (commits: kit-lock.json).
# The creator has published no image of the pinned kit commit (its CHANGELOG: setup builds the image locally until one
# is published), so this rebuilds the creator's /opt/tensorfold (P15): the engine (zig build ... fatbins install
# native), its licence notices, the sm121 Triton AOT kernel set and the vision Python tree, from the kit's TensorFold
# commit, patches, Zig and kernel-source mtime, with the creator's build lines, on the published image (the kit's base
# and CUDA toolkit). Only the thin Controller wrapper below is ours.
FROM ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold@sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c AS build
# The kit and TensorFold commits come from kit-lock.json (tools/kit-pins), checked by commit after the fetch.
COPY kit-lock.json /opt/vonk-build/kit-lock.json
RUN set -- $(python3 -c 'import json; d = json.load(open("/opt/vonk-build/kit-lock.json")); k, e = d["kit"], d["dependencies"]["engine"]; print(k["repo"], k["commit"], e["repo"], e["commit"])') \
 && fetch() { git init -q "$1" && git -C "$1" fetch -q --depth 1 "https://github.com/$2.git" "$3" \
      && git -C "$1" checkout -q --detach FETCH_HEAD && test "$(git -C "$1" rev-parse HEAD)" = "$3"; } \
 && fetch /kit "$1" "$2" && fetch /tensorfold "$3" "$4"
# Values the kit owns in scripts/config.sh, read as data (never sourced); the build fails unless each is declared once.
RUN kit() { sed -n "s/^$1=\"\\\${$1:-\\($2\\)}\".*/\\1/p" /kit/scripts/config.sh; } \
 && version=$(kit ZIG_VERSION '[0-9][0-9.]*') && sha=$(kit ZIG_SHA256 '[0-9a-f]\{64\}') \
 && mtime=$(kit KERNEL_SOURCE_MTIME '[0-9][0-9]*') \
 && for value in "$version" "$sha" "$mtime"; do \
      if [ -z "$value" ] || [ "$(echo "$value" | wc -l)" -ne 1 ]; then \
        echo "scripts/config.sh does not declare ZIG_VERSION, ZIG_SHA256 and KERNEL_SOURCE_MTIME once each" >&2; exit 1; fi; done \
 && echo "$mtime" > /opt/vonk-build/kernel-source-mtime \
 && curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/$version/zig-aarch64-linux-$version.tar.xz" \
 && echo "$sha  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz \
 && test "$(/opt/zig/zig version)" = "$version"
# The build lines below are the kit's (its scripts/prepare.sh Dockerfile, config.sh): fail when the kit no longer has
# them. The kit's BASE_IMAGE is what the published image was built on; this image keeps that base, so a kit bump of it,
# with no published image, must fail here.
RUN for line in '/opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold' \
      '-j"$(nproc)" fatbins install native' 'for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do' \
      'PYTHONPATH=/tensorfold/src python -B tools/zig/flashnext_aot.py build' \
      '--spec zig/tests/cuda/flashnext/kernels.json --jit zig/tests/cuda/flashnext/jit.json --tp 1,2' \
      '--out /opt/tensorfold/share/tensorfold/cuda/sm121' \
      'COPY --from=build /tensorfold/src/tensorfold /opt/tensorfold/python/tensorfold'; do \
      grep -qF -- "$line" /kit/scripts/prepare.sh || { echo "the kit's prepare.sh no longer has: $line" >&2; exit 1; }; done \
 && grep -qF -- 'BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"' /kit/scripts/config.sh \
    || { echo "the kit's config.sh no longer has its BASE_IMAGE line" >&2; exit 1; } \
 && cd /tensorfold \
 && for p in /kit/patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
# The kit's whole final stage (pip installs, ARG, LABEL, ENV, WORKDIR) is what the published image holds and what the
# last stage below mirrors. kit-final-stage.txt is a copy of it: any change to that stage in the kit fails here.
COPY kit-final-stage.txt /opt/vonk-build/kit-final-stage.txt
RUN sed -n '/^FROM [$]{BASE_IMAGE}$/,/^DOCKERFILE$/p' /kit/scripts/prepare.sh \
    | diff -u /opt/vonk-build/kit-final-stage.txt - \
 && echo "the kit's final stage is unchanged"
# The creator's /opt/tensorfold is rebuilt from scratch, so nothing of the published one is left in it.
RUN rm -rf /opt/tensorfold \
 && cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Doptimize=fast --prefix /opt/tensorfold \
      -j"$(nproc)" fatbins install native \
 && test -x /opt/tensorfold/native/bin/tensorfold-native
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold && cd /tensorfold \
 && for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done
# Every Triton specialization, compiled for sm_121 without a GPU. flashnext_aot.py fails the build when a kernel the
# spec captured from the creator's reference build gets another Triton hash or cubin sha256 than the capture's.
RUN cd /tensorfold && python -B -c 'import json, os, sys; t = int(sys.argv[1]); [os.utime(os.path.join("src", f), (t, t)) for f in sorted({k["source"]["file"] for k in json.load(open("zig/tests/cuda/flashnext/kernels.json"))["kernels"]})]' "$(cat /opt/vonk-build/kernel-source-mtime)" \
 && PYTHONPATH=/tensorfold/src python -B tools/zig/flashnext_aot.py build \
      --spec zig/tests/cuda/flashnext/kernels.json --jit zig/tests/cuda/flashnext/jit.json --tp 1,2 \
      --out /opt/tensorfold/share/tensorfold/cuda/sm121 \
 && test -f /opt/tensorfold/share/tensorfold/cuda/sm121/aot.json
# Build-time sm_121 kernel scan (scan-sm121.py; rules in docs/recipe-authoring.md). Every rebuilt fatbin (the images
# tensorfold-native embeds with @embedFile) and every cubin of the AOT kernel set must carry a loadable image; the
# rebuilt binaries are scanned too.
COPY scan-sm121.py /opt/vonk-build/scan-sm121.py
RUN ls /opt/tensorfold/fatbin/*.fatbin /opt/tensorfold/share/tensorfold/cuda/sm121/cubins/*.cubin >/dev/null \
 && python3 /opt/vonk-build/scan-sm121.py \
      $(for f in /opt/tensorfold/fatbin/*.fatbin /opt/tensorfold/share/tensorfold/cuda/sm121/cubins/*.cubin; do echo --require $f; done) \
      /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tensorfold /opt/tensorfold/bin/tf-cuda-test

FROM ghcr.io/miaai-lab/qwen3.8-flash-dual-dgx-sparks-tensorfold@sha256:e188128ddabdc67ac02c05781f5c0c6b348db5f050cac297861d82b7f086e18c

# The base image's tf.patches and description name its own patch set; this image's commits are in kit-lock.json.
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Dual-DGX-Sparks-TensorFold" \
      org.opencontainers.image.revision="dc05c912342ec19467b3b7b686888de763f2ada1" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="Qwen3.8 Flash Next on two DGX Sparks: TensorFold's Zig engine with the kit's patches, both at the commits in kit-lock.json" \
      ai.vonkforge.runtime-interface="v1" \
      tf.patches=""

# The creator's two COPY lines into a fresh /opt/tensorfold (its /usr/local/bin/tensorfold-native link stays valid).
RUN rm -rf /opt/tensorfold
COPY --from=build /opt/tensorfold /opt/tensorfold
COPY --from=build /tensorfold/src/tensorfold /opt/tensorfold/python/tensorfold
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
