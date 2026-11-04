# bertholomus's DeepSeek-V4.1-Flash TensorFold TP4 recipe v0.3.6 (deepseek-v4.1-tensorfold-tp4-4xgb10 16f65a98). The
# creator publishes no image: RUNNING.md section 1 builds the engine (bertholomus/TensorFold deepseek-v41-zig-tp4 at the
# kit's ENGINE_COMMIT 953bd6fd, locked in kit-lock.json) with Zig 0.17.0 and the CUDA toolchain of
# nvcr.io/nvidia/pytorch:26.07-py3 for aarch64 + sm_121, which is what this image does, with a build-time sm_121
# kernel scan after. tonyd2wild's TF_RDMA_GID and TF_SEED_RANDOM/TF_TOP_K_MAX changes are upstream in this engine.
FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10" \
      org.opencontainers.image.revision="16f65a98c6ff1c628e49ef29a8037fe2feb034c6" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      ai.vonkforge.runtime-interface="v1"
RUN curl -fsSL https://codeload.github.com/bertholomus/TensorFold/tar.gz/953bd6fd50149041bceaa5abd0971de81e2cbbd0 -o /tmp/engine.tar.gz \
 && curl -fsSL https://codeload.github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/tar.gz/16f65a98c6ff1c628e49ef29a8037fe2feb034c6 -o /tmp/kit.tar.gz \
 && echo "e43272277d9b23a9ced020cf87b8ee23a226b50df0d18c0cb01ffe0bccd37500  /tmp/engine.tar.gz" | sha256sum -c - \
 && echo "c68c794a84ccb348d711aadc96ca03d0f740eb51967a613a175a2891f6424ad0  /tmp/kit.tar.gz" | sha256sum -c - \
 && mkdir -p /tmp/engine /tmp/recipe /opt/tensorfold/bin /opt/tensorfold/fatbin /opt/tensorfold/tools /opt/vonk/bin \
 && tar -xzf /tmp/engine.tar.gz --strip-components=1 -C /tmp/engine \
 && tar -xzf /tmp/kit.tar.gz --strip-components=1 -C /tmp/recipe \
 && test "$(cat /tmp/recipe/ENGINE_COMMIT)" = 953bd6fd50149041bceaa5abd0971de81e2cbbd0
# RUNNING.md section 1's build; the `fatbins` step of the same build installs the kernel images it embeds, among them
# dsv41_rdma.fatbin (zig/kernels/cuda/dsv41/rdma_gather.cu for sm_121), the served launch's rdma_gather.fatbin.
RUN curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz -o /tmp/zig.tar.xz \
 && echo '9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz' | sha256sum -c - \
 && tar -xJf /tmp/zig.tar.xz -C /tmp \
 && cd /tmp/engine \
 && /tmp/zig-aarch64-linux-0.17.0/zig build install native fatbins -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 \
 && cp zig-out/native/bin/tensorfold-native zig-out/bin/tf-dsv41-lanes zig-out/bin/tf-dsv41-rank-cache /opt/tensorfold/bin/ \
 && cp zig-out/fatbin/*.fatbin /opt/tensorfold/fatbin/ \
 && cp tools/dsv41-zig/inputs/make_token_map.py /opt/tensorfold/tools/ \
 && cp -a /tmp/recipe/kit /opt/tensorfold/kit \
 && echo 'c68c794a84ccb348d711aadc96ca03d0f740eb51967a613a175a2891f6424ad0' > /opt/tensorfold/kit/VONK_SOURCE_SHA256 \
 && cp /tmp/recipe/kit/tools/make_rope.py /tmp/recipe/tools/serving_1017.py /opt/tensorfold/tools/ \
 && cp /tmp/recipe/chat/chat_template.jinja /opt/vonk/bin/chat_template.jinja \
 && cp LICENSE /opt/tensorfold/LICENSE && cp /tmp/recipe/LICENSE /opt/tensorfold/RECIPE-LICENSE && cp /tmp/recipe/NOTICE /opt/tensorfold/RECIPE-NOTICE \
 && rm -rf /tmp/engine /tmp/recipe /tmp/*.tar.gz /tmp/zig* /root/.cache/zig
# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or
# sm_121a; every file with device code needs a loadable image; each file's listing is echoed). Covered: every kernel
# image the engine embeds, through the `fatbins` install of the same build (each required to carry a loadable image):
# the Zig build embeds those exact files as raw module bytes (zig/build/cuda.zig), so cuobjdump sees no device code in
# tensorfold-native itself. The three binaries are still scanned (any device code they hold must load). Not covered: the
# kit's prebuilt cubins, which the pinned kit archive supplies as the creator serves them.
COPY scan-sm121.py /opt/scan-sm121.py
RUN python3 /opt/scan-sm121.py $(for f in /opt/tensorfold/fatbin/*.fatbin; do echo --require $f; done) \
      /opt/tensorfold/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes /opt/tensorfold/bin/tf-dsv41-rank-cache \
 && find /opt/tensorfold/fatbin -type f ! -name dsv41_rdma.fatbin -delete \
 && test -s /opt/tensorfold/fatbin/dsv41_rdma.fatbin \
 && rm /opt/scan-sm121.py
COPY requirements.lock /opt/tensorfold/requirements.lock
RUN python3 -m pip install --no-deps --require-hashes --no-cache-dir -r /opt/tensorfold/requirements.lock \
 && python3 -c 'import torch, tokenizers; print(tokenizers.__version__)' \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY prepare_cache.py launcher.py /opt/vonk/bin/
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
ENV VONK_TF_WORLD=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
