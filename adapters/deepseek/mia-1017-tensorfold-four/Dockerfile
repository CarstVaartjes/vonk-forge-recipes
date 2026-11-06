# Mia AI Lab's DeepSeek-V4.1-Flash-1017 / 4x DGX Sparks / TensorFold kit (MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold 8ffa67d5).
# The creator publishes no image: scripts/prepare.sh builds zig/Dockerfile locally. This file is that build, pinned: BertholomusAI's
# TensorFold fork at the kit's TF_REF (59402086, locked in kit-lock.json) plus the kit's zig/patches (vendored), Zig 0.17.0, the
# xgrammar v0.2.8 checkout for libtfgrammar and PyAV 18.1.0, on the nvcr.io/nvidia/pytorch:26.07-py3 the kit builds on, for aarch64 +
# sm_121, with a build-time sm_121 kernel scan after. The kernel kit is BertholomusAI's four-node kit at the kit's KIT_REF (6033ea48);
# its RoPE tables and image bias are made from the checkpoint at the first start (prepare_cache.py), as scripts/prepare.sh does.
FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold" \
      org.opencontainers.image.revision="8ffa67d53ef15bab997399135f85e8b0910f30be" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      ai.vonkforge.runtime-interface="v1"
COPY requirements.lock /opt/tensorfold/requirements.lock
RUN python3 -m pip install --no-deps --require-hashes --no-cache-dir -r /opt/tensorfold/requirements.lock \
 && python3 -c 'import av, PIL, torch, tokenizers; print("PyAV", av.__version__, tokenizers.__version__)'
RUN curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz -o /tmp/zig.tar.xz \
 && echo '9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz' | sha256sum -c - \
 && mkdir -p /opt/zig \
 && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
 && rm /tmp/zig.tar.xz \
 && /opt/zig/zig version
RUN curl -fsSL https://codeload.github.com/bertholomus/TensorFold/tar.gz/59402086381a42c74d53053443224eec2790b4c5 -o /tmp/engine.tar.gz \
 && curl -fsSL https://codeload.github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/tar.gz/6033ea48a27004b2715db82190e2a61f33411d92 -o /tmp/kit.tar.gz \
 && echo "2ec8c714ea41752fdc217081c220c451fa875d35324ff3a5da9dae36e24f815f  /tmp/engine.tar.gz" | sha256sum -c - \
 && echo "05718cd5477e83cb4f434b0b5b9bc494e97b39e471bbff730eabb176324f657a  /tmp/kit.tar.gz" | sha256sum -c - \
 && mkdir -p /tmp/engine /tmp/recipe \
 && tar -xzf /tmp/engine.tar.gz --strip-components=1 -C /tmp/engine \
 && tar -xzf /tmp/kit.tar.gz --strip-components=1 -C /tmp/recipe \
 && rm /tmp/engine.tar.gz /tmp/kit.tar.gz \
 && test "$(cat /tmp/recipe/ENGINE_COMMIT)" = 59402086381a42c74d53053443224eec2790b4c5
# zig/patches/*.patch, in file name order, applied in the engine's root (zig/Dockerfile).
COPY zig/patches /opt/tf-patches
RUN cd /tmp/engine \
 && for p in /opt/tf-patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done
# CPU-only lost-wakeup and park/wake tests; no GPU or model weights needed (zig/Dockerfile).
RUN timeout 120s /opt/zig/zig test /tmp/engine/zig/src/families/dsv41/idle_gate.zig -O ReleaseSafe -lc -j1
# CPU-only rendezvous tests over loopback (0035).
RUN timeout 120s /opt/zig/zig test /tmp/engine/zig/src/families/dsv41/link.zig -O ReleaseSafe -lc -j1
# Structured output (0036): libtfgrammar, built by `zig build grammar` from xgrammar's v0.2.8 release (zig/Dockerfile ARG XGRAMMAR_TAG / XGRAMMAR_REF).
RUN git clone -q --depth 1 --branch v0.2.8 --recurse-submodules --shallow-submodules https://github.com/mlc-ai/xgrammar /opt/xgrammar \
 && test "$(git -C /opt/xgrammar rev-parse HEAD)" = 97787376faee5ed8466cfad57c99855e4ce2f6aa
# zig/Dockerfile's build with the `fatbins` step added: the Zig build embeds the nvcc kernel images in the binaries (zig/build/cuda.zig), and
# `fatbins` installs those same files so the scan below can read their device code.
RUN cd /tmp/engine \
 && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 -Doptimize=ReleaseFast -Dxgrammar=/opt/xgrammar --prefix /opt/tensorfold -j"$(nproc)" install native grammar fatbins \
 && test -x /opt/tensorfold/native/bin/tensorfold-native && test -x /opt/tensorfold/bin/tf-dsv41-lanes \
 && test -x /opt/tensorfold/bin/tf-dsv41-rank-cache && test -f /opt/tensorfold/lib/libtfgrammar.so \
 && mkdir -p /opt/tensorfold/share/doc/tensorfold/LICENSES /opt/tensorfold/tools /opt/vonk/bin \
 && cp /opt/xgrammar/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-Apache-2.0.txt \
 && cp /opt/xgrammar/NOTICE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-NOTICE.txt \
 && cp /opt/xgrammar/3rdparty/dlpack/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES/dlpack-Apache-2.0.txt \
 && cp tools/dsv41-zig/inputs/make_token_map.py /opt/tensorfold/tools/ \
 && for f in LICENSE NOTICE ATTRIBUTION.md THIRD_PARTY_NOTICES.md LICENSES; do if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done \
 && cp -a /tmp/recipe/kit /opt/tensorfold/kit \
 && cp /tmp/recipe/kit/tools/make_rope.py /tmp/recipe/kit/tools/make_bias_vl.py /opt/tensorfold/tools/ \
 && cp /tmp/recipe/LICENSE /opt/tensorfold/share/doc/tensorfold/KIT-LICENSE \
 && cp /tmp/recipe/NOTICE /opt/tensorfold/share/doc/tensorfold/KIT-NOTICE \
 && echo '05718cd5477e83cb4f434b0b5b9bc494e97b39e471bbff730eabb176324f657a' > /opt/tensorfold/kit/VONK_SOURCE_SHA256 \
 && rm -rf /tmp/engine /tmp/recipe /opt/xgrammar /opt/tf-patches /opt/zig /root/.cache/zig
# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or sm_121a; every file
# with device code needs a loadable image; each file's listing is echoed). Covered: every kernel image the engine embeds, through the
# `fatbins` install of the same build (each required to carry a loadable image), and the three binaries (any device code they hold must
# load). Not covered: the kit's prebuilt cubins (aot/cubins, cubins, vision/torch_fmha_sm120.cubin), which the pinned kit archive supplies
# as the creator serves them.
COPY scan-sm121.py /opt/scan-sm121.py
RUN python3 /opt/scan-sm121.py $(for f in /opt/tensorfold/fatbin/*.fatbin; do echo --require $f; done) \
      /opt/tensorfold/native/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes /opt/tensorfold/bin/tf-dsv41-rank-cache \
 && rm -rf /opt/tensorfold/fatbin /opt/scan-sm121.py \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY licenses /opt/tensorfold/licenses/mia
COPY chat_template.jinja prepare_cache.py launcher.py /opt/vonk/bin/
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
ENV VONK_TF_WORLD=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PATH=/opt/tensorfold/native/bin:/opt/tensorfold/bin:${PATH} \
    TENSORFOLD_GRAMMAR_LIB=/opt/tensorfold/lib/libtfgrammar.so \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
