# tonyd2wild's DeepSeek-V4.1-Flash-1017 TensorFold TP4 lane on four DGX Sparks with LibertAI's 3.5 bpw EXL3 pack
# (tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks 74ac0136, runs/2026-10-17-1017-tensorfold and
# runs/2026-10-26-1017-tensorfold-max-pool). The creator publishes no image: his launch/setup.sh and launch/build.sh build
# bertholomus/TensorFold deepseek-v41-zig-tp4 at 59402086 with his four patches (patches/, `git am`; applied here with
# patch --fuzz=0) in nvcr.io/nvidia/pytorch:26.07-py3 with Zig 0.17.0 for aarch64 + sm_121, and run bertholomus's
# deepseek-v4.1-tensorfold-tp4-4xgb10 v0.1.1 (6033ea48; its ENGINE_COMMIT is the same engine commit) scripts/serve.sh and kernel kit.
# This file is that build, pinned, with a build-time sm_121 kernel scan after.
FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-TP2-and-TP4-DGX-Sparks" \
      org.opencontainers.image.revision="74ac01362e96c1dc72df96737354ce408dceb272" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      ai.vonkforge.runtime-interface="v1"
COPY patches /tmp/patches
RUN curl -fsSL https://codeload.github.com/bertholomus/TensorFold/tar.gz/59402086381a42c74d53053443224eec2790b4c5 -o /tmp/engine.tar.gz \
 && curl -fsSL https://codeload.github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10/tar.gz/6033ea48a27004b2715db82190e2a61f33411d92 -o /tmp/kit.tar.gz \
 && echo "2ec8c714ea41752fdc217081c220c451fa875d35324ff3a5da9dae36e24f815f  /tmp/engine.tar.gz" | sha256sum -c - \
 && echo "05718cd5477e83cb4f434b0b5b9bc494e97b39e471bbff730eabb176324f657a  /tmp/kit.tar.gz" | sha256sum -c - \
 && mkdir -p /tmp/engine /tmp/recipe /opt/tensorfold/bin /opt/tensorfold/fatbin /opt/tensorfold/tools /opt/vonk/bin \
 && tar -xzf /tmp/engine.tar.gz --strip-components=1 -C /tmp/engine \
 && tar -xzf /tmp/kit.tar.gz --strip-components=1 -C /tmp/recipe \
 && test "$(cat /tmp/recipe/ENGINE_COMMIT)" = 59402086381a42c74d53053443224eec2790b4c5 \
 && cd /tmp/engine && for p in /tmp/patches/*.patch; do patch -p1 --batch --fuzz=0 < "$p"; done
# launch/build.sh: `zig build install native`, plus the `fatbins` step of the same build, which installs the kernel images
# the binaries embed (zig/build/cuda.zig) so the scan below can read their device code.
RUN curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz -o /tmp/zig.tar.xz \
 && echo '9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz' | sha256sum -c - \
 && tar -xJf /tmp/zig.tar.xz -C /tmp \
 && cd /tmp/engine \
 && /tmp/zig-aarch64-linux-0.17.0/zig build install native fatbins -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 \
 && cp zig-out/native/bin/tensorfold-native zig-out/bin/tf-dsv41-lanes zig-out/bin/tf-dsv41-rank-cache /opt/tensorfold/bin/ \
 && cp zig-out/fatbin/*.fatbin /opt/tensorfold/fatbin/ \
 && cp tools/dsv41-zig/inputs/make_token_map.py /opt/tensorfold/tools/ \
 && cp -a /tmp/recipe/kit /opt/tensorfold/kit \
 && echo '05718cd5477e83cb4f434b0b5b9bc494e97b39e471bbff730eabb176324f657a' > /opt/tensorfold/kit/VONK_SOURCE_SHA256 \
 && cp /tmp/recipe/kit/tools/make_rope.py /tmp/recipe/kit/tools/make_bias_vl.py /opt/tensorfold/tools/ \
 && cp LICENSE /opt/tensorfold/LICENSE && cp /tmp/recipe/LICENSE /opt/tensorfold/RECIPE-LICENSE && cp /tmp/recipe/NOTICE /opt/tensorfold/RECIPE-NOTICE \
 && rm -rf /tmp/engine /tmp/recipe /tmp/*.tar.gz /tmp/patches /tmp/zig* /root/.cache/zig
# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or sm_121a; every
# file with device code needs a loadable image; each file's listing is echoed). Covered: every kernel image the engine embeds
# (through the `fatbins` install of the same build, each required to carry a loadable image) and the three binaries. Not covered:
# the kit's prebuilt cubins, which the pinned kit archive supplies as the creator serves them.
COPY scan-sm121.py /opt/scan-sm121.py
RUN python3 /opt/scan-sm121.py $(for f in /opt/tensorfold/fatbin/*.fatbin; do echo --require $f; done) \
      /opt/tensorfold/bin/tensorfold-native /opt/tensorfold/bin/tf-dsv41-lanes /opt/tensorfold/bin/tf-dsv41-rank-cache \
 && rm -rf /opt/tensorfold/fatbin /opt/scan-sm121.py
COPY requirements.lock /opt/tensorfold/requirements.lock
RUN python3 -m pip install --no-deps --require-hashes --no-cache-dir -r /opt/tensorfold/requirements.lock \
 && python3 -c 'import torch, tokenizers; print(tokenizers.__version__)' \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY chat_template.jinja prepare_cache.py launcher.py /opt/vonk/bin/
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
ENV VONK_TF_WORLD=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
