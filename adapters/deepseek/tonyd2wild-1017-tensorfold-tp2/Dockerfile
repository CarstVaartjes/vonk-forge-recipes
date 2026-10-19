FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark" \
      org.opencontainers.image.revision="9fa230d3eb119f588e690bf9b87ce41ed902db60" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      ai.vonkforge.runtime-interface="v1"
COPY vendor /tmp/source
COPY patches /tmp/patches
RUN curl -fsSL https://codeload.github.com/bertholomus/TensorFold/tar.gz/3ff3b2ac1d71a789cb930068c46d9abed025eefa -o /tmp/source/engine.tar.gz \
 && curl -fsSL https://codeload.github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/tar.gz/1b5ebbee3cc8b63b8b875e46a314135eb6d669db -o /tmp/source/kit.tar.gz \
 && echo "d21ab5470320ecb88a8cbeee2085380ec6bb9fdbb2436bdfe8090c71934d77ff  /tmp/source/engine.tar.gz" | sha256sum -c - \
 && echo "9be67d01862125809fc9f148921f66d129f2ea6393461108d68f5e9189bcb403  /tmp/source/kit.tar.gz" | sha256sum -c - \
 && mkdir -p /tmp/engine /tmp/recipe /opt/tensorfold/bin /opt/tensorfold/tools /opt/vonk/bin \
 && tar -xzf /tmp/source/engine.tar.gz --strip-components=1 -C /tmp/engine \
 && tar -xzf /tmp/source/kit.tar.gz --strip-components=1 -C /tmp/recipe \
 && cd /tmp/engine && for p in /tmp/patches/*.patch; do patch -p1 --batch --fuzz=0 < "$p"; done
RUN curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz -o /tmp/zig.tar.xz \
 && echo '9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz' | sha256sum -c - \
 && tar -xJf /tmp/zig.tar.xz -C /tmp \
 && cd /tmp/engine \
 && /tmp/zig-aarch64-linux-0.17.0/zig build install native -Dtarget=aarch64-linux-gnu.2.28 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 \
 && cp zig-out/native/bin/tensorfold-native zig-out/bin/tf-dsv41-lanes zig-out/bin/tf-dsv41-rank-cache /opt/tensorfold/bin/ \
 && cp tools/dsv41-zig/inputs/make_token_map.py /opt/tensorfold/tools/ \
 && cp -a /tmp/recipe/kit /opt/tensorfold/kit \
 && echo '9be67d01862125809fc9f148921f66d129f2ea6393461108d68f5e9189bcb403' > /opt/tensorfold/kit/VONK_SOURCE_SHA256 \
 && cp /tmp/source/make_rope.py /opt/tensorfold/tools/make_rope.py \
 && cp /tmp/recipe/kit/tools/make_bias_vl.py /opt/tensorfold/tools/ \
 && cp LICENSE /opt/tensorfold/LICENSE \
 && rm -rf /tmp/engine /tmp/recipe /tmp/source /tmp/patches /tmp/zig* /root/.cache/zig
COPY requirements.lock /opt/tensorfold/requirements.lock
RUN python3 -m pip install --no-deps --require-hashes --no-cache-dir -r /opt/tensorfold/requirements.lock \
 && python3 -c 'import torch, tokenizers; print(tokenizers.__version__)' \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY chat_template.jinja prepare_cache.py launcher.py /opt/vonk/bin/
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
ENV VONK_TF_WORLD=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
