FROM docker.io/lmsysorg/sglang@sha256:67873eb93b994736ab534111f79b5aa93d2575b973ebf9d276c40d548ff9afec

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/DeepSeek-V4-Flash-Vision-SGLang-DGX-Spark" \
      org.opencontainers.image.revision="b42871622be907defdbb01faecbbe382fe9a8855" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      io.vonk.sglang.upstream-image="lmsysorg/sglang:dev-v4f-2dgx-v2 (SGLang branch b12x-vision @ 452239a74f)"

# Upstream runs the SGLang DGX Spark preview image unchanged (index digest above, the
# dev-v4f-2dgx-v2 tag); this layer only adds the placement wrapper and the non-root user.
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.tonyd2wild-sglang-vision
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py

RUN python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

# Upstream mounts a host directory at /root/.cache so the CuTeDSL and FlashInfer JIT and
# autotune results survive a relaunch; keep them on the platform-owned cache volume.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    DO_NOT_TRACK=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
