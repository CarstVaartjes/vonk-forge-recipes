FROM docker.io/lmsysorg/sglang@sha256:6fb82d4fa57425bc0df5b35fba328b08bda950ecf78ce60ea63d8830fc271eb5

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-gb10-native-mtp" \
      org.opencontainers.image.revision="c962efdf2b80bbf71e092bb3f4ca65a6d1380386" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Upstream mounts patches/qwen3_5.py over the image's loader (treats compressed-tensors like
# modelopt_fp4 for the GDN linear-attention projections); the recipe bakes it in instead.
COPY qwen3_5.py /sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
