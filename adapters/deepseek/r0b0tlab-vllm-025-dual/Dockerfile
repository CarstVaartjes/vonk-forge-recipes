FROM ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark" \
      org.opencontainers.image.revision="8a4ef5a5ce1196f3c1b6b8c2005087ffd9c52f03" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1" \
      io.r0b0tlab.model.id="deepseek-ai/DeepSeek-V4-Flash-DSpark" \
      io.r0b0tlab.model.revision="913f0657a874f76844e2e91cbe706dbcaceeb6d7" \
      io.r0b0tlab.kv-cache.dtype="nvfp4_ds_mla" \
      io.r0b0tlab.speculative.method="dspark"

# Parent: Anemll dspark-vllm-gx10 at the digest upstream pins (source 47503f8e, vLLM 0.25.2.dev0,
# arm64, CUDA 13.0). recipe/Dockerfile.production in this directory is upstream's production
# layer (runtime manifest, audit and entrypoint); this layer adds only the placement wrapper
# and the non-root user.
COPY docker/runtime-manifest.production.json /opt/r0b0tlab/runtime-manifest.json
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN test -x /usr/local/bin/vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    KV_CACHE_DTYPE=nvfp4_ds_mla \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
