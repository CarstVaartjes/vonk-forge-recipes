FROM docker.io/eugr/spark-vllm-b12x@sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd

LABEL org.opencontainers.image.source="https://github.com/sfxnz/DeepSeek-V4-Flash-Vision-Exp-vLLM-2x-DGX-Spark" \
      org.opencontainers.image.revision="099b90321fa6629ce15b02f909fd9df1921b80ad" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

# Upstream docker/Dockerfile.b12x-vision installs the official ViT/aligner as a
# vLLM plugin and patches the DSpark loader; both are baked byte-identically.
COPY docker/plugin /opt/dsv4_vision_src
COPY docker/patch_dspark_skip_bias_vl.py /tmp/patch_dspark_skip_bias_vl.py
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY LICENSE /opt/vonk/LICENSE.upstream

RUN python3 -m pip install --no-cache-dir --no-deps /opt/dsv4_vision_src \
    && python3 /tmp/patch_dspark_skip_bias_vl.py \
    && rm -f /tmp/patch_dspark_skip_bias_vl.py \
    && python3 -c 'import dsv4_vision' \
    && install --directory --mode=0755 /outputs \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && chown -R 10001:10001 /outputs /opt/vonk

ENV VLLM_PLUGINS=dsv4_vision \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600

WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
