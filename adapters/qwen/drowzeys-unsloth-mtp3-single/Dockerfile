FROM ghcr.io/drowzeys/keys-vllm-027-gb10-qwen38@sha256:abd8ea18080ae571a72e8113d80775c38f4cfe07eacd591d4fedd70e95cd1bbf

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-NVFP4-MTP3-Single-DGX-Spark" \
      org.opencontainers.image.revision="047aa237efa334fde70427629e59b6f9606ec21d" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# The pinned image is upstream's mirror of eugr's spark-vllm-b12x
# nightly-20260813 (vLLM 0.27 GB10 build, loads Qwen3_5MTP). deploy/ and
# oneshot.sh are upstream's launchers, carried byte-identically for provenance;
# the recipe reproduces deploy/serve_throughput.sh (Profile A).
COPY deploy /opt/vonk/upstream-deploy
COPY oneshot.sh /opt/vonk/upstream-oneshot.sh

RUN install --directory --mode=0755 /opt/vonk/bin \
    && ln -s "$(command -v vllm)" /opt/vonk/bin/vllm \
    && test -x /opt/vonk/bin/vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch-extensions

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
