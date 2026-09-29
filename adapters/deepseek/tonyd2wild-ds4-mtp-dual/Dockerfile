FROM docker.io/aidendle94/sparkrun-vllm-ds4-gb10@sha256:f869281d869b2a1d418cade7dcbabe65216cb8f54891d5e2e50718c3dc8b630f

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/DeepSeek-V4-Flash-2x-Spark-1M" \
      org.opencontainers.image.revision="7cc2c55c7b6428dcf49401dae63583c2e0141cdf" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

# Upstream runs this published image unchanged: aidendle94's vLLM fork with b12x MoE
# kernels for GB10, launched through /usr/local/bin/dsv4-vllm-entrypoint, which only
# sets toolchain and cache environment and then execs `vllm`. This layer adds the
# placement wrapper (it execs /opt/env/bin/vllm with the same toolchain environment the
# image already bakes in) and the non-root user.
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.tonyd2wild-ds4
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN test -x /opt/env/bin/vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_XET=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/state \
    HF_HOME=/state/huggingface \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    DG_JIT_CACHE_DIR=/outputs/cache/deepgemm \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache/flashinfer \
    TILELANG_CACHE_DIR=/outputs/cache/tilelang \
    TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
