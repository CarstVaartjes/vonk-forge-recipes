FROM docker.io/scitrera/dgx-spark-vllm-jasl-ds4@sha256:cdd18e2f1dee30af144001400fe9dd4e2a6e33b7d1d8111bb53ea4cbd1e2be1a

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/DS4-Jasl-Sparkrun-GB10-Fix" \
      org.opencontainers.image.revision="050f1ba36d376f6a629894be795ebb79bc74501d" \
      org.opencontainers.image.licenses="Apache-2.0"

# Upstream runs scitrera's published image (vLLM 0.22.1-jasl-ds4+21a54ea, tag 20260609,
# digest above) unchanged through sparkrun; this layer adds the placement wrapper and
# the non-root user.
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN python3 -c "import shutil; assert shutil.which('vllm', path='/usr/local/bin:/usr/bin:/data/vllm')" \
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
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
