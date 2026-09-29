FROM ghcr.io/tonyd2wild/vllm-m3-chthonic@sha256:cdeae56d23d0d3f35a21c67c9f3e47979725a1ef9cd65faa6134b91871497be2

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/Minimax-M3-NVFP-3x-DGX-Sparks-TP-3" \
      org.opencontainers.image.revision="e29e14595086d0251109d22aa219eb089d24a988" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT"

# Upstream (m3vllm-roce.sh) installs Ray and b12x master at every container
# start and then re-points NCCL to the host-built v2.30u1 (the pip installs
# clobber it). The runtime root is read-only here, so the same steps run once at
# build time, in the same order, with the same pins.
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py

RUN pip install --no-cache-dir "ray==2.55.1" \
    && pip install --no-cache-dir --force-reinstall --no-deps \
       https://github.com/lukealonso/b12x/archive/08e980c303b0b6291700a6b85aa09aa874fc27cb.tar.gz \
    && PIPNCCL=/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib \
    && rm -f "$PIPNCCL/libnccl.so.2" \
    && ln -sf /opt/nccl230/build/lib/libnccl.so.2 "$PIPNCCL/libnccl.so.2" \
    && python /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs \
    && test -x /opt/vonk/bin/vllm

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    RAY_DEDUP_LOGS=0 \
    RAY_memory_monitor_refresh_ms=0 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
