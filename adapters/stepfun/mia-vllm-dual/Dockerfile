FROM docker.io/vllm/vllm-openai@sha256:00417ffbb7db2c9b0c7aafe6bb18348facfe45d2d0211fa908ca75af8f962e75

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Dual-DGX-Spark-Step-3.7-Flash-NVFP4" \
      org.opencontainers.image.revision="11b9a4187a363bcc993ccda7d38edc02c7624dfd" \
      org.opencontainers.image.licenses="MIT"

# The base is the official vllm/vllm-openai:stepfun37 image (linux/arm64) that
# upstream's Dockerfile.stepfun37-workspace wraps, pinned by digest. Upstream
# applies the NCCL load-order fix and the MTP drafter patch inside the container
# at every serve; here the same steps run once at build time so the image is
# read-only at run time. The MTP patch only touches the step3p5_mtp.py drafter.
COPY fix-nccl-load-order.sh patch-mtp-drafter-quant.py verify-runtime.py /opt/vonk/build/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN bash /opt/vonk/build/fix-nccl-load-order.sh \
    && python3 /opt/vonk/build/patch-mtp-drafter-quant.py \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
