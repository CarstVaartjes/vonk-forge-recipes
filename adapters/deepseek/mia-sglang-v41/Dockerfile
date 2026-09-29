# The base is upstream's pin, lmsysorg/sglang:dev-dsv41@sha256:3dbc3130... (a multi-arch index);
# this is its linux/arm64 manifest. The dev-dsv41 tag moves, the digest does not.
FROM docker.io/lmsysorg/sglang@sha256:800cc9adea5be1e18f48185451220c4bc487c545b7095c720d2ccc9ba9bb3b5d

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks" \
      org.opencontainers.image.revision="cad252b7d3000cf21d919764e70a6912b6d3b0b0" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later AND Apache-2.0 AND MIT"

# Steps below follow the upstream Dockerfile at the pinned commit, including its in-image tests.
WORKDIR /opt/dsv41
COPY adapter /opt/dsv41/adapter
RUN g++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC -pthread \
    adapter/row_store.cpp -o adapter/librow_store.so
COPY runtime/flash_mla_sm120.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
COPY boot.py /opt/dsv41/boot.py
COPY scripts /opt/dsv41/scripts
COPY tests /opt/dsv41/tests
COPY benchmarks /opt/dsv41/benchmarks
RUN PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_thinking_alias.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_max_new_tokens.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_loop_abort.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_wo_a_w8.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_draft_head_fp8.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_draft_tau.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_folded_fence.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_autotune_keep.py

# Vonk packaging: the placement wrapper, a non-root user and the writable state directory.
COPY --chmod=0755 dsv41-serve /opt/vonk/bin/dsv41-serve
COPY verify-runtime.py /opt/dsv41/verify-runtime.py
RUN python3 /opt/dsv41/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV PYTHONPATH=/opt/dsv41/adapter \
    MODEL_PATH=/models/DeepSeek-V4.1-Flash \
    STATE_PATH=/state \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

EXPOSE 8888
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
