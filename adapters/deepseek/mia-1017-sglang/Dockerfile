# Arm64 sibling of the 0xSero pin (lmsysorg/sglang:dev-dsv41, linux/amd64
# digest sha256:c4ca651192e57e91989b5176c3665148131b9a171e53861dee87f5e57cef25b5).
# Pinned by digest: the dev-dsv41 tag moves (on 2026-09-24 it pointed at a different SGLang build,
# da64c5cbb, which one node had pulled while the others ran this one). This is the arm64-capable
# index every result in docs/ was measured on (image id 37939c26 on arm64). Keep BASE_IMAGE in
# .env.example / .env.tp4.example / start.sh identical.
FROM lmsysorg/sglang:dev-dsv41@sha256:3dbc313030a6ef2c5d7de8ecf48e9aece722694a82182cb618cc82b588816349
LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks" \
      org.opencontainers.image.revision="64b02dea8681fb1d8e3a659ea432a8e9e9ba3995" \
      ai.vonkforge.runtime-interface="v1"
WORKDIR /opt/dsv41
COPY adapter /opt/dsv41/adapter
RUN g++ -O2 -Wall -Wextra -Werror -std=c++17 -shared -fPIC -pthread \
    adapter/row_store.cpp -o adapter/librow_store.so
COPY runtime/flash_mla_sm120.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
# Tolerate the DeepSeek image placeholder token in message text. Upstream raises there,
# which turns any transcript carrying the token into a self-sustaining HTTP 500 loop on
# the Anthropic endpoint. Idempotent, and it fails the build if the base image's
# encoding_dsv41.py no longer matches the anchor.
COPY runtime/patch_encoding_dsv41.py /opt/dsv41/runtime/patch_encoding_dsv41.py
RUN python3 /opt/dsv41/runtime/patch_encoding_dsv41.py
COPY tests /opt/dsv41/tests
RUN PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_thinking_alias.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_max_new_tokens.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_loop_abort.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_history_repair.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_wo_a_w8.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_draft_head_fp8.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_draft_tau.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_folded_fence.py \
 && PYTHONPATH=/opt/dsv41/adapter python3 /opt/dsv41/tests/test_autotune_keep.py
RUN mkdir -p /opt/vonk/bin \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
COPY distributed.py /opt/vonk/bin/distributed.py
ENV PYTHONPATH=/opt/dsv41/adapter HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    DSV41_SOURCE=/models/target OFFLOAD_MODE=nvme \
    STATE_PATH=/outputs/cache/deepseek-1017-sglang \
    B12X_COMPILE_CACHE_DIR=/outputs/cache/deepseek-1017-sglang/b12x \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache TMPDIR=/outputs/cache/tmp
USER 10001:10001
ENTRYPOINT []
