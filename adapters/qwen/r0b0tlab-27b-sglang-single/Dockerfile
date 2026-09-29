FROM docker.io/lmsysorg/sglang@sha256:3c0abdf41ef22de9d7a859dc16ed71eae69452e36c91f071a25e60c85a6d1fc6

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/qwen38-27b-nvfp4-sm121-sglang" \
      org.opencontainers.image.revision="535e580eea3e777306629746c9ab45153b653a1a" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Base: the official lmsysorg/sglang Qwen3.8-27B day-0 image (arm64 digest of the
# 2026-08-17 SM121 campaign). Overlay: upstream docker/Dockerfile.dflash2 at the
# pinned commit, sgl-project/sglang PR #35371 @ c14312a plus r0b0tlab's compat layer.
COPY python/sglang/kernels/ops/speculative/dflash.py /sgl-workspace/sglang/python/sglang/kernels/ops/speculative/dflash.py
COPY python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py /sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/spec_aux_hidden_state.py
COPY python/sglang/srt/models/dflash.py /sgl-workspace/sglang/python/sglang/srt/models/dflash.py
COPY python/sglang/srt/speculative/dflash2_compat.py /sgl-workspace/sglang/python/sglang/srt/speculative/dflash2_compat.py
COPY python/sglang/srt/speculative/dflash_utils.py /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_utils.py
COPY python/sglang/srt/speculative/dflash_worker_v2.py /sgl-workspace/sglang/python/sglang/srt/speculative/dflash_worker_v2.py
COPY test/registered/unit/spec/test_dflash_logits.py /sgl-workspace/sglang/test/registered/unit/spec/test_dflash_logits.py
COPY test/registered/unit/model_executor/model_runner_components/test_spec_aux_hidden_state.py /sgl-workspace/sglang/test/registered/unit/model_executor/model_runner_components/test_spec_aux_hidden_state.py

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/state XDG_CACHE_HOME=/outputs/cache TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
