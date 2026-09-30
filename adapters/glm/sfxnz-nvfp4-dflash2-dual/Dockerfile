FROM docker.io/vllm/vllm-openai@sha256:b0501f99fec5136f248f78d5850977a2ec32d55cd9a665f4a9ffef24cbdf7fe5

LABEL org.opencontainers.image.source="https://github.com/sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark" \
      org.opencontainers.image.revision="072c3f68fe425edb141e08cc60576f4bb0477ed2" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Upstream builds glm53-sm121-v8 -> v9 -> v10 -> v11 -> v13 as five local images
# (docker/Dockerfile.sm121-v*), each FROM the previous one, the first FROM the
# vllm/vllm-openai:glm53-flash-arm64-cu130 tag. The digest above is that tag
# (resolved 2026-09-29, pushed 2026-09-09). The layers below are upstream's, in
# upstream's order. Only the COPY source paths are adapted to this build context,
# and upstream's two inline python heredocs in docker/Dockerfile.sm121-v8 (NoPE MLA
# backend, PDL gate) sit in docker/patch_v8_*.py with the same text, because the
# catalog reads any line starting with `from` as a FROM instruction. The other
# patch scripts are byte-identical to upstream's docker/ files.

# ---- glm53-sm121-v8 ----
# NoPE MLA on sm_121: prefer the SM90 bf16/FA2 sparse backend over SM120 packed pe_dim=64.
COPY docker/patch_v8_sm121_nope_mla.py /tmp/patch_v8_sm121_nope_mla.py
RUN python3 /tmp/patch_v8_sm121_nope_mla.py \
    && rm /tmp/patch_v8_sm121_nope_mla.py

# 0.6.17 FA2 MLA NaNs on 64-256 row batches on sm_121. Nightly also downgrades NCCL
# and skews cutlass-dsl; re-pin both in the same layer.
RUN pip install -q --pre \
      flashinfer-python==0.6.18.dev20260819 \
      flashinfer-cubin==0.6.18.dev20260819 \
      --extra-index-url https://flashinfer.ai/whl/nightly/ \
    && pip uninstall -q -y flashinfer-jit-cache \
    && pip install -q nvidia-nccl-cu13==2.30.7 nvidia-cutlass-dsl==4.6.2 \
    && pip show nvidia-nccl-cu13 | grep -q "Version: 2.30.7" \
    && pip show nvidia-cutlass-dsl | grep -q "Version: 4.6.2" \
    && python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

# PDL is unvalidated on SM12x and races KDA state kernels.
COPY docker/patch_v8_pdl_off.py /tmp/patch_v8_pdl_off.py
RUN python3 /tmp/patch_v8_pdl_off.py \
    && rm /tmp/patch_v8_pdl_off.py

COPY docker/patch_v7.py /tmp/patch_v7.py
COPY docker/patch_v8_fp8.py /tmp/patch_v8_fp8.py
RUN python3 /tmp/patch_v7.py && python3 /tmp/patch_v8_fp8.py \
    && rm /tmp/patch_v7.py /tmp/patch_v8_fp8.py

# ---- glm53-sm121-v9 ----
# DFlash2 backport: vLLM PR #52816 (merged upstream as b389ac2946, missing from
# this image's vLLM snapshot). dflash2_backport.diff is the PR's vllm/ changes
# regenerated with exact context against this tree, so --fuzz=0 refuses drift.
COPY docker/dflash2_backport.diff /tmp/dflash2_backport.diff
RUN cd /usr/local/lib/python3.12/dist-packages \
    && patch -p1 --fuzz=0 --no-backup-if-mismatch < /tmp/dflash2_backport.diff \
    && python3 -m compileall -q \
        vllm/config/vllm.py \
        vllm/model_executor/layers/logits_processor.py \
        vllm/model_executor/models/qwen3_dflash.py \
        vllm/model_executor/models/qwen3_dflash2.py \
        vllm/model_executor/models/registry.py \
        vllm/v1/worker/gpu/sample/gumbel.py \
        vllm/v1/worker/gpu/spec_decode/ \
    && grep -q '"DFlash2DraftModel"' vllm/model_executor/models/registry.py \
    && rm /tmp/dflash2_backport.diff

# ---- glm53-sm121-v10 ----
# DFlash needs target hidden states via the EAGLE3 interface, which the
# fork-only Glm5Next model does not implement. patch_v10 adds the capture with
# the SGLang-verified mHC contraction semantics.
COPY docker/patch_v10_dflash_glm5.py /tmp/patch_v10_dflash_glm5.py
RUN python3 /tmp/patch_v10_dflash_glm5.py \
    && python3 -m compileall -q \
        /usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/model.py \
    && rm /tmp/patch_v10_dflash_glm5.py

# ---- glm53-sm121-v11 ----
# The GLM-5-Next bespoke KV grouping rejects the DFlash draft's sliding-window
# layers, falling back to a generic path that cannot unify the indexer page.
# patch_v11 adds a dedicated draft KV group to the bespoke layout.
COPY docker/patch_v11_dflash_kv_groups.py /tmp/patch_v11_dflash_kv_groups.py
RUN python3 /tmp/patch_v11_dflash_kv_groups.py \
    && python3 -m compileall -q \
        /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py \
    && rm /tmp/patch_v11_dflash_kv_groups.py

# ---- glm53-sm121-v13 ----
# v13: opt-in runtime patches, Python only. Every behaviour sits behind a
# GLM53_* env var and is off by default, so with no GLM53_* set v13 serves
# like v11. docker/README-v13.md lists the switches and their GPU checks.
ARG VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm

# Router fp32, indexer workspace factor, mHC warmup, KDA .contiguous() trims,
# MTP-weight skip, DFlash prefix-cache fix.
COPY docker/patch_v13_misc.py /tmp/patch_v13_misc.py
RUN touch /tmp/v13.stamp \
    && python3 /tmp/patch_v13_misc.py "$VLLM_ROOT" \
    && rm /tmp/patch_v13_misc.py

# FP8 weight-only lane (patch_v13_fp8.py is owned by that lane).
COPY docker/patch_v13_fp8.py /tmp/patch_v13_fp8.py
RUN python3 /tmp/patch_v13_fp8.py "$VLLM_ROOT" \
    && rm /tmp/patch_v13_fp8.py

# Routed-expert census, measurement only (GLM53_EXPERT_CENSUS; tools/README.md).
COPY docker/patch_v13_census.py /tmp/patch_v13_census.py
RUN python3 /tmp/patch_v13_census.py "$VLLM_ROOT" \
    && rm /tmp/patch_v13_census.py

# Fixed-shape adaptive verification (GLM53_ADAPTIVE_VERIFY; README-v13.md).
COPY docker/patch_v13_verify.py /tmp/patch_v13_verify.py
RUN python3 /tmp/patch_v13_verify.py "$VLLM_ROOT" \
    && rm /tmp/patch_v13_verify.py

# Tail ring sized for the verify window (GLM53_KPOOL_TAIL_FIX; README-v13.md).
COPY docker/patch_v13_kpool_tail.py /tmp/patch_v13_kpool_tail.py
RUN python3 /tmp/patch_v13_kpool_tail.py "$VLLM_ROOT" \
    && rm /tmp/patch_v13_kpool_tail.py

# Byte-compile exactly the files the patch layers wrote or created.
RUN find "$VLLM_ROOT" -name '*.py' -newer /tmp/v13.stamp -print0 \
        | xargs -0 -r python3 -m compileall -q \
    && rm /tmp/v13.stamp

# ---- Vonk packaging ----
# Upstream run.sh bind-mounts chat_template.jinja and passes --chat-template; the
# recipe bakes the same byte-identical file and passes this path.
COPY chat_template.jinja /opt/vonk/templates/glm53-chat-template.jinja
COPY LICENSE /opt/vonk/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

# Upstream's GLM53_* switches are read at import time and are off unless set;
# the recipe sets them (and its options change them) through runtime.environment.
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_DISABLE_VERSION_CHECK=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
