# sfxnz's GLM-5.3-Flash NVFP4 two-Spark kit (072c3f68, MIT): the image chain glm53-sm121-v8 -> v9 -> v10 -> v11 -> v13 that
# the kit documents ("Build the image") and that run.sh runs as the local tag glm53-sm121-v13. The kit publishes no image,
# so this Dockerfile builds the same chain in one file on the base the kit's FROM tag vllm/vllm-openai:glm53-flash-arm64-cu130 named when
# its dflash2_backport.diff was made (the 2026-08-26 push, digest below; the tag was re-pushed on 2026-09-09 with the
# backport already in it, which the diff cannot patch; kit-lock.json records this as an override). The kit's patch scripts and diff are vendored byte for byte under
# docker/. The two inline python heredocs of the kit's Dockerfile.sm121-v8 (NoPE MLA on sm_121, PDL gate) are the files
# patch_v8_sm121_nope_mla.py and patch_v8_sm121_pdl.py, unchanged; the kit's pip installs are requirements.lock, by hash.
# The layer on top adds the Controller placement wrapper at the platform's launcher path, the kit's chat template and the
# non-root user. Base: the linux/arm64 image vllm/vllm-openai, by digest.
FROM vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce

LABEL org.opencontainers.image.source="https://github.com/sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark" \
      org.opencontainers.image.revision="072c3f68fe425edb141e08cc60576f4bb0477ed2" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

ARG VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm

# v8: NoPE MLA on sm_121 prefers the SM90 bf16/FA2 sparse backend over SM120 packed pe_dim=64.
COPY patch_v8_sm121_nope_mla.py /tmp/patch_v8_sm121_nope_mla.py
RUN python3 /tmp/patch_v8_sm121_nope_mla.py && rm /tmp/patch_v8_sm121_nope_mla.py

# v8: FlashInfer 0.6.17 FA2 MLA NaNs on 64-256 row batches on sm_121, so the nightly 0.6.18 replaces it. The nightly also
# downgrades NCCL and skews cutlass-dsl; both are re-pinned in the same lock.
COPY requirements.lock /tmp/requirements.lock
RUN pip install -q --no-deps --require-hashes -r /tmp/requirements.lock \
    && pip uninstall -q -y flashinfer-jit-cache \
    && pip show nvidia-nccl-cu13 | grep -q "Version: 2.30.7" \
    && pip show nvidia-cutlass-dsl | grep -q "Version: 4.6.2" \
    && python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)" \
    && rm /tmp/requirements.lock

# v8: PDL is unvalidated on SM12x and races KDA state kernels.
COPY patch_v8_sm121_pdl.py /tmp/patch_v8_sm121_pdl.py
RUN python3 /tmp/patch_v8_sm121_pdl.py && rm /tmp/patch_v8_sm121_pdl.py

COPY docker/patch_v7.py /tmp/patch_v7.py
COPY docker/patch_v8_fp8.py /tmp/patch_v8_fp8.py
RUN python3 /tmp/patch_v7.py && python3 /tmp/patch_v8_fp8.py \
    && rm /tmp/patch_v7.py /tmp/patch_v8_fp8.py

# v9: DFlash2 backport, vLLM PR #52816 (merged upstream as b389ac2946, missing from this image's vLLM snapshot).
# dflash2_backport.diff is the PR's vllm/ changes regenerated with exact context against this tree, so --fuzz=0 refuses
# drift.
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

# v10: DFlash needs target hidden states via the EAGLE3 interface, which the fork-only Glm5Next model does not implement.
COPY docker/patch_v10_dflash_glm5.py /tmp/patch_v10_dflash_glm5.py
RUN python3 /tmp/patch_v10_dflash_glm5.py \
    && python3 -m compileall -q "$VLLM_ROOT/models/glm5next/nvidia/model.py" \
    && rm /tmp/patch_v10_dflash_glm5.py

# v11: the GLM-5-Next bespoke KV grouping rejects the DFlash draft's sliding-window layers; a dedicated draft KV group.
COPY docker/patch_v11_dflash_kv_groups.py /tmp/patch_v11_dflash_kv_groups.py
RUN python3 /tmp/patch_v11_dflash_kv_groups.py \
    && python3 -m compileall -q "$VLLM_ROOT/v1/core/kv_cache_utils.py" \
    && rm /tmp/patch_v11_dflash_kv_groups.py

# v13: opt-in runtime patches, Python only. Every behaviour sits behind a GLM53_* env var and is off by default.
COPY docker/patch_v13_misc.py /tmp/patch_v13_misc.py
RUN touch /tmp/v13.stamp \
    && python3 /tmp/patch_v13_misc.py "$VLLM_ROOT" \
    && rm /tmp/patch_v13_misc.py
COPY docker/patch_v13_fp8.py /tmp/patch_v13_fp8.py
RUN python3 /tmp/patch_v13_fp8.py "$VLLM_ROOT" \
    && rm /tmp/patch_v13_fp8.py
COPY docker/patch_v13_census.py /tmp/patch_v13_census.py
RUN python3 /tmp/patch_v13_census.py "$VLLM_ROOT" \
    && rm /tmp/patch_v13_census.py
COPY docker/patch_v13_verify.py /tmp/patch_v13_verify.py
RUN python3 /tmp/patch_v13_verify.py "$VLLM_ROOT" \
    && rm /tmp/patch_v13_verify.py
COPY docker/patch_v13_kpool_tail.py /tmp/patch_v13_kpool_tail.py
RUN python3 /tmp/patch_v13_kpool_tail.py "$VLLM_ROOT" \
    && rm /tmp/patch_v13_kpool_tail.py
RUN find "$VLLM_ROOT" -name '*.py' -newer /tmp/v13.stamp -print0 \
        | xargs -0 -r python3 -m compileall -q \
    && rm /tmp/v13.stamp

COPY LICENSE /opt/vonk/licenses/LICENSE.upstream
COPY chat_template.jinja /opt/vonk/templates/glm53-chat-template.jinja
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN { test -x /opt/vllm/.venv/bin/vllm || test -x /usr/local/bin/vllm; } \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
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
