FROM docker.io/vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce

LABEL ai.vonkforge.runtime-interface="v1"

# GLM-5.3-Flash (glm5_next, NoPE MLA: qk_rope_head_dim=0) on DGX Spark SM121.
# The day-0 image's only capability-12 sparse-MLA backend is
# FLASHINFER_MLA_SPARSE_SM120, whose packed fp8_ds_mla cache layout hard-requires
# DeepSeek's pe_dim=64 and dies in warmup ("pe_dim must be 64 for fp8_ds_mla").
# The SM90 NoPE backend (FlashInfer BatchMLAPagedAttentionWrapper, plain bf16
# cache) supports kpe=0, and its FA2 path passed a direct probe on SM121 with
# GLM's real shape (32 heads/rank, ckv 512, kpe 0, page_size 1). Extend it to
# capability 12, select FA2 off-Hopper, and scope the FlashInfer>=0.6.18
# feature gate to the fp8-cache path it actually protects.
COPY patch_sm121_nope_mla.py /tmp/patch_sm121_nope_mla.py
RUN python3 /tmp/patch_sm121_nope_mla.py && rm /tmp/patch_sm121_nope_mla.py

# Exact upstream v3-v5 dependency corrections, kept in the source build rather
# than relying on an intermediate radixark image: FlashInfer 0.6.18 nightly,
# NCCL 2.30.7 for Spark RoCE, and Cutlass DSL 4.6.2.
RUN pip install -q --pre \
      flashinfer-python==0.6.18.dev20260819 \
      flashinfer-cubin==0.6.18.dev20260819 \
      --extra-index-url https://flashinfer.ai/whl/nightly/ \
    && pip uninstall -q -y flashinfer-jit-cache \
    && pip install -q nvidia-nccl-cu13==2.30.7 \
    && pip install -q nvidia-cutlass-dsl==4.6.2 \
    && pip show nvidia-nccl-cu13 | grep -q "Version: 2.30.7" \
    && pip show nvidia-cutlass-dsl | grep -q "Version: 4.6.2"

# Continue the public current v1 -> v8 source stack.  These are the exact
# upstream v6/v7/v8 transformations, rebased onto the digest-pinned day-0
# image above so the Controller owns the complete build closure.
COPY patch_sm121_pdl.py /tmp/patch_sm121_pdl.py
RUN python3 /tmp/patch_sm121_pdl.py && rm /tmp/patch_sm121_pdl.py

COPY patch_v7.py /tmp/patch_v7.py
RUN python3 /tmp/patch_v7.py && rm /tmp/patch_v7.py

COPY patch_v8_fp8.py /tmp/patch_v8_fp8.py
RUN python3 /tmp/patch_v8_fp8.py && rm /tmp/patch_v8_fp8.py

# Apply the public current DFlash2 overlay after the v8 runtime fixes.
ARG VLLM=/usr/local/lib/python3.12/dist-packages/vllm
COPY overlay-dflash2/qwen3_dflash2.py ${VLLM}/model_executor/models/qwen3_dflash2.py
COPY overlay-dflash2/dflash2/ ${VLLM}/v1/worker/gpu/spec_decode/dflash2/
COPY overlay-dflash2/patch_registry_and_select.py /tmp/patch_registry_and_select.py
COPY overlay-dflash2/patch_glm_aux_capture.py /tmp/patch_glm_aux_capture.py
COPY overlay-dflash2/patch_kv_page_lcm2.py /tmp/patch_kv_page_lcm2.py
COPY overlay-dflash2/patch_glm5_drafter_group.py /tmp/patch_glm5_drafter_group.py
RUN python3 /tmp/patch_registry_and_select.py \
    && python3 /tmp/patch_glm_aux_capture.py \
    && python3 /tmp/patch_kv_page_lcm2.py \
    && python3 /tmp/patch_glm5_drafter_group.py \
    && python3 -c "from vllm.model_executor.models.registry import ModelRegistry; assert 'DFlash2DraftModel' in ModelRegistry.get_supported_archs()"

COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY verify-runtime.py /opt/vonk/verify-runtime.py
COPY chat_template_mm.jinja /opt/vonk/templates/glm53-chat-template-mm.jinja

RUN python3 /opt/vonk/verify-runtime.py \
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
