# GLM-5.3-Flash NoPE sparse MLA on SM121 (NVIDIA GB10 / DGX Spark).
#
# EXL3 weights, not NVFP4: this overlay is the MLA/KV geometry, not the GEMM.
# Do not pass --moe-backend marlin; that flag is the NVFP4 FLASHINFER_CUTLASS
# workaround and is the wrong backend for an EXL3 checkpoint.
#
# The stock image cannot serve this checkpoint on GB10. GLM-5.3-Flash is
# NoPE MLA (qk_rope_head_dim=0, d_qk=512), and on SM12x the only sparse-MLA backend is
# FLASHINFER_MLA_SPARSE_SM120, whose impl mandates the packed fp8_ds_mla record. Two
# things then reject it:
#
#   1. concat_and_cache_mla (csrc/.../cache_kernels.cu:866) -> "pe_dim must be 64 for
#      fp8_ds_mla", because the 656 B record is 512 NoPE + 16 B scales + 128 B RoPE.
#   2. flashinfer/mla/_core.py:583 -> the SM120 sparse entry requires kv_lora_rank=512,
#      qk_rope_head_dim=64 and query head dim 576.
#
# FlashInfer instantiates three SM120 sparse-MLA model types
# (flashinfer/mla/_sparse_mla_sm120.py:125-170): DSV3_2 and GLM_NSA at d_qk=576 /
# 656 B with topk in {128,512,1024,2048}, and DSV4 at d_qke=512 / 584 B (448 fp8 +
# 64 bf16 + 8 B footer) with topk in {128,512,1024} only. GLM-5.3-Flash needs
# d_qk=512 with index_topk=2048, which no kernel instantiates, and its uniform 512-dim
# latent does not fit the DSv4 448/64 split.
#
# So run it in the GLM_NSA geometry the GLM-5.2 line already uses: zero-pad the 512-dim
# latent to 576. A zero RoPE block adds nothing to the QK dot, softmax is unchanged, and
# the value is taken from the 512 NoPE region (d_v=512), so the result is exact. It costs
# 656 B/token instead of ~528 (~24% more DSA KV over 11 DSA layers) and needs no kernel
# work. MLAAttentionSpec.real_page_size_bytes already allocates block_size * 656 for
# fp8_ds_mla on non-deepseek_v4 models, so allocation already matches.
#
# The impl also inherits MLAAttentionImpl.forward_mha, which raises NotImplementedError
# (only the SM100 sibling gets a real one via SparseMLACommonImpl). Declaring
# supports_dense_mha_prefill=False routes prefill through the same top-k MQA kernel --
# the same declaration rocm_aiter_mla_sparse.py:668 makes -- which is the model's own
# sparse semantics, and it also clears the builder's prefill metadata so use_dense_mha
# can never select the missing path.
#
# Third, the SM120 impl passes index_topk as both the page-table width and the top-k,
# which this checkpoint breaks: index_kpool=4 widens the indexer's top-k buffer past
# index_topk and rounds it up to a multiple of 128 (2048 -> 2176), so the kernel rejects
# the table ("expects sparse block_tables shape (256, 1, 2048), got (256, 1, 2176)").
# The SM100 sibling treats sparse_mla_top_k as page-table *capacity* and bounds each
# query by its own valid count (flashinfer_mla_sparse.py:461-476), so ask the index
# converter for those counts and pass the buffer width. Zero-length rows are pointed at
# slot 0 with length 1 and their output zeroed afterwards, mirroring the same file's
# empty-row handling.
#
# Capacity alone is not enough on SM120: the top-k width is a kernel template parameter,
# not a runtime bound. sparse_mla_sm120_prefill.cu:220 rejects any GLM_NSA/DSV3_2 prefill
# with topk != 2048, its dual-cache (two-segment) dispatcher is DSV4-only (line 406), and
# the DSV4 dual path hardcodes a 128-wide primary. _DECODE_DSV3_2_DISPATCH likewise only
# instantiates topk in {128,512,1024,2048}. So the candidate set has to be exactly 2048
# wide, while this model's indexer emits up to 2051: 512 pools expanded by index_kpool=4,
# plus an always-selected tail of up to kpool-1 tokens that have no compressed pool entry
# yet (glm5next/nvidia/model.py:606-616, ops/kpool_compress.py:950-967).
#
# Fourth, the indexer's DeepGEMM paged-MQA page size. deepgemm attention.hpp:320 accepts
# block_kv in {32,64,128} on sm_10x but only 64 on sm_120 (non-FP4 cache), while
# vllm/utils/deep_gemm.py hardcodes PAGED_MQA_PAGE_SIZES = (32, 64) arch-blind. The kpool
# indexer's storage block is block_size // index_kpool, and cuda.py aligns block_size to
# index_kpool * min(PAGED_MQA_PAGE_SIZES) = 128; the hybrid KDA page constraint then
# lands on block_size 1664, giving a 416-entry storage block. 416 is not a multiple of
# 64, so compressed_kernel_block_size (v1/worker/utils.py:69) falls back to a 32-entry
# page and DeepGEMM asserts. Aligning to index_kpool * 64 = 256 moves block_size to 1792,
# whose 448-entry storage block does tile by 64.
#
# That raise also has to not change the MLA kernel page: select_kernel_block_size walks
# the backend's int sizes in descending order, so [64, 256] would pick 256 for a 1792
# manager block, and every GLM_NSA/DSV3_2 kernel is instantiated at PAGE_BLOCK_SIZE=64
# only (_DECODE_DSV3_2_PAGE_BLOCK_SIZE, launch_prefill_sg<..., 2048, 64>). At block_size
# 1664 that was masked -- 256 does not divide 1664 -- so pin the SM120 backend to [64].
#
# Drop the lowest-ranked selected pool instead of the tail: expand 511 pools (2044
# tokens) + 3 tail = 2047 candidates in a 2048-wide buffer. That gives up 4 of 2048
# candidate tokens (0.2%), the *last* ones in top-k rank order, and keeps the recent
# tail, which carries the highest attention weight in a causal model and can never be
# recovered from the pools. The alternative -- truncating the 2051-wide row to 2048 --
# drops exactly that tail. select_k stays 512, so the indexer keeps its fused top-k fast
# path (sparse_attn_indexer_kpool.py:810 gates on select_k in {512,1024,2048}).
#
# GB10 extras on top of the SM12x overlay: FlashInfer's sparse-MLA autotune and
# fused_moe gemm1/gemm2 autotune kill rank 0 on SM121, and PDL lowering races
# KDA state kernels here, so skip both and gate PDL to Hopper / SM10x.
#
# Build (context = repo root, aarch64 host):
#   docker build -t glm53-flash-sm121:local .

FROM docker.io/vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce

COPY patch_sm120_nope_mla.py /tmp/patch_sm120_nope_mla.py
RUN python3 /tmp/patch_sm120_nope_mla.py && rm /tmp/patch_sm120_nope_mla.py

COPY verify_sm120_nope_mla.py /tmp/verify_sm120_nope_mla.py
RUN python3 /tmp/verify_sm120_nope_mla.py && rm /tmp/verify_sm120_nope_mla.py

# EXL3/MCG trellis for routed experts (compiled for SM121, not SM120 cubins).
# Python overlay (exl3.py) is copied AFTER the CUDA compile so edits do not
# rebuild exllamav3_ext. The aarch64 stub patch must stay in this layer.
COPY overlay/patch_exl3_ext_aarch64.py /opt/glm53/patch_exl3_ext_aarch64.py
COPY overlay/patch_exl3_fat_kernel.py /opt/glm53/patch_exl3_fat_kernel.py
COPY overlay/exl3_fat_gemm.cu /opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cu
COPY overlay/exl3_fat_gemm.cuh /opt/glm53/exl3-fat-kernel/exl3_fat_gemm.cuh

ARG EXLLAMAV3_COMMIT=c5d9c657966ffeeaa9353f0cc899f18629da4a13
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV FLASHINFER_CUDA_ARCH_LIST=12.1a
ENV MAX_JOBS=8
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

COPY register_exl3.py /tmp/register_exl3.py
RUN python3 /tmp/register_exl3.py && rm /tmp/register_exl3.py

# The vLLM image keeps CUDA toolkit headers under nvidia/cu13 (cusparse.h), not
# /usr/local/cuda/include. ExLlamaV3 also ships AVX2/AVX512 CPU targets that
# do not compile on aarch64; stub them so the SM121 GEMM still builds.
RUN set -eux; \
    mkdir -p /tmp/exllamav3; \
    curl -fsSL "https://github.com/turboderp-org/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz" \
      | tar -xz -C /tmp/exllamav3 --strip-components=1; \
    python3 -c "from pathlib import Path; assert (Path('/tmp/exllamav3')/'exllamav3/modules/quant/exl3.py').is_file()"; \
    python3 /opt/glm53/patch_exl3_ext_aarch64.py /tmp/exllamav3/exllamav3/exllamav3_ext; \
    python3 /opt/glm53/patch_exl3_fat_kernel.py /tmp/exllamav3/exllamav3/exllamav3_ext /opt/glm53/exl3-fat-kernel; \
    export CPATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPATH:+:$CPATH}"; \
    export CPLUS_INCLUDE_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"; \
    export C_INCLUDE_PATH="/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"; \
    cd /tmp/exllamav3; \
    TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=8 \
      pip install --no-deps --no-build-isolation --no-cache-dir .; \
    python3 -c "import torch; import exllamav3_ext; assert hasattr(exllamav3_ext, 'exl3_moe'), dir(exllamav3_ext); assert hasattr(exllamav3_ext, 'exl3_fat_gemm'), dir(exllamav3_ext); assert hasattr(exllamav3_ext, 'exl3_fat_gemm_scatter'), dir(exllamav3_ext); print('exllamav3_ext', exllamav3_ext.__file__, 'exl3_moe=yes fat_gemm=yes')"; \
    rm -rf /tmp/exllamav3 /root/.cache/pip

# Keep this AFTER the CUDA compile layer so Python-only hook edits do not
# rebuild exllamav3_ext. Exl3Config.override_quantization_method requires
# "exl3" in ModelConfig's ordered overrides list.
COPY overlay/exl3.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
COPY overlay/patch_model_overrides.py /opt/glm53/patch_model_overrides.py
COPY overlay/qwen3_dflash2.py /opt/glm53/qwen3_dflash2.py
COPY overlay/dflash2_speculator.py /opt/glm53/dflash2_speculator.py
COPY overlay/patch_dflash2.py /opt/glm53/patch_dflash2.py
COPY overlay/patch_glm_eagle3.py /opt/glm53/patch_glm_eagle3.py
COPY overlay/patch_glm5_drafter_group.py /opt/glm53/patch_glm5_drafter_group.py
COPY tests/test_exl3_overlay.py /opt/glm53/test_exl3_overlay.py
COPY files/chat_template.jinja /opt/glm53/chat_template.jinja
COPY overlay/patch_glm_video_placeholders.py /opt/glm53/patch_glm_video_placeholders.py
COPY overlay/patch_suppress_stops_in_reasoning.py /opt/glm53/patch_suppress_stops_in_reasoning.py
COPY tests/test_suppress_stops.py /opt/glm53/test_suppress_stops.py
COPY overlay/patch_scheduler_decode_floor.py /opt/glm53/patch_scheduler_decode_floor.py
COPY tests/test_scheduler_decode_floor.py /opt/glm53/test_scheduler_decode_floor.py
COPY overlay/patch_hybrid_prefix_hit.py /opt/glm53/patch_hybrid_prefix_hit.py
COPY tests/test_hybrid_prefix_hit.py /opt/glm53/test_hybrid_prefix_hit.py
COPY overlay/patch_xgrammar_termination.py /opt/glm53/patch_xgrammar_termination.py
COPY tests/test_xgrammar_termination.py /opt/glm53/test_xgrammar_termination.py
COPY overlay/patch_kpool_tail_slotmap.py /opt/glm53/patch_kpool_tail_slotmap.py
COPY tests/test_kpool_tail_slotmap.py /opt/glm53/test_kpool_tail_slotmap.py
COPY overlay/patch_spinwait.py /opt/glm53/patch_spinwait.py
COPY tests/test_spinwait_patch.py /opt/glm53/test_spinwait_patch.py
COPY overlay/patch_indexer_workspace.py /opt/glm53/patch_indexer_workspace.py
COPY tests/test_indexer_workspace.py /opt/glm53/test_indexer_workspace.py
COPY overlay/ablit_runtime.py /opt/glm53/ablit_runtime.py
COPY overlay/patch_ablit.py /opt/glm53/patch_ablit.py
COPY tests/test_ablit.py /opt/glm53/test_ablit.py
COPY ablit/LAYER_MAP.json ablit/fetch_transplant.py ablit/refusal_direction_glm53_bf_oproj.pt ablit/refusal_direction_glm53_dealign_late.pt /opt/glm53/ablit/
RUN python3 /opt/glm53/patch_model_overrides.py
RUN python3 /opt/glm53/patch_dflash2.py
RUN python3 /opt/glm53/patch_glm_eagle3.py
RUN python3 /opt/glm53/patch_glm5_drafter_group.py
RUN python3 /opt/glm53/patch_suppress_stops_in_reasoning.py
RUN python3 /opt/glm53/patch_scheduler_decode_floor.py
RUN python3 /opt/glm53/patch_hybrid_prefix_hit.py
RUN python3 /opt/glm53/patch_xgrammar_termination.py
RUN python3 /opt/glm53/patch_kpool_tail_slotmap.py
# Applied unconditionally; the injected sizing reads GLM53_INDEXER_WORKSPACE
# at runtime and returns the stock expression unless it is "rightsize".
RUN python3 /opt/glm53/patch_indexer_workspace.py
RUN python3 /opt/glm53/patch_spinwait.py --preflight
RUN python3 /opt/glm53/patch_ablit.py

RUN EXL3_SELFCHECK_GPU=0 python3 /opt/glm53/test_exl3_overlay.py \
    && python3 /opt/glm53/test_suppress_stops.py \
    && python3 /opt/glm53/test_scheduler_decode_floor.py \
    && python3 /opt/glm53/test_hybrid_prefix_hit.py \
    && python3 /opt/glm53/test_xgrammar_termination.py \
    && python3 /opt/glm53/test_kpool_tail_slotmap.py \
    && python3 /opt/glm53/test_spinwait_patch.py \
    && python3 /opt/glm53/test_indexer_workspace.py \
    && python3 /opt/glm53/test_ablit.py

# Baked by start.sh --build-arg so a git pull that changes overlay/Dockerfile
# misses this label and rebuilds once. Keep last so stamp-only rebuilds are cheap.
ARG GLM53_RECIPE_STAMP=unknown
LABEL glm53.recipe.stamp=${GLM53_RECIPE_STAMP}
LABEL ai.vonkforge.runtime-interface="v1"

# Vonk Controller entrypoint: retain MiaAI-Lab's current source-built runtime
# while binding placement, cache paths, and the unprivileged service account to
# the platform contract.  The upstream SSH launcher is retained only as
# provenance in upstream-start.sh; it is never used by the recipe.
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY verify-runtime.py /opt/vonk/verify-runtime.py

RUN python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton /outputs/cache/tilelang

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TILELANG_CACHE_DIR=/outputs/cache/tilelang

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
