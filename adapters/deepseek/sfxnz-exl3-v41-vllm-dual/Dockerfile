# sfxnz DeepSeek-V4.1-Flash EXL3 for two DGX Sparks (vLLM TP=2), Vonk Forge packaging.
# https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark at 3de914682d0880c8e478712a60ada930339c22e4
#
# The build steps below are upstream's docker/Dockerfile verbatim (only the two
# patch COPY sources are prefixed with docker/ because our build context is the
# adapter directory). Upstream's run.sh bind-mounts docker/patch over
# /opt/dsv41-patch and sitecustomize.py at container start; the image already
# carries both copies, so this recipe bakes the same byte-identical files.
# Base: linux/arm64 image of docker.io/vllm/vllm-openai:deepseekv41-flash-0909, by digest.
FROM docker.io/vllm/vllm-openai@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58

LABEL org.opencontainers.image.source="https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark" \
      org.opencontainers.image.revision="3de914682d0880c8e478712a60ada930339c22e4" \
      org.opencontainers.image.licenses="MIT AND AGPL-3.0" \
      ai.vonkforge.runtime-interface="v1"


COPY docker/patch /opt/dsv41-patch
RUN python3 /opt/dsv41-patch/apply_engram_disk.py \
      --engram /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py \
      --weights /usr/local/lib/python3.12/dist-packages/vllm/model_executor/model_loader/weight_utils.py \
 && python3 /opt/dsv41-patch/apply_engram_prestage.py \
      --engram /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/common/engram.py \
      --model-state /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4_1/nvidia/model_state.py

# Plugin source is cloned at build so the pin is in the RUN line, not a floating latest.
# Native p2b CUDA is parameterized by hidden/inter in 128-wide tiles. The
# host gate used to allow only 4096 x {1024,2048}; widen_p2b_shapes.py also
# allows V4.1 TP=2 5120x1152. coop + dataflow (DSV41_P2B_COOP=1|2; run.sh defaults to 2
# since round 35; unset or 0 runs the SORT=0/1 kernels, byte-identical to canonical-e13's)
# are the k3 round-3 stages that docker/Dockerfile.e14 layers on canonical-e13 as
# canonical-e14 (the round-3 serve arms' review-e14, same image ID).
ARG VLLM_EXL3_REF=d3cfd394920360d69f820d2dc96f8292a9e10283
ARG EXLLAMAV3_REF=5be886578ec80324c2c715269387be2058724b6e
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      git libcusparse-dev-13-0 libcusolver-dev-13-0 \
 && git clone https://github.com/turboderp-org/exllamav3.git /opt/exllamav3 \
 && git -C /opt/exllamav3 checkout "$EXLLAMAV3_REF" \
 && python3 /opt/dsv41-patch/patch_exllamav3_aarch64.py /opt/exllamav3/exllamav3/exllamav3_ext \
 && TORCH_CUDA_ARCH_LIST=12.1a python3 -m pip install --no-build-isolation --no-deps /opt/exllamav3 \
 && python3 -c "import torch, exllamav3, exllamav3_ext; assert hasattr(exllamav3_ext, 'exl3_moe'); print('exllamav3 ok')" \
 && git clone https://github.com/vcruz305/vllm-exl3.git /opt/vllm-exl3 \
 && git -C /opt/vllm-exl3 checkout "$VLLM_EXL3_REF" \
 && python3 /opt/dsv41-patch/fix_vllm_exl3_setup.py /opt/vllm-exl3/setup.py \
 && python3 /opt/dsv41-patch/widen_p2b_shapes.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_mrow.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_cfg1.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_codebook.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_fshift.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_srcsort.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_coop.py /opt/vllm-exl3 \
 && python3 /opt/dsv41-patch/widen_p2b_dataflow.py /opt/vllm-exl3 \
 && TORCH_CUDA_ARCH_LIST=12.1a python3 -m pip install --no-build-isolation --no-deps /opt/vllm-exl3 \
 && python3 -c "import torch, vllm_exl3, vllm_exl3_c; so = open(vllm_exl3_c.__file__, 'rb').read(); assert hasattr(vllm_exl3_c, 'p2b_fused_moe'); assert b'DSV41_P2B_SRC_SORT' in so and b'DSV41_P2B_COOP' in so and b'p2b coop dataflow kernel engaged' in so; print('vllm_exl3 cuda ok')" \
 && rm -rf /opt/exllamav3 /opt/vllm-exl3 /var/lib/apt/lists/*

ENV VLLM_PLUGINS=vllm_exl3
ENV VLLM_EXL3_MOE_KERNEL=native
ENV DSV41_ENGRAM_DISK=1
COPY docker/patch/sitecustomize.py /usr/lib/python3.12/sitecustomize.py
# E10: (16,64) tiles for decode-sized m in the flashinfer b12x dense GEMM
# (2x CTAs at n<=8192; no K-split exists in that kernel), and a boot log of
# the o_proj wo_a runtime dtype (pack stores F8; loader hands over bf16 ->
# torch.bmm -> sm_80 WMMA, the next decode target).
RUN python3 /opt/dsv41-patch/widen_b12x_smalls.py /usr/local/lib/python3.12/dist-packages/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py \
 && python3 /opt/dsv41-patch/probe_wo_a.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py \
 && python3 /opt/dsv41-patch/fix_o_proj_woa_fp8.py /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/ops/o_proj.py
# NOTE: the optional `pip install b12x` extra is deliberately NOT installed:
# with the package present, vLLM routes dense MXFP8 projections through the
# b12x package kernels and prose decode measured 5-9% slower than the
# flashinfer cute-dsl path at equal acceptance (2026-09-19 A/B, two 5-run
# batches per image). Revisit only with a proper kernel-level A/B.
# run.sh warns when IMAGE lacks this label (e.g. a stale local :latest).
LABEL dsv41.recipe.patches="shapes,mrow,cfg1,codebook,fshift,srcsort,coop,dataflow,b12x_smalls,fix_o_proj_woa_fp8,woa_prepack"

# Vonk Controller entrypoint: binds the placement rendezvous to vLLM's
# multi-node arguments and maps the recipe switches upstream run.sh handles in
# shell (SPEC=none, ENFORCE_EAGER=1, LANGUAGE_MODEL_ONLY=1). Upstream's
# SSH/docker launcher (run.sh) is not used.
COPY --chmod=0755 dsv41-vllm-wrapper.py /opt/vonk/bin/dsv41-vllm-serve
COPY LICENSE /opt/vonk/LICENSE.upstream

# Upstream's sitecustomize.py rewrites vLLM sources at interpreter start
# (idempotent: it writes only when the text changes). The Vonk container root is
# read-only and runs unprivileged, so apply the default rewrites once at build;
# at run time they report "already applied". The environment is upstream run.sh's
# default set. Failures here only print (DSV41_PATCH_STRICT is unset), exactly as
# in upstream's own docker build steps.
RUN env \
        VLLM_EXL3_MOE_KERNEL=native \
        DSV41_ENGRAM_DISK=1 \
        LANGUAGE_MODEL_ONLY=0 \
        MM_ENCODER_TP_MODE=data \
        DSV41_DSPARK_MARKOV_SCALE=1 \
        DSV41_STEP_CENSUS=0 \
        DSV41_INDEX_TOPK=0 \
        DSV41_MHC_DECODE_SPLITS=0 \
        DSV41_MHC_DET_SPLITS=16 \
        DSV41_MHC_DET_OVERLAP=0 \
        DSV41_ENGRAM_CACHE=0 \
        DSV41_ENGRAM_CENSUS=1 \
        DSV41_ENGRAM_FAST_STAGE=1 \
        DSV41_ENGRAM_STAGE_THREADS=16 \
        DSV41_ENGRAM_PREFETCH=1 \
        DSV41_ENGRAM_PF_DUMP=0 \
        DSV41_ENGRAM_PREFETCH_DEBUG=0 \
        DSV41_ENGRAM_CPU_HASH=0 \
        DSV41_ENGRAM_GATHER_V2=1 \
        DSV41_ENGRAM_WILLNEED=1 \
        DSV41_ENGRAM_WILLNEED_MIN_ROWS=512 \
        DSV41_ENGRAM_DEFER=0 \
        DSV41_ENGRAM_NATIVE_STAGE=1 \
        DSV41_ENGRAM_EARLY_HASH=1 \
        DSV41_ATTN_T2R_DEDUP=1 \
        DSV41_MOE_PREP_FUSED=1 \
        DSV41_CANDIDATE_MASK_BOUNDED=1 \
        DSV41_INDEXER_WP_GEMV=1 \
        DSV41_SWA_META_FUSED=1 \
        DSV41_ENGRAM_WKV_TP=0 \
        DSV41_LOAD_PF_G8=0 \
        DSV41_STREAM_FEED=1 \
        DSV41_MHC_NO_DEEPGEMM=0 \
        DSV41_DSPARK_DRAFT_TOPK=0 \
        DSV41_DSPARK_SPARSE_MARKOV=1 \
        DSV41_DSPARK_SPARSE_MARKOV_TOPK=1024 \
        DSV41_WOA_PREPACK=1 \
        DSV41_DSPARK_TAIL_NGRAM=0 \
        DSV41_DSPARK_TAIL_NGRAM_POS=3 \
        DSV41_DSPARK_SOFTMAX_VERIFY=0 \
        DSV41_DSPARK_REFINE_PASS=0 \
        DSV41_DSPARK_CONF_GATE=0 \
        DSV41_MLA_IO_WARPS=0 \
        DSV41_MLA_CHUNKS_PER_BLOCK=0 \
        DSV41_DENSE_DG_SMALLM=0 \
        DSV41_DENSE_GEMV=1 \
        DSV41_LMHEAD_MXFP8=1 \
        DSV41_DROP_PAGE_CACHE=1 \
        DSV41_INDEXER_PREFILL_FACTOR=1 \
        DSV41_PREFILL_EMPTY_CACHE_TOKENS=8192 \
        DSV41_PREFILL_EMPTY_CACHE_MEMAVAIL_GIB=2.5 \
        VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
        DSV41_P2B_COOP=2 \
        DSV41_ALLOW_CUDA_GRAPHS=1 \
        VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
        VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=256 \
        DSV41_NCCL_EAGER_TWIN=0 \
        DSV41_AR_L2_PREFETCH=0 \
        NCCL_MAX_NCHANNELS=8 \
        NCCL_BUFFSIZE=1048576 \
        NCCL_LL128_BUFFSIZE=262144 \
        NCCL_PROTO=^LL128 \
        NCCL_CROSS_NIC=1 \
        python3 -c "pass" \
 && chmod -R a+rX /opt/dsv41-patch \
 && install --directory --mode=0755 /outputs \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
 && chown -R 10001:10001 /outputs /opt/vonk

ENV VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600

WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
