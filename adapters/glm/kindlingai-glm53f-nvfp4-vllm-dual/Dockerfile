# kindlingai/glm-5.3-flash-gx10 at commit 0f96475969a84c4fdef51f73680c5534b4f59487: GLM-5.3-Flash NVFP4 on vLLM with the DFlash2 drafter,
# the creator's kernel overlays and mentat in place of Ray. The creator publishes no image (compose runs a locally built spark-glm53:v29),
# so this file is the kit's image/Dockerfile built in CI from its pinned sources, with three changes:
#   * every FROM is a literal digest (the arm64 manifest of the kit's own tags): the vLLM nightly the creator preserved under its
#     namespace (the upstream tag was garbage-collected from Docker Hub), and the mentat artifacts image of the version the kit names;
#   * the compose overlays the creator bind-mounts over the image (arx, snapshot, adaptive-k with draft-trunc, fp8, megamoe, fixes, sp,
#     recoverssm) are copied to the same target paths after the kit's verify-base.py ran, in the order its launcher stacks them: the
#     platform has no recipe bind mounts, and the kit's own check runs against the image the overlays are mounted on;
#   * the Vonk layer replaces the kit's entrypoint: the launcher at /opt/vonk/bin/vllm starts the mentat daemon in rank 0 on the
#     rendezvous port and then vllm serve, the non-root user and the licence texts are added;
#     the image ENV puts the TileLang cache in /outputs/cache and the FlashInfer autotune cache in the run's fresh /tmp (the creator wipes it
#     every start: a persisted one deadlocks the next boot at TP>1).
# Dropped from the kit, as interface deviations: the status server on :8082 (an unauthenticated file search on 0.0.0.0, from the
# unlicensed mmastrac/spark-agent submodule) and the allocator counters it reads (TORCH_MEM_STATS_DIR=/logs), self-test.py,
# fabric-check.py, discover.sh and the entrypoint's preflight table and head election, the router mentatd-serve (the platform's gateway
# is the front door), dispram (a host daemon of kindling-spark-os) and fabric_ring.py (the ring-cabled TP=RING3/RING4 layouts; it
# assigns NCCL_IB_HCA, which the platform owns, and a two-box claim never runs it). The launcher keeps the preflight's page-cache release.
# FlashKDA is the only compile (98 s on a GX10 by the creator's measure; -gencode is passed, no GPU needed); arx, arxbig, megamoe, dense4
# compile at run time into the platform's persistent extension cache.
FROM docker.io/mmastrac/mentat-artifacts:0.17.1@sha256:27db96b9c0d298721db02725739194d1ab113a58f4672b42eadbb45b972f8e86 AS mentat

# GLM's KDA layers run chunked prefill through vllm/_flashkda_C on SM12x. The nightly builds FlashKDA b59532f, which rounds the recurrent
# state to bf16 after every 16-token tile; vllm#58846 bumps it to 17a037d, which keeps that state in fp32.
FROM docker.io/mmastrac/vllm-openai:nightly-ddd6fbca148a867aad1fcab7ec72f582b9977db4@sha256:ac8b63cdb18294ab4b9efa4a2f0a4474896fbaf9d48700512f8f2082b548c2c0 AS flashkda
ARG VLLM_REF=ddd6fbca148a867aad1fcab7ec72f582b9977db4
ARG FLASHKDA_REF=17a037d98da546deb4591e967cf961a43c034d8b
ARG CUTLASS_REF=5c149f52a436782210263fb2f19b354443a61c6a
COPY image/flashkda/ /build/
RUN FLASHKDA_REF=${FLASHKDA_REF} CUTLASS_REF=${CUTLASS_REF} VLLM_REF=${VLLM_REF} bash /build/build.sh

FROM docker.io/mmastrac/vllm-openai:nightly-ddd6fbca148a867aad1fcab7ec72f582b9977db4@sha256:ac8b63cdb18294ab4b9efa4a2f0a4474896fbaf9d48700512f8f2082b548c2c0

LABEL org.opencontainers.image.source="https://github.com/kindlingai/glm-5.3-flash-gx10" \
      org.opencontainers.image.revision="0f96475969a84c4fdef51f73680c5534b4f59487" \
      org.opencontainers.image.licenses="NOASSERTION" \
      ai.vonkforge.runtime-interface="v1"

# nvcc's include path is missing the headers whose libraries this base ships, which breaks any FlashInfer JIT path.
COPY image/patches/link_cuda_headers.sh /tmp/link_cuda_headers.sh
RUN bash /tmp/link_cuda_headers.sh && rm /tmp/link_cuda_headers.sh

# sm_121 sparse MLA: MiaAI's patcher (MIT, licence beside it) with its anchors retargeted to this nightly in a copy, then the plugin switch.
COPY image/patches/glm53-flash_SM121.py image/patches/LICENSE.MiaAI-Lab image/patches/mia_retarget.py /tmp/mia/
RUN python3 /tmp/mia/mia_retarget.py /tmp/mia/glm53-flash_SM121.py \
 && python3 /tmp/mia/glm53-flash_SM121.py && rm -rf /tmp/mia

COPY image/patches/gb10_plugin_backend.py /tmp/gb10_plugin_backend.py
RUN python3 /tmp/gb10_plugin_backend.py && rm /tmp/gb10_plugin_backend.py

COPY image/patches/sm90_fp8_kv_dtype.py /tmp/sm90_fp8_kv_dtype.py
RUN python3 /tmp/sm90_fp8_kv_dtype.py && rm /tmp/sm90_fp8_kv_dtype.py

# A per-worker memory ceiling: on unified memory an unbounded worker takes the node down rather than its own request.
COPY image/patches/spark_mem_trace.py /usr/local/lib/python3.12/dist-packages/spark_mem_trace.py
COPY image/patches/worker_memory_cap.py /tmp/worker_memory_cap.py
RUN python3 /tmp/worker_memory_cap.py && rm /tmp/worker_memory_cap.py

COPY image/patches/spin_wait.py /tmp/spin_wait.py
RUN python3 /tmp/spin_wait.py && rm /tmp/spin_wait.py

# DFlash2 reads auxiliary hidden states from target layers 5, 14, 24, 33, 42, which upstream's GLM5next does not expose.
COPY image/patches/glm53_eagle3_aux.py /tmp/glm53_eagle3_aux.py
RUN python3 /tmp/glm53_eagle3_aux.py && rm /tmp/glm53_eagle3_aux.py

# nvidia/GLM-5.3-Flash-NVFP4 stores its MTP layer in BF16 while its config declares every Linear NVFP4.
COPY image/patches/glm53_mtp_bf16.py /tmp/glm53_mtp_bf16.py
RUN python3 /tmp/glm53_mtp_bf16.py && rm /tmp/glm53_mtp_bf16.py

COPY image/patches/glm53_reasoning_always_parsed.py /tmp/glm53_reasoning_always_parsed.py
RUN python3 /tmp/glm53_reasoning_always_parsed.py && rm /tmp/glm53_reasoning_always_parsed.py

COPY image/patches/glm53_deterministic_topk.py /tmp/glm53_deterministic_topk.py
RUN python3 /tmp/glm53_deterministic_topk.py && rm /tmp/glm53_deterministic_topk.py

# vllm#58720 (ad6817b) landed after this nightly.
COPY image/patches/vllm-58720-routed-experts.patch /tmp/
RUN cd /usr/local/lib/python3.12/dist-packages \
 && patch -p1 --forward < /tmp/vllm-58720-routed-experts.patch \
 && rm /tmp/vllm-58720-routed-experts.patch

COPY --from=flashkda /out/_flashkda_C.abi3.so /out/FLASHKDA_REF /usr/local/lib/python3.12/dist-packages/vllm/

# DFlash2's sliding-window drafter layers need their own KV group on the GLM-5-Next layout.
COPY image/patches/glm53_dflash2_kv_groups.py /tmp/glm53_dflash2_kv_groups.py
RUN python3 /tmp/glm53_dflash2_kv_groups.py && rm /tmp/glm53_dflash2_kv_groups.py

# The fail-closed GLM tool parser, loaded with --tool-parser-plugin.
COPY image/patches/glm47_failclosed.py /usr/local/share/glm47_failclosed.py

# iproute2: the kit finds the ConnectX address with it; ibverbs is NCCL's RoCE data plane (missing, NCCL falls back to TCP);
# ripgrep, pciutils and usbutils are the kit's inventory tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
        iproute2 libibverbs1 librdmacm1 ibverbs-providers \
        ripgrep pciutils usbutils \
    && rm -rf /var/lib/apt/lists/* \
    && command -v ip && command -v rg

# mentat replaces Ray: the daemon binary, its machine probe (without it the agent registers zero GPUs) and the `ray` symlink, plus the
# pure-Python shim package vLLM's Ray executor talks to.
COPY --from=mentat /out/ /tmp/mentat-out/
RUN install -m755 /tmp/mentat-out/mentatd /usr/local/bin/mentatd \
    && install -m755 /tmp/mentat-out/mentatd-probe-machine /usr/local/bin/mentatd-probe-machine \
    && ln -sf /usr/local/bin/mentatd /usr/local/bin/ray \
    && pip install --no-cache-dir --no-deps /tmp/mentat-out/mentatd-*-py3-none-any.whl \
    && rm -rf /tmp/mentat-out \
    && ray --version

COPY image/verify-base.py /tmp/verify-base.py
RUN python3 /tmp/verify-base.py && rm /tmp/verify-base.py

# The creator's compose overlays, baked in at the paths it bind-mounts them to, in its launcher's order; later files win (sp and
# recoverssm both mount fixes/model.py).
ARG VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm
# arx
COPY experimental/arx/arx.py experimental/arx/arx_vllm.cu experimental/arx/arx_site.h experimental/arx/arxbig.cu experimental/arx/cuda_communicator.py ${VLLM_ROOT}/distributed/device_communicators/
# snapshot
COPY experimental/snapshot/weight_snapshot.py experimental/snapshot/base_loader.py ${VLLM_ROOT}/model_executor/model_loader/
COPY experimental/snapshot/modelopt.py ${VLLM_ROOT}/model_executor/layers/quantization/modelopt.py
COPY experimental/snapshot/flashinfer_cutlass_moe.py ${VLLM_ROOT}/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py
# adaptive-k with draft-trunc
COPY experimental/adaptive-k/adaptive_k.py ${VLLM_ROOT}/v1/core/sched/adaptive_k.py
COPY experimental/draft-trunc/draft_trunc.py experimental/draft-trunc/rejection_sampler.py experimental/draft-trunc/speculator.py ${VLLM_ROOT}/v1/worker/gpu/spec_decode/
COPY experimental/draft-trunc/model_runner.py ${VLLM_ROOT}/v1/worker/gpu/model_runner.py
COPY experimental/draft-trunc/grouped_topk_router.py ${VLLM_ROOT}/model_executor/layers/fused_moe/router/grouped_topk_router.py
COPY experimental/draft-trunc/dflash2_speculator.py ${VLLM_ROOT}/v1/worker/gpu/spec_decode/dflash2/speculator.py
# fp8
COPY experimental/snapshot/dense_fp8.py ${VLLM_ROOT}/model_executor/model_loader/dense_fp8.py
COPY experimental/megamoe/megadense4.cu /opt/megamoe/megadense4.cu
# megamoe
COPY experimental/megamoe/megamoe_rm.cu experimental/megamoe/moe_prefill.cu /opt/megamoe/
COPY experimental/megamoe/megamoe_vllm.py ${VLLM_ROOT}/model_executor/layers/fused_moe/megamoe_vllm.py
# fixes
COPY experimental/fixes/_fa_common.py /usr/local/lib/python3.12/dist-packages/flashinfer/mla/_batch_mla/_backends/_fa_common.py
COPY experimental/fixes/attention.py ${VLLM_ROOT}/models/glm5next/common/attention.py
COPY experimental/fixes/tilelang_kernels.py ${VLLM_ROOT}/model_executor/kernels/mhc/tilelang_kernels.py
COPY experimental/fixes/mhc_tilelang.py ${VLLM_ROOT}/model_executor/kernels/mhc/tilelang.py
COPY experimental/fixes/flashinfer_mla_sparse_sm90.py experimental/fixes/gb10_sparse_mla.py ${VLLM_ROOT}/v1/attention/backends/mla/
COPY experimental/fixes/gdn_attn.py ${VLLM_ROOT}/v1/attention/backends/gdn_attn.py
COPY experimental/fixes/kda.py ${VLLM_ROOT}/models/glm5next/common/kda.py
COPY experimental/fixes/causal_conv1d.py ${VLLM_ROOT}/model_executor/layers/mamba/ops/causal_conv1d.py
COPY experimental/fixes/kv_cache_utils.py experimental/fixes/kv_cache_coordinator.py ${VLLM_ROOT}/v1/core/
COPY experimental/fixes/kv_cache_interface.py ${VLLM_ROOT}/v1/kv_cache_interface.py
COPY experimental/fixes/worker_utils.py ${VLLM_ROOT}/v1/worker/utils.py
COPY experimental/fixes/warmup.py ${VLLM_ROOT}/v1/worker/gpu/warmup.py
# sp and recoverssm
COPY experimental/fixes/model.py ${VLLM_ROOT}/models/glm5next/common/model.py
COPY experimental/fixes/recoverssm.py ${VLLM_ROOT}/models/glm5next/common/recoverssm.py

COPY image/chat-template.jinja /usr/local/share/glm53-chat-template.jinja
COPY image/patches/LICENSE.MiaAI-Lab /opt/vonk/licenses/LICENSE.MiaAI-Lab
COPY experimental/megamoe/LICENSE /opt/vonk/licenses/LICENSE.megamoe
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY ray_ready.py /opt/vonk/lib/ray_ready.py

# Every overlay's marker file is in place, and the launcher's executables exist.
RUN test -e ${VLLM_ROOT}/distributed/device_communicators/arx.py \
    && test -e ${VLLM_ROOT}/model_executor/model_loader/weight_snapshot.py \
    && test -e ${VLLM_ROOT}/v1/core/sched/adaptive_k.py \
    && test -e ${VLLM_ROOT}/model_executor/model_loader/dense_fp8.py \
    && test -e ${VLLM_ROOT}/model_executor/layers/fused_moe/megamoe_vllm.py \
    && test -e ${VLLM_ROOT}/v1/attention/backends/mla/gb10_sparse_mla.py \
    && test -e ${VLLM_ROOT}/models/glm5next/common/recoverssm.py \
    && test -x /usr/local/bin/vllm && test -x /usr/local/bin/ray && test -x /usr/local/bin/mentatd \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TILELANG_CACHE_DIR=/outputs/cache/tilelang \
    VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/flashinfer_autotune

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
