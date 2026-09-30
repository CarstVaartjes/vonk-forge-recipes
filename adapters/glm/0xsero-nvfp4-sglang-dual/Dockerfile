# syntax=docker/dockerfile:1.7
# 0xSero glm-5.3-flash-sglang-sm121: the six GLM/SGLang SM121 patch files are laid
# over the digest-pinned lmsysorg/sglang GLM-5.3-Flash image exactly as upstream's
# Dockerfile does. Our additions are the placement wrapper and the non-root user;
# upstream's launch-spark.sh is carried byte-identically as reference only (the
# recipe arguments reproduce its `exec python3 -m sglang.launch_server`).
FROM docker.io/lmsysorg/sglang@sha256:3a97bd50034ca60c6e6c86b8e36a73675d261f6a5eb71197796aee5175409290

LABEL org.opencontainers.image.source="https://github.com/0xSero/glm-5.3-flash-sglang-sm121" \
      org.opencontainers.image.revision="dfb1bb7e45c20058d37df7a39cceda45a9d216a8" \
      org.opencontainers.image.base.digest="sha256:3a97bd50034ca60c6e6c86b8e36a73675d261f6a5eb71197796aee5175409290" \
      org.opencontainers.image.description="Offline Vonk adapter for the GLM-5.3-Flash NVFP4 two-Spark SGLang profile" \
      ai.vonkforge.runtime-interface="v1"

ENV CUDA_DEVICE_ORDER=PCI_BUS_ID \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    CUTE_DSL_ARCH=sm_121a \
    SGLANG_OPT_USE_TOPK_V2=0 \
    SGLANG_OPT_USE_TILELANG_INDEXER=1 \
    SGLANG_FP8_PAGED_MQA_LOGITS_TORCH=1 \
    SGLANG_OPT_DEEPGEMM_HC_PRENORM=0 \
    SGLANG_OPT_FP8_WO_A_GEMM=0

COPY patches/sglang-glm5_next-debug.py /sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py
COPY patches/sglang-deepseek_nextn-glm53.py /sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py
COPY patches/sglang-quant-utils-sm120.py /sgl-workspace/sglang/python/sglang/srt/layers/quantization/utils.py
COPY patches/sglang-modelopt-quant-sm120.py /sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py
COPY patches/sglang-flash_mla_sm120-glm53.py /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py
COPY patches/sglang-dsa_backend-glm53.py /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py
COPY chat-template-mm.jinja /opt/glm53/chat-template-mm.jinja
COPY --chmod=0755 launch-spark.sh /opt/glm53/launch-spark.sh
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve

RUN python3 -m py_compile \
      /sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py \
      /sgl-workspace/sglang/python/sglang/srt/models/deepseek_nextn.py \
      /sgl-workspace/sglang/python/sglang/srt/layers/quantization/utils.py \
      /sgl-workspace/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py \
      /sgl-workspace/sglang/python/sglang/kernels/ops/attention/flash_mla_sm120.py \
      /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa_backend.py \
    && chmod -R a+rX /opt/glm53 /sgl-workspace/sglang/python/sglang \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache \
    && test -x /opt/vonk/bin/sglang-serve

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
