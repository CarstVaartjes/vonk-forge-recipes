# DeepSeek-V4.1-Flash EXL3 on SM121 (NVIDIA GB10 / DGX Spark).
#
# Base image is the official DeepSeek-V4.1-Flash vLLM build (DSpark, CSA2, Engram).
# We do NOT use the GLM-5.3-Flash glm53-flash-arm64-cu130 image: that tree has
# no DeepseekV41 architecture. GLM NoPE-MLA padding patches do not apply here.
#
# This overlay adds:
#   * native SM121 exllamav3_ext (fused exl3_moe + fat kernels compiled in)
#   * packed EXL3 mul1 loader (mixed K, experts + linears)
#   * operational GB10 patches (stops-in-reasoning, decode floor, spinwait,
#     xgrammar) that are model-agnostic
#
# E2 fat kernels are K4/MCG-only (unused here); the E3 v2 grouped kernels below cover mul1 K2/K3/K4.
# selects them. Fused exl3_moe is the decode path.
#
# Build (context = repo root, aarch64 host):
#   docker build -t dsv41-flash-exl3:local .

# linux/arm64 manifest of docker.io/vllm/vllm-openai:deepseekv41-flash-0909.
FROM docker.io/vllm/vllm-openai@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58

COPY overlay/patch_exl3_ext_aarch64.py /opt/dsv41/patch_exl3_ext_aarch64.py
COPY overlay/patch_exl3_fat_kernel.py /opt/dsv41/patch_exl3_fat_kernel.py
COPY overlay/exl3_fat_gemm.cu /opt/dsv41/exl3-fat-kernel/exl3_fat_gemm.cu
COPY overlay/exl3_fat_gemm.cuh /opt/dsv41/exl3-fat-kernel/exl3_fat_gemm.cuh
COPY overlay/exl3_fat_moe.cu /opt/dsv41/exl3-fat-kernel/exl3_fat_moe.cu
COPY overlay/exl3_fat_moe.cuh /opt/dsv41/exl3-fat-kernel/exl3_fat_moe.cuh
# v2 grouped fat-expert kernels (bits/codebook generic, separate gate/up inputs):
# built as the additive exl3_fat_moe_ext module after exllamav3_ext (no ext rebuild).
COPY overlay/e3v2/exl3_fat_moe.cu /opt/dsv41/e3v2/exl3_fat_moe.cu
COPY overlay/e3v2/exl3_fat_moe.cuh /opt/dsv41/e3v2/exl3_fat_moe.cuh
COPY overlay/build_exl3_fat_moe_ext.py /opt/dsv41/build_exl3_fat_moe_ext.py

# v0.0.43 fused exl3_moe TORCH_CHECKs mcg-only. This checkpoint is mul1.
# v1.4.5 instantiates cb1 (mcg) and cb2 (mul1) fused MoE kernels.
ARG EXLLAMAV3_COMMIT=e648f1a131365aae15920073e761a3fa5a527654
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV FLASHINFER_CUDA_ARCH_LIST=12.1a
ENV MAX_JOBS=8
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}

# Register EXL3 if this image's vLLM does not already list it.
COPY overlay/register_exl3_quantization.py /opt/dsv41/register_exl3_quantization.py
RUN python3 /opt/dsv41/register_exl3_quantization.py

# The vLLM image keeps CUDA toolkit headers under nvidia/cu13, not
# /usr/local/cuda/include. ExLlamaV3 ships AVX2/AVX512 CPU targets that
# do not compile on aarch64; stub them so the SM121 GEMM still builds.
# After pip install, import torch before exllamav3_ext so libc10.so is
# already mapped (this image does not set LD_LIBRARY_PATH during build).
RUN set -eux; \
    mkdir -p /tmp/exllamav3; \
    curl -fsSL "https://github.com/turboderp-org/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz" \
      | tar -xz -C /tmp/exllamav3 --strip-components=1; \
    python3 -c "from pathlib import Path; assert (Path('/tmp/exllamav3')/'exllamav3/modules/quant/exl3.py').is_file()"; \
    python3 /opt/dsv41/patch_exl3_ext_aarch64.py /tmp/exllamav3/exllamav3/exllamav3_ext; \
    python3 /opt/dsv41/patch_exl3_fat_kernel.py /tmp/exllamav3/exllamav3/exllamav3_ext /opt/dsv41/exl3-fat-kernel; \
    PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    export CPATH="${PY_SITE}/nvidia/cu13/include${CPATH:+:$CPATH}"; \
    export CPLUS_INCLUDE_PATH="${PY_SITE}/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"; \
    export C_INCLUDE_PATH="${PY_SITE}/nvidia/cu13/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"; \
    cd /tmp/exllamav3; \
    TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=8 \
      pip install --no-deps --no-build-isolation --no-cache-dir .; \
    python3 -c "import torch; import exllamav3_ext; assert hasattr(exllamav3_ext, 'exl3_moe'), dir(exllamav3_ext); assert hasattr(exllamav3_ext, 'exl3_fat_gemm'); print('exllamav3_ext', exllamav3_ext.__file__, 'exl3_moe=yes fat_gemm=yes')"; \
    rm -rf /tmp/exllamav3 /root/.cache/pip

# E3 v2 grouped MoE kernels for this checkpoint's K=3/K=2 mul1 experts (and K=4 mcg/mul1):
# the exllamav3_ext-embedded v1 kernels above are K4/MCG-only and are ignored by the
# overlay once this module (abi 2) is importable.
RUN set -eux; \
    PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    python3 /opt/dsv41/build_exl3_fat_moe_ext.py --src /opt/dsv41/e3v2 --out /tmp/e3v2 --arch 121a --install "$PY_SITE"; \
    python3 -c "import torch, exl3_fat_moe_ext as m; assert int(m.exl3_fat_moe_abi()) == 2; print('exl3_fat_moe_ext abi', int(m.exl3_fat_moe_abi()))"; \
    rm -rf /tmp/e3v2

# Python overlay AFTER the CUDA compile so edits do not rebuild exllamav3_ext.
COPY overlay/exl3.py /opt/dsv41/exl3.py
COPY overlay/patch_model_overrides.py /opt/dsv41/patch_model_overrides.py
COPY overlay/patch_exl3_packed_names.py /opt/dsv41/patch_exl3_packed_names.py
COPY overlay/patch_exl3_lm_head.py /opt/dsv41/patch_exl3_lm_head.py
COPY overlay/patch_engram_secondary.py /opt/dsv41/patch_engram_secondary.py
COPY overlay/patch_engram_file.py /opt/dsv41/patch_engram_file.py
COPY overlay/engram_layout.py /opt/dsv41/engram_layout.py
COPY overlay/engram_file_backend.py /opt/dsv41/engram_file_backend.py
COPY overlay/row_store.cpp /opt/dsv41/row_store.cpp
COPY overlay/patch_suppress_stops_in_reasoning.py /opt/dsv41/patch_suppress_stops_in_reasoning.py
COPY overlay/patch_scheduler_decode_floor.py /opt/dsv41/patch_scheduler_decode_floor.py
COPY overlay/patch_xgrammar_termination.py /opt/dsv41/patch_xgrammar_termination.py
COPY overlay/patch_spinwait.py /opt/dsv41/patch_spinwait.py
COPY overlay/patch_memory_log.py /opt/dsv41/patch_memory_log.py
COPY overlay/patch_h2d_stage.py /opt/dsv41/patch_h2d_stage.py
COPY overlay/patch_sm120_block64.py /opt/dsv41/patch_sm120_block64.py
COPY overlay/patch_responses_content_types.py /opt/dsv41/patch_responses_content_types.py
COPY files/chat_template.jinja /opt/dsv41/chat_template.jinja
COPY files/exl3_k_map.json /opt/dsv41/exl3_k_map.json
COPY tests/test_exl3_overlay.py /opt/dsv41/test_exl3_overlay.py
COPY tests/test_exl3_lm_head.py /opt/dsv41/test_exl3_lm_head.py
COPY tests/test_suppress_stops.py /opt/dsv41/test_suppress_stops.py
COPY tests/test_scheduler_decode_floor.py /opt/dsv41/test_scheduler_decode_floor.py
COPY tests/test_xgrammar_termination.py /opt/dsv41/test_xgrammar_termination.py
COPY tests/test_spinwait_patch.py /opt/dsv41/test_spinwait_patch.py
COPY tests/test_engram_secondary.py /opt/dsv41/test_engram_secondary.py
COPY tests/test_engram_layout.py /opt/dsv41/test_engram_layout.py
COPY tests/test_row_store.py /opt/dsv41/test_row_store.py
COPY tests/test_memory_log.py /opt/dsv41/test_memory_log.py
COPY tests/test_h2d_stage.py /opt/dsv41/test_h2d_stage.py
COPY tests/test_sm120_block64.py /opt/dsv41/test_sm120_block64.py
COPY tests/test_exl3_pretune.py /opt/dsv41/test_exl3_pretune.py
COPY tests/test_engram_dequant.py /opt/dsv41/test_engram_dequant.py

COPY overlay/install_exl3_overlay.py /opt/dsv41/install_exl3_overlay.py
RUN python3 /opt/dsv41/install_exl3_overlay.py

RUN python3 /opt/dsv41/patch_model_overrides.py
RUN python3 /opt/dsv41/patch_exl3_packed_names.py
RUN python3 /opt/dsv41/patch_exl3_lm_head.py
RUN python3 /opt/dsv41/patch_engram_secondary.py
RUN python3 /opt/dsv41/patch_engram_file.py
RUN g++ -O3 -shared -fPIC -pthread -o /opt/dsv41/librow_store.so /opt/dsv41/row_store.cpp
RUN python3 /opt/dsv41/patch_suppress_stops_in_reasoning.py || echo "WARN: suppress-stops patch skipped"
RUN python3 /opt/dsv41/patch_scheduler_decode_floor.py || echo "WARN: scheduler patch skipped"
RUN python3 /opt/dsv41/patch_xgrammar_termination.py || echo "WARN: xgrammar patch skipped (likely already upstream)"
RUN python3 /opt/dsv41/patch_spinwait.py --preflight || echo "WARN: spinwait preflight skipped"
RUN python3 /opt/dsv41/patch_h2d_stage.py
RUN python3 /opt/dsv41/patch_sm120_block64.py
# Upstream start.sh applies this one at container start; the recipe root is read-only.
RUN python3 /opt/dsv41/patch_responses_content_types.py
RUN python3 /opt/dsv41/patch_memory_log.py

RUN EXL3_SELFCHECK_GPU=0 python3 /opt/dsv41/test_exl3_overlay.py \
    && python3 /opt/dsv41/test_suppress_stops.py \
    && python3 /opt/dsv41/test_scheduler_decode_floor.py \
    && python3 /opt/dsv41/test_spinwait_patch.py \
    && python3 /opt/dsv41/test_engram_secondary.py \
    && python3 /opt/dsv41/test_exl3_lm_head.py \
    && python3 /opt/dsv41/test_engram_layout.py \
    && python3 /opt/dsv41/test_row_store.py \
    && python3 /opt/dsv41/test_memory_log.py \
    && python3 /opt/dsv41/test_h2d_stage.py \
    && python3 /opt/dsv41/test_sm120_block64.py \
    && EXL3_SELFCHECK_GPU=0 python3 /opt/dsv41/test_exl3_pretune.py \
    && python3 /opt/dsv41/test_engram_dequant.py
RUN python3 /opt/dsv41/test_xgrammar_termination.py \
    || echo "WARN: xgrammar test skipped (patch may already be upstream)"

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks" \
      org.opencontainers.image.revision="6f7d1590ad49a2b8995188e45d7b9db31e677452" \
      org.opencontainers.image.licenses="AGPL-3.0 AND MIT" \
      ai.vonkforge.runtime-interface="v1"

# Vonk Controller entrypoint: binds placement rendezvous, caches and the
# unprivileged service account to the platform contract. Upstream's SSH/NFS
# launcher (start.sh) is not used.
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
