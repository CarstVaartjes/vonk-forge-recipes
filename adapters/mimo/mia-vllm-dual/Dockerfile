FROM ghcr.io/miaai-lab/mimo-v2.5-vllm-dual-dgx-sparks@sha256:e37e2972f3e8eec2b326767293a99e61cbdc11c1551e2c481c7c49f5f2a31809

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/MiMo-V2.5-vLLM-Dual-DGX-Sparks" \
      org.opencontainers.image.revision="eb9c5fa61ed183a085f622dbb4c67f9fcb065a47"

# The base is MiaAI-Lab's published dev vLLM image (ARM64, CUDA 13.2, torch 2.11,
# system Python 3.12, tag 20260704) pinned by digest. It already carries part of
# the upstream mods. Upstream's start.sh applies the full set of mods and engine
# patches inside a running container at every bring-up; here the same scripts run
# once at build time, in the same order, so the image is read-only at run time.
# Like upstream, one script's non-zero exit does not stop the rest; the end state
# upstream asserts is then checked by verify-runtime.py, which fails the build.
# drop-caches (a root page-cache loop on the host) is not applied; see NOTICE. The
# ray-keep-node-nccl-hca file (keeps the head's NCCL_IB_HCA from overriding the
# worker's) goes to a config root the non-root runtime user can read.
COPY recipe /opt/vonk/build/recipe
COPY patches /opt/vonk/build/patches
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN set -u \
    && for mod in ray-keep-node-nccl-hca fix-prometheus-instrumentator-router fix-mimo-v2-vllm fix-mimo-mtp-patches fix-modelopt-mixed-mxfp8 nvfp4-kv-diffkv; do \
         echo "Applying mod: $mod"; \
         bash "/opt/vonk/build/recipe/mods/$mod/run.sh" || echo "[build] $mod returned non-zero (continuing, as upstream apply-mods.sh does)"; \
       done \
    && for patch in patch_mimo_v2_eagle3 patch_triton_noncausal patch_nc_fix patch_kv_page_lcm patch_aux_layer_off_by_one patch_diffkv_noncausal patch_draft_cache_auto patch_spec_dtype_guard patch_max_tokens_clamp patch_merge_multimodal_error; do \
         echo "Applying patch: $patch"; \
         python3 "/opt/vonk/build/patches/$patch.py" || echo "[build] $patch returned non-zero (continuing, as upstream start.sh does)"; \
       done \
    && install -D -m 0644 /root/.config/vllm/ray_non_carry_over_env_vars.json /opt/vonk/vllm-config/ray_non_carry_over_env_vars.json \
    && find /usr/local/lib/python3.12/dist-packages/vllm -name __pycache__ -type d -prune -exec rm -rf {} + \
    && python3 /opt/vonk/build/verify-runtime.py \
    && python3 -m compileall -q -j 0 /usr/local/lib/python3.12/dist-packages/vllm \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_CONFIG_ROOT=/opt/vonk/vllm-config \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
