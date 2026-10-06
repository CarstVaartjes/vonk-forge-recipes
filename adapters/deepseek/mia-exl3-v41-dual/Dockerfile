# MiaAI-Lab's published image of her DeepSeek-V4.1-Flash EXL3 two-Spark kit
# (https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks at 6f7d1590ad49a2b8995188e45d7b9db31e677452):
# vLLM (DSpark, CSA2, Engram) with her EXL3 overlay, native SM121 exllamav3_ext and fat-expert kernels, packed EXL3
# loader and GB10 patches. The kit's start.sh pulls tag 2.9bpw (public, no login); this recipe runs that image by digest
# and does not rebuild it from the kit's sources. This layer only adds the one overlay the kit applies at container
# start (the Responses API text-content patch; the platform's root filesystem is read-only), the Controller placement
# wrapper at the platform's launcher path, the licence texts and the non-root user.
# Base: the linux/arm64 image ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks, by digest.
FROM ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks@sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks" \
      org.opencontainers.image.revision="6f7d1590ad49a2b8995188e45d7b9db31e677452" \
      org.opencontainers.image.licenses="AGPL-3.0 AND MIT" \
      ai.vonkforge.runtime-interface="v1"

COPY upstream-LICENSE /opt/vonk/licenses/LICENSE
COPY upstream-LICENSE-MIT /opt/vonk/licenses/LICENSE.MIT

# Upstream start.sh applies this one at container start; the platform's root is read-only, so it is applied here.
COPY overlay/patch_responses_content_types.py /opt/dsv41/patch_responses_content_types.py
RUN python3 /opt/dsv41/patch_responses_content_types.py

# Controller entrypoint: binds placement rendezvous, caches and the unprivileged service account to the
# platform contract. Upstream's SSH/NFS launcher (start.sh) is not used.
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
