FROM ghcr.io/drowzeys/keys-qwen38-27b-nvfp4-dflash2-ablit-cyber-unlock-1m-yarn-single-dgxspark@sha256:fd31bd450510bebbb6d28623eb88ff8a86a6ac6c8db2a9009f21fafe8ff1eff9

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-Qwen3.8-27B-NVFP4-DFlash2-Ablit-Cybersecurity-Unlock-1M-Context-YARN-Single-DGXSpark" \
      org.opencontainers.image.revision="fca3d18bcab52b65e4a428b883cbcbcb0a7a3443" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: drowzeys' retag of AEON-7's aeon-vllm-ultimate (vLLM on GB10, arm64, CUDA 13);
# upstream's Dockerfile.prebuild records the AEON digest it wraps. The overlay files and
# launch scripts are upstream's, byte-identical: the chat template and drafter config
# that upstream bind-mounts, the YaRN hf-overrides, and serve_b5_*.sh / one-shot.sh for
# provenance. The recipe reproduces serve_b5_on_4.sh (1M seat).
COPY chat_template_b5_uncensored.jinja dflash2_config_yarn_1m.json yarn_hf_overrides.json /opt/vonk/overlay/
COPY serve_b5_on_4.sh serve_b5_16k.sh one-shot.sh Dockerfile.prebuild LICENSE RESPONSIBLE_USE.md CREDITS.md /opt/vonk/upstream/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN command -v vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch-extensions

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
