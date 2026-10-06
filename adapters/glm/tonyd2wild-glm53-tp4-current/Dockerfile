# tonyd2wild's published image of his GLM-5.3-Flash NVFP4 4x DGX Spark kit (2ac4e8dd): vLLM glm53 day-0 image with the
# kit's v1-v9 SM121 patch stack and the DFlash2 overlay, tag sm121-v11-dflash2. The kit's 1M-context fp8/marlin
# launcher (launch-glm53-tp4-24g.sh) runs exactly this tag; this recipe runs that image, pinned by digest, and does not
# rebuild it from the kit's sources. This layer adds what that launcher mounts at run time (the kit's SM121 indexer
# patch, which the kit says the engine dies without past ~24K context, and its multimodal chat template), the
# Controller placement wrapper at the platform's launcher path, and the non-root user. Base: the linux/arm64 image
# ghcr.io/tonyd2wild/vllm-glm53-flash, by digest.
FROM ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark" \
      org.opencontainers.image.revision="2ac4e8dd661b12e6a1d7e69df711707abd6b9cd4" \
      ai.vonkforge.runtime-interface="v1"

# The kit bind-mounts this file over vLLM's kpool indexer (-v $HOME/patches/sparse_attn_indexer_kpool.py:...:ro).
COPY sparse_attn_indexer_kpool_sm121.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY chat_template_mm.jinja /opt/vonk/templates/glm53-chat-template-mm.jinja

RUN chmod 0644 /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py \
    && { test -x /opt/vllm/.venv/bin/vllm || test -x /usr/local/bin/vllm; } \
    && python3 -c "import ast, pathlib; ast.parse(pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py').read_text())" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --mode=0755 /inputs \
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
