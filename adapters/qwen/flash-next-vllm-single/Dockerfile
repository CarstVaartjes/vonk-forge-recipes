FROM docker.io/vllm/vllm-openai@sha256:3b0e188ffceb3d07e09c3cb5215433a0020eacf02d7f882ed3a8bfd15454477e

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark" \
      org.opencontainers.image.revision="7d0712dcb83cc58b467235e24d3a4429ed14518b" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

COPY patch_ple_layer.py patch_modelopt_mxfp8.py patch_qsa_fp8_kv.py patch_determinism.py patch_mtp_draft_vocab.py patch_block_drop.py patch_ple_offload.py build_ple_packed_table.py apply_patches.py /opt/vonk/build/
COPY draft-vocab /opt/vonk/draft-vocab
COPY chat_template.froggeric.jinja LICENSE.upstream /opt/vonk/
COPY --chmod=0755 qwen38-single-serve.py /opt/vonk/bin/vllm

RUN python3 /opt/vonk/build/apply_patches.py \
    && install --directory --mode=0755 /outputs \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && chown -R 10001:10001 /outputs /opt/vonk

ENV VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600

WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
