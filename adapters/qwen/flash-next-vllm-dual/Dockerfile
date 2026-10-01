FROM docker.io/vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks" \
      org.opencontainers.image.revision="cd839d0b62f6c737178a0c2037733420e0524ebf" \
      org.opencontainers.image.licenses="AGPL-3.0-or-later AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

COPY patch_mtp_draft_vocab.py patch_mtp_draft_vocab_v030.py patch_qsa_fp8_kv_v030.py draft_vocab_en_code_47k.txt apply_patches.py /opt/vonk/build/
COPY draft-vocab/ /opt/vonk/draft-vocab/
COPY --chmod=0755 qwen38-vllm-wrapper.py /opt/vonk/bin/vllm
COPY LICENSE.upstream NOTICE.upstream /opt/vonk/

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
