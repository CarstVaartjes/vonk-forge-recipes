FROM vllm/vllm-omni@sha256:f713b909f16cf8c7901f17c3aee3f471cbee5fbde5603b7dbe2631630887d7c2

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm-omni" \
      org.opencontainers.image.revision="5917730a4b02aedb44d374173cdc1b10484ccd52" \
      org.opencontainers.image.version="vllm-omni 0.32.0" \
      io.vonk.upstream-image="vllm/vllm-omni:v0.32.0 (the release whose Qwen3-TTS online serving the creator's example launches)"

# The creator's launch, `vllm-omni serve <model> --deploy-config qwen3_tts.yaml --omni`, with the
# recipe's context setting scoped to the talker stage by the launcher.
COPY --chmod=0755 qwen3-tts-omni-serve.py /opt/vonk/bin/vllm
COPY qwen3_tts.yaml /opt/vonk/qwen3_tts.yaml
COPY verify-runtime.py /opt/vonk/verify-runtime.py
RUN python3 /opt/vonk/verify-runtime.py

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
