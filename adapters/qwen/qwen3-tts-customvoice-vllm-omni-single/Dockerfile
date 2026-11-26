FROM vllm/vllm-omni@sha256:170d323159f5992ec11822b36fd9157b7299c7f4e4dc718c747e24fcd13758cc

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm-omni" \
      org.opencontainers.image.revision="06a28821a7f72bdb436c3df656b10f67fded0846" \
      org.opencontainers.image.version="vllm-omni 0.34.0" \
      io.vonk.upstream-image="vllm/vllm-omni:v0.34.0 (the release whose Qwen3-TTS online serving the creator's example launches)"

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
