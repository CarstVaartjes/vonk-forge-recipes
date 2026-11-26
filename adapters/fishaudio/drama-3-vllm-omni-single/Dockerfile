FROM vllm/vllm-omni@sha256:6da195b87ca635601559ea11a6dd175758b0fa68cbada3a4529e01f36bc92df1

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm-omni" \
      org.opencontainers.image.revision="5917730a4b02aedb44d374173cdc1b10484ccd52" \
      org.opencontainers.image.version="vllm-omni 0.32.0" \
      io.vonk.upstream-image="vllm/vllm-omni:v0.32.0 (the last release that serves Drama 3: its DAC stage needs Model Runner V1, which 0.34 retired)"

# The creator's launch, `vllm serve fishaudio/drama-3 --omni` with its default
# fish_omni.yaml deploy config, with the recipe's --max-model-len and
# --gpu-memory-utilization (stage 0) and --codec-gpu-memory-utilization (stage 1)
# scoped per stage by the launcher.
COPY --chmod=0755 drama3-omni-serve.py /opt/vonk/bin/vllm
COPY fish_omni.yaml /opt/vonk/fish_omni.yaml
COPY verify-runtime.py /opt/vonk/verify-runtime.py
RUN python3 /opt/vonk/verify-runtime.py

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
