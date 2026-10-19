FROM docker.io/vllm/vllm-openai@sha256:c6c3c61579c905a3a0f3f10d8cd7f1df54a1b1c664807320699734bc5f014fb4
LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.revision="e68efc685ab1e6f1eda004ced7157e3ca411da2f"
RUN install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs
COPY --chmod=0755 launcher /opt/vonk/bin/vllm
COPY --chmod=0444 smoke.py /opt/vonk/decision-smoke.py
RUN python3 /opt/vonk/decision-smoke.py
ENV HOME=/state HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
