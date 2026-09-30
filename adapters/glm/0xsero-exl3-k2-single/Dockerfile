# syntax=docker/dockerfile:1.7
# 0xSero GLM-5.3-Flash EXL3 K2 2.0 bpw + native MTP on one Spark: the public
# arm64 runtime image that upstream's start.sh and .env.example select
# (b12x-mtp-1xspark-v2; the README table still names v1), pinned by manifest digest.
# Our layer adds upstream's MTP draft-view tool (byte-identical), the entrypoint
# wrapper that runs it, and the non-root user.
FROM ghcr.io/0xsero/glm53-b12x-exl3@sha256:1193a2e92d59f99257592f06ccf233cda4da019179fff7c02ac21eac242e0230

LABEL org.opencontainers.image.source="https://github.com/0xSero/GLM-5.3-Flash-EXL3-1x-DGX-Spark" \
      org.opencontainers.image.revision="6ff8595aa271f2eac49579fbf5e90cde9e90674e" \
      org.opencontainers.image.licenses="NOASSERTION" \
      org.opencontainers.image.description="Offline Vonk adapter for the one-DGX-Spark GLM-5.3-Flash EXL3 K2 native-MTP profile" \
      ai.vonkforge.runtime-interface="v1"

COPY --chmod=0755 tools/prepare_mtp_draft.py /opt/vonk/tools/prepare_mtp_draft.py
COPY --chmod=0755 vllm-wrapper.sh /opt/vonk/bin/vllm
COPY LICENSE /opt/vonk/LICENSE.glm53-exl3-k2-single

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home \
           --shell /usr/sbin/nologin vonk; \
       fi \
    && install --directory --owner 10001 --group 10001 /state /outputs \
    && chmod -R a+rX /opt/vonk \
    && python3 -c 'import vllm, vllm.entrypoints.openai.api_server' \
    && test -x /opt/vonk/bin/vllm

ENV HOME=/state \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
