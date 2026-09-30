# syntax=docker/dockerfile:1.7
# 0xSero deepseek-spark: DeepSeek V4 Flash REAP (K160/K144) on one Spark.
# The published arm64 runtime image that upstream's installer pulls
# (cutlass451-g27, manifest digest pinned). Upstream applies its GB10 patcher at every
# container start (`python3 patch_vllm_k160_native.py && exec vllm serve ...`); the
# same byte-identical script is applied once here, at build time.
FROM ghcr.io/0xsero/deepseek-v4-flash-spark-vllm@sha256:e4462a915ba56026f9c7b5ed195180e07986983ac1aa26a8bb0160d7a031f396

LABEL org.opencontainers.image.source="https://github.com/0xSero/deepseek-spark" \
      org.opencontainers.image.revision="97fd6f02e26870416d3c6b450d4fa3217d9db124" \
      org.opencontainers.image.description="Offline Vonk adapter for the one-DGX-Spark DeepSeek V4 Flash REAP profiles" \
      ai.vonkforge.runtime-interface="v1"

COPY runtime/scripts/patch_vllm_reap_gb10.py /opt/vonk/build/patch_vllm_reap_gb10.py
RUN python3 /opt/vonk/build/patch_vllm_reap_gb10.py && rm -rf /opt/vonk/build

COPY --chmod=0755 vllm-wrapper.sh /opt/vonk/bin/vllm

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home \
           --shell /usr/sbin/nologin vonk; \
       fi \
    && install --directory --owner 10001 --group 10001 /state /outputs \
    && chmod -R a+rX /opt/vonk \
    && test -x /usr/local/bin/vllm

ENV HOME=/state \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
