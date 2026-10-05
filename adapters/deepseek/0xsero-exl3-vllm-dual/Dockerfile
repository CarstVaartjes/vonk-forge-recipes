# 0xSero DeepSeek-V4.1-Flash on two DGX Sparks (vLLM TP=2, EXL3 routed experts), Vonk Forge packaging.
# https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks at ee688184827e50f296bfdaf4215e04d3a4f942c3
#
# The kit ships no Dockerfile: scripts/launch.sh runs the prebuilt image below
# (its default IMG) with docker run flags and a vLLM argument list. This layer
# only adds the Controller placement wrapper at the platform's launcher path and
# the non-root user; the engine, the B12X kernels and the EXL3 MoE path are the
# image's own. Base: linux/arm64 manifest of
# ghcr.io/0xsero/deepseek-v4.1-flash-spark, by digest.
FROM ghcr.io/0xsero/deepseek-v4.1-flash-spark@sha256:5668e35e5ee021da4b9ce88a1964caf0e082e8ca01d81d1d64b6213a55c6add5

LABEL org.opencontainers.image.source="https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks" \
      org.opencontainers.image.revision="ee688184827e50f296bfdaf4215e04d3a4f942c3" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

COPY LICENSE /opt/vonk/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN test -x /usr/local/bin/vllm \
    && install --directory --mode=0755 /outputs \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && chown -R 10001:10001 /outputs /opt/vonk

ENV VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600

WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
