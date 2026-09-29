FROM docker.io/vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Muse-Glimmer-30B-DGX-Spark-RTX-5090-6000-PRO" \
      org.opencontainers.image.revision="1b71dc011010a3e2c373eafd8a41f12c7c7913fc" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      io.vonk.vllm.build-commit="ced6857afa0ea7b2e3f0846a62e1394e90f15607"

# Upstream installs a vLLM fork branch (xianbaoqian/vllm tiezhen/new-model-support) at run
# time. Its Muse Glimmer model, parsers and DFlash support were merged into vLLM
# (vllm-project/vllm#51655), so the pinned official 0.30.0 image serves the same launch profile.
COPY start.sh /opt/vonk/upstream/start.sh
COPY LICENSE.upstream /opt/vonk/upstream/LICENSE
COPY NOTICE.upstream /opt/vonk/upstream/NOTICE

RUN install --directory --mode=0755 /opt/vonk/bin \
    && printf '%s\n' '#!/bin/sh' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

USER 10001:10001
ENTRYPOINT []
