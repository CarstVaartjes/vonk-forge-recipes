FROM lmsysorg/sglang@sha256:b4a4745fab5393dc0aca573fe754b86477f57691c9e971390b21102d3d94ebf9
# Build: docker build -f docker/Dockerfile --build-arg SOURCE_REVISION=<clean SHA> .
# Upstream engine da64c5cbb8cf6bfd39be19da43573fdfd484c43a; no dependency refresh.
ARG SOURCE_REVISION
LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/dsv41-flash-tp4-sglang-sm121" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.r0b0tlab.upstream_revision="da64c5cbb8cf6bfd39be19da43573fdfd484c43a" \
      org.r0b0tlab.q200_sandbox_driver_version="r0b0tlab.qwen38.q200_sandbox_driver.v1"
COPY adapter/ /opt/dsv41/adapter/
COPY docker/q200/q200_sandbox_driver.py /opt/r0b0tlab/q200_sandbox_driver.py
RUN python3 -S -c "import hashlib,pathlib; p=pathlib.Path('/opt/r0b0tlab/q200_sandbox_driver.py'); assert hashlib.sha256(p.read_bytes()).hexdigest() == 'cac0513ca9bfcf5bbaa8ef55b0b14f1bda86262faf3ab9557f7a90d49aa6bccf'"
ENV PYTHONPATH=/opt/dsv41/adapter:${PYTHONPATH:-}
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=5 \
  CMD python3 -S /opt/dsv41/adapter/health.py || exit 1
# Required: sandbox driver stdout is exactly its JSON protocol (no NVIDIA banner).
ENTRYPOINT []
