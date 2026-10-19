FROM ghcr.io/ggml-org/llama.cpp@sha256:e5a2f80c0f1bdbb829d19b3ba3e84359acad95ea9fae2161661e08af46a31ce0

# Upstream shared libraries live beside the binary; Vonk runs from /tmp.
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"

LABEL org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp" \
      org.opencontainers.image.revision="620d2225cc436ab0b001e3200e590e3b286ef9ab" \
      org.opencontainers.image.version="0.5.0" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.upstream-image-tag="server-cuda13-v0.5.0"

COPY --chmod=0755 llama-server /opt/vonk/bin/llama-server
COPY --chmod=0755 verify-runtime.sh /opt/vonk/verify-runtime.sh

RUN /opt/vonk/verify-runtime.sh \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
