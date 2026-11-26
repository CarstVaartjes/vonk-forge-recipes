FROM ghcr.io/remsky/kokoro-fastapi-gpu@sha256:97a6f3dab7b3a9508f0fc1c08946ae90495344316f77291b1e99b654c14ea015

LABEL org.opencontainers.image.source="https://github.com/remsky/Kokoro-FastAPI" \
      org.opencontainers.image.revision="f2a5270d2ccd9590bc87e2393c774814f23fe7a1" \
      org.opencontainers.image.version="v0.10.0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.upstream-image="ghcr.io/remsky/kokoro-fastapi-gpu:v0.10.0 (linux/arm64, CUDA 12.9)"

# The creator's image runs `uvicorn api.src.main:app` from /app; the launcher starts the
# same app on the platform's host and port, with the weights read from the /models mount.
COPY --chmod=0755 kokoro-serve.py /opt/vonk/bin/pytorch-server
RUN /app/.venv/bin/python -m py_compile /opt/vonk/bin/pytorch-server

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
CMD []
