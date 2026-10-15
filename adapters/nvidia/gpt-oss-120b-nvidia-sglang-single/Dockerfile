FROM docker.io/lmsysorg/sglang@sha256:b0d8718a4424bb22e448e04407ab3ce5f7399a4c5fc702d6fbe36c3772ec8862

LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.version="0.5.20" \
      io.vonk.upstream-image="lmsysorg/sglang:latest-cu130 as named by the playbook (resolves to v0.5.20, build 94602c9c)"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
COPY verify-runtime.py /opt/vonk/verify-runtime.py
COPY gpt-oss-chat-template.jinja /opt/vonk/gpt-oss-chat-template.jinja

RUN echo "1be6557bc6b612a6a1481ea4d02d3279f312d765fbd6be7fea6a565809fe69fd  /opt/vonk/gpt-oss-chat-template.jinja" | sha256sum -c - \
    && python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
