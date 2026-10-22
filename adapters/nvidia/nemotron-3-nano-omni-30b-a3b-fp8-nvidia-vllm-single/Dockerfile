FROM docker.io/vllm/vllm-openai@sha256:871fc9b75a97dc6f58449bfb2aa30cab685c87a8bf6b4519a75c4760c1f8bcd8

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.version="0.20.0" \
      io.vonk.upstream-image="vllm/vllm-openai:v0.20.0 (named by the Nemotron playbook)"

COPY --chmod=0755 vllm /opt/vonk/bin/vllm
COPY verify-runtime.py /opt/vonk/verify-runtime.py
# The playbook installs the audio extra at container start ("pip install vllm[audio]"); a Vonk container runs
# offline, so the same install happens once at image build.
COPY requirements-audio.lock /opt/vonk/requirements-audio.lock
RUN pip install --no-cache-dir --require-hashes -r /opt/vonk/requirements-audio.lock \
    && python3 /opt/vonk/verify-runtime.py

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
