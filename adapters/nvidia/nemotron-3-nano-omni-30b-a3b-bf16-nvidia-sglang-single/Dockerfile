FROM docker.io/lmsysorg/sglang@sha256:7b7ede416bebd9ae0ba8bd9009614ba3cc78338ec01941a4f624876ab9558007

LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.version="1.0.1" \
      io.vonk.upstream-image="lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3 as named by the Nemotron Omni model card"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
COPY verify-runtime.py /opt/vonk/verify-runtime.py
COPY requirements-audio.lock /opt/vonk/requirements-audio.lock
# The pinned model card installs librosa at container start. Vonk serves
# offline, so install its Python 3.12 ARM64 closure during the image build.
RUN pip install --no-cache-dir --require-hashes -r /opt/vonk/requirements-audio.lock

RUN python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
