FROM docker.io/lmsysorg/sglang@sha256:7b7ede416bebd9ae0ba8bd9009614ba3cc78338ec01941a4f624876ab9558007

LABEL org.opencontainers.image.source="https://github.com/sgl-project/sglang" \
      org.opencontainers.image.version="1.0.0" \
      io.vonk.upstream-image="lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3 as named by the Nemotron Omni model card"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
COPY verify-runtime.py /opt/vonk/verify-runtime.py
# The model card installs librosa at container start; a Vonk container runs offline, so it is installed at image build.
RUN pip install --no-cache-dir librosa

RUN python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
