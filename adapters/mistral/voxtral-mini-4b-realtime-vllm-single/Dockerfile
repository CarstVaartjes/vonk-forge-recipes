FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

LABEL org.opencontainers.image.source="https://github.com/vllm-project/recipes" \
      org.opencontainers.image.revision="8cbf0867ef903c11a50b5dc8c25785100f16d539" \
      org.opencontainers.image.version="vllm 0.32.0" \
      io.vonk.upstream-image="vllm/vllm-openai:v0.32.0 (a vLLM release newer than the 0.20.0 the Voxtral Realtime recipe requires)"

COPY --chmod=0755 vllm /opt/vonk/bin/vllm
COPY verify-runtime.py /opt/vonk/verify-runtime.py
# The vLLM recipe installs `mistral-common[audio]>=1.9.0` (soundfile and soxr) on top of vLLM. The immutable
# vLLM 0.32.0 base lacks vLLM's `audio` extra, so its packages are installed from a hash lock.
COPY requirements-audio.lock /opt/vonk/requirements-audio.lock
RUN pip install --no-cache-dir --no-deps --require-hashes -r /opt/vonk/requirements-audio.lock \
    && python3 /opt/vonk/verify-runtime.py

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
