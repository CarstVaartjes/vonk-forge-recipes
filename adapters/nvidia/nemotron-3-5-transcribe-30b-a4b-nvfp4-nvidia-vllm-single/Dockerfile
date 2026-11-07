FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34

LABEL org.opencontainers.image.source="https://github.com/NVIDIA-NeMo/Speech" \
      org.opencontainers.image.revision="9b3fc483286e428e93d4426fb08599e89b8d46e3" \
      org.opencontainers.image.version="vllm 0.32.0" \
      io.vonk.upstream-image="vllm/vllm-openai:v0.32.0 (named by the model card: vLLM 0.32.0 and NeMo Speech main)"

COPY --chmod=0755 vllm /opt/vonk/bin/vllm
COPY verify-runtime.py /opt/vonk/verify-runtime.py
# The model card installs NeMo Speech with `uv sync --extra asr --extra vllm` and adds
# av==19.0.2. The immutable vLLM 0.32.0 base already supplies the Torch and vLLM stack,
# so only the packages it lacks are installed, at the versions NeMo's uv.lock pins.
COPY requirements-nemo.lock /opt/vonk/requirements-nemo.lock
RUN pip install --no-cache-dir --no-deps --require-hashes -r /opt/vonk/requirements-nemo.lock
# NeMo Speech main, pinned by commit and archive digest; it provides the nemo_speechlm vLLM plugin.
RUN curl -fsSL -o /tmp/nemo-speech.tar.gz https://codeload.github.com/NVIDIA-NeMo/Speech/tar.gz/9b3fc483286e428e93d4426fb08599e89b8d46e3 \
    && echo "2fdc00b8827cecd05586fb4d1bc24e00dbb195214ade797d157fa645f3ffe796  /tmp/nemo-speech.tar.gz" | sha256sum --check --strict - \
    && mkdir /tmp/nemo-speech \
    && tar -xzf /tmp/nemo-speech.tar.gz -C /tmp/nemo-speech --strip-components=1 \
    && pip install --no-cache-dir --no-deps /tmp/nemo-speech \
    && rm -rf /tmp/nemo-speech /tmp/nemo-speech.tar.gz \
    && python3 /opt/vonk/verify-runtime.py

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
