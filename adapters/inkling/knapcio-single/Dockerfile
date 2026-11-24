FROM docker.io/lmsysorg/sglang@sha256:bc811ac7ac68c6fca116a49750baca817b8613a30abface8a347682b55d132e3

# knapcio's Inkling 2 Mini kit runs SGLang's published inkling-2-mini image
# (lmsysorg/sglang, pinned by digest in scripts/launch.sh) with the kit's speed
# overlays mounted and put on PYTHONPATH; we pin that image by arm64 digest, bake
# the overlay in and add the Vonk launcher and runtime user.
LABEL org.opencontainers.image.source="https://github.com/knapcio/Inkling-2-Mini-1x-DGX-Spark" \
      org.opencontainers.image.revision="6d89c0ec24cca90ce58991446cf74b1dac449cc8" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
COPY overlay /opt/inkling-overlay
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py
COPY inkling-forced-tool-grammar.py /opt/vonk/build/inkling-forced-tool-grammar.py

RUN python3 /opt/vonk/build/inkling-forced-tool-grammar.py \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache \
    && test -x /opt/vonk/bin/sglang-serve

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    PYTHONPATH=/opt/inkling-overlay

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
