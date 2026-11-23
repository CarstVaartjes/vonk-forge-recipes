FROM vllm/vllm-openai@sha256:ecdf0ccc663d86332459f5a50a38c30861a5079bc3924aa08fdc657ef10dd648

# tonyd2wild's Inkling 2 Mini kit runs vLLM's published Inkling 2 Mini image
# (vllm/vllm-openai:inkling-2-mini, named in its launch scripts); we pin that
# image by digest and only add the Vonk launcher and the runtime user.
LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark" \
      org.opencontainers.image.revision="300942daf6be61f39a7e6c6055bd7cb6e49c4e7c" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0"

COPY --chmod=0755 vllm /opt/vonk/bin/vllm
COPY verify-runtime.py /opt/vonk/verify-runtime.py

RUN python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

USER 10001:10001
ENTRYPOINT []
