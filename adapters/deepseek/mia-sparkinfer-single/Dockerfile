# syntax=docker/dockerfile:1.7
FROM ghcr.io/0xsero/deepseek-v4-flash-0731-spark-sparkinfer@sha256:2e077489a83a0360952828051fe7f7a32c1801e5ce8436d85f7267583d614ff4

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark" \
      org.opencontainers.image.revision="fdcd538fbf95fb15b2d6850db9613d22b2c889b8" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      org.opencontainers.image.description="Offline Vonk adapter for MiaAI-Lab's 384K compact-NVFP4 one-DGX-Spark profile"

# The base image's PATH has no groupadd/useradd; write the entries directly.
RUN if ! getent group 10001 >/dev/null; then echo 'vonk:x:10001:' >> /etc/group; fi \
    && if ! getent passwd 10001 >/dev/null; then \
         echo 'vonk:x:10001:10001::/state:/usr/sbin/nologin' >> /etc/passwd; \
       fi \
    && install --directory --owner 10001 --group 10001 \
         /opt/vonk/bin /opt/vonk/tiktoken-cache /state /outputs \
    && if test -d /root/.cache/tiktoken_cache; then \
         cp -a /root/.cache/tiktoken_cache/. /opt/vonk/tiktoken-cache/; \
       fi \
    && chmod -R a+rX /opt/vonk/tiktoken-cache

COPY overlays/coalesce_rank_sliced_exl3.py /opt/recipe/scripts/coalesce_rank_sliced_exl3.py
COPY overlays/sparkinfer/moe/fused_moe/_impl.py /opt/sparkinfer/sparkinfer/moe/fused_moe/_impl.py
COPY overlays/sparkinfer/moe/_shared/kernels/tiny_decode.py /opt/sparkinfer/sparkinfer/moe/_shared/kernels/tiny_decode.py
COPY overlays/vllm/models/deepseek_v4/nvidia/dspark.py /opt/vllm/vllm/models/deepseek_v4/nvidia/dspark.py
COPY overlays/sparkinfer/attention/_shared/mla/prefill.py /opt/sparkinfer/sparkinfer/attention/_shared/mla/prefill.py
COPY overlays/sparkinfer/attention/_shared/mla/prefill_mg.py /opt/sparkinfer/sparkinfer/attention/_shared/mla/prefill_mg.py
COPY --chmod=0755 overlays/serve-ds4-flash.sh /opt/vllm/serve-ds4-flash.sh
COPY --chmod=0755 vllm-wrapper.sh /opt/vonk/bin/vllm
COPY NOTICE /opt/vonk/NOTICE.mia-sparkinfer-single

ENV HOME=/state \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=1 \
    TIKTOKEN_CACHE_DIR=/opt/vonk/tiktoken-cache \
    TIKTOKEN_RS_CACHE_DIR=/opt/vonk/tiktoken-cache \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1

# The base image pins xgrammar 0.1.27, but its vLLM needs normalize_tool_choice (>= 0.2.1).
# The creator's image-patch/entrypoint-toolfix.sh upgrades it at boot (xgrammar>=0.2.4, then
# transformers back to 5.13.1); a runtime pip install cannot work offline as USER 10001, so the
# same step runs here at build time with the creator's floor, hash-checked. --no-deps leaves
# transformers untouched; the assert proves the creator's restored version still stands.
RUN printf 'xgrammar==0.2.4 --hash=sha256:b1d2bef13aec8122dfa6b96e3ece2cc0a42a732685f1d44f6248c305274e8eae\n' > /tmp/xgrammar.txt \
    && env -u PIP_CONSTRAINT /opt/runtime-venv/bin/python -m pip install \
         --no-deps --only-binary :all: --require-hashes -r /tmp/xgrammar.txt \
    && /opt/runtime-venv/bin/python -c \
         'import importlib.metadata as m; from xgrammar import normalize_tool_choice; assert m.version("xgrammar") == "0.2.4"; assert m.version("transformers") == "5.13.1"' \
    && rm -f /tmp/xgrammar.txt

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
