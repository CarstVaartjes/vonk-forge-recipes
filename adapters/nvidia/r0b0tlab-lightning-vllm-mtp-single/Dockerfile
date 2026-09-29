FROM docker.io/vllm/vllm-openai@sha256:3af90144a0926e5c5fe46ee16e5201e763dd854538b9d7ce433755f11dadaf78

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/nemotron-3.5-lightning-sm121-nvfp4" \
      org.opencontainers.image.revision="024bb77b60482ef833e85376260ccb5d4b767225" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

# Upstream runtime/Dockerfile: the pinned image plus the guide-required fastokens
# package (0.3.1 has an ARM64 cp39-abi3 wheel).
RUN python3 -m pip install --no-cache-dir --no-deps 'fastokens==0.3.1' \
    && python3 -c 'import fastokens; print("fastokens", getattr(fastokens, "__version__", "0.3.1"))'

COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
