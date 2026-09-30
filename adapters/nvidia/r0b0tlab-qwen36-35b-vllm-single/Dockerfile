FROM ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121@sha256:2d144fafe3f330fa17fa1facf4f589eee49b75bdf539ac69d1fe002b5b5bb0a5

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-sm121-vllm" \
      org.opencontainers.image.revision="106fb53372c260e719c97d5b59e48142792783c7" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Base: r0b0tlab's digest-pinned vLLM v0.25.0 SM121 build (vllm-v0250-cu130-sm121).
# The steps below are upstream's Dockerfile.v025: route ModelOpt W4A16 checkpoints to
# the native SM121 W4A4 path, then byte-compile the patched file and the audit script.
USER root
RUN install -d -m 0755 /opt/r0b0tlab
COPY patch_modelopt_w4a16_native_w4a4.py /tmp/patch_modelopt_w4a16_native_w4a4.py
COPY audit_runtime_v025.py /opt/r0b0tlab/audit_runtime_v025.py
RUN set -eux; \
    MODELOPT_PATH="$(python3 -c 'import importlib.util; s=importlib.util.find_spec("vllm.model_executor.layers.quantization.modelopt"); assert s and s.origin; print(s.origin)')"; \
    python3 /tmp/patch_modelopt_w4a16_native_w4a4.py --path "${MODELOPT_PATH}"; \
    python3 -m py_compile "${MODELOPT_PATH}" /opt/r0b0tlab/audit_runtime_v025.py; \
    chmod 0755 /opt/r0b0tlab/audit_runtime_v025.py; \
    rm -f /tmp/patch_modelopt_w4a16_native_w4a4.py

COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 vllm-api /opt/vonk/bin/vllm-api
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
