FROM docker.io/vllm/vllm-openai@sha256:2cc49b81319f7a66a33dd8bd63a7bfddae079122b33ce51989b6828a1f038c37

LABEL org.opencontainers.image.source="https://github.com/vllm-project/vllm" \
      org.opencontainers.image.revision="752a3a504485790a2e8491cacbb35c137339ad34" \
      org.opencontainers.image.licenses="Apache-2.0" \
      io.vonk.vllm.build-commit="752a3a504485790a2e8491cacbb35c137339ad34"

# Upstream's start.sh bootstraps the FlashInfer 0.6.15.dev20260712 nightly trio (python, cubin,
# jit-cache) with pip on every container start and fails closed on mismatch (the stock 0.25.1
# image ships 0.6.13; mismatched FP4 kernels collapse draft acceptance). Bake the exact wheels
# in at build time instead, pinned by hash, so the runtime needs no network. Upstream then
# installs cuda-tile 1.5.0 and nccl4py 0.3.1 (versions pinned, dependencies resolved by pip).
COPY requirements-flashinfer.txt /opt/vonk/requirements-flashinfer.txt
RUN pip install --no-cache-dir --no-deps --require-hashes -r /opt/vonk/requirements-flashinfer.txt \
    && pip install --no-cache-dir "cuda-tile==1.5.0" "nccl4py==0.3.1" \
    && python3 -c "import importlib.metadata as m; [print(p, m.version(p)) for p in ('flashinfer-python','flashinfer-cubin','flashinfer-jit-cache','cuda-tile','nccl4py')]; assert m.version('flashinfer-python') == m.version('flashinfer-cubin') == '0.6.15.dev20260712' and m.version('flashinfer-jit-cache') == '0.6.15.dev20260712+cu130'"

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --mode=0755 /opt/vonk/bin \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache \
    && printf '%s\n' '#!/bin/sh' 'set -eu' 'mkdir -p "$HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE" "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$UV_CACHE_DIR"' 'exec vllm "$@"' > /opt/vonk/bin/vllm \
    && chmod 0755 /opt/vonk/bin/vllm

ENV HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache/xdg \
    XDG_CONFIG_HOME=/outputs/cache/config \
    HF_HOME=/outputs/cache/huggingface \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache/flashinfer \
    VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/outputs/cache/flashinfer/autotune \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_HOME=/outputs/cache/torch \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TORCHINDUCTOR_CACHE_DIR=/outputs/cache/torchinductor \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    UV_CACHE_DIR=/outputs/cache/uv \
    CUTE_DSL_ARCH=sm_121a \
    VLLM_TARGET_DEVICE=cuda

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
