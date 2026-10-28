FROM ghcr.io/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang@sha256:3da6bf4220b511b309818d3af1560e8eabc4e91e008cef4347b4dfdff76ccbd3

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/ornith-15-35b-nvfp4-w4a16-sm121-sglang" \
      org.opencontainers.image.revision="6d0a25d9cc8b8963161cde085a83768b6bc545e0" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Base: r0b0tlab's published GB10 image (pinned venv with SGLang main and pre-warmed
# FlashInfer/Triton JIT caches, image built 2026-08-21), used unchanged. Upstream warns
# a cold FlashInfer CUTLASS compile with the model resident can run out of memory, so the
# pre-warmed caches are kept aside and seeded into the platform cache on first start.
# The platform runs the workload as its own runtime user, never the image's: the seed
# must be readable by any user (cp -a keeps the upstream owner's private modes).
USER root
RUN mkdir -p /opt/vonk/prewarm \
    && cp -a /home/sglang/.cache/sglang /opt/vonk/prewarm/sglang \
    && if [ -d /home/sglang/.cache/triton ]; then cp -a /home/sglang/.cache/triton /opt/vonk/prewarm/triton; fi \
    && chmod -R a+rX /opt/vonk/prewarm
# The image is runtime-only: no CUDA toolkit, no CUDA_HOME. The two toolkit-dependent
# extensions in it, DeepEP and DeepGEMM, assert on import without a toolkit (DeepEP in
# find_cuda_home at deep_ep/__init__.py, outside the ImportError SGLang's dispatcher
# guards), and this recipe uses neither (no --moe-a2a-backend, SGLANG_ENABLE_JIT_DEEPGEMM=0).
# With them removed, SGLang's own ImportError guards (token_dispatcher/deepep.py,
# deep_gemm_wrapper/configurer.py) take their documented fallback. The import below
# runs without a GPU or toolkit, so a base that import-fails again stops the build.
RUN rm -rf /opt/venv/lib/python3.12/site-packages/deep_ep \
        /opt/venv/lib/python3.12/site-packages/deep_gemm \
        /opt/venv/lib/python3.12/site-packages/sgl_deep_ep-*.dist-info \
        /opt/venv/lib/python3.12/site-packages/sgl_deep_gemm-*.dist-info \
    && env -u CUDA_HOME -u CUDA_PATH /opt/venv/bin/python -c \
        "import importlib.util as u; assert u.find_spec('deep_ep') is None and u.find_spec('deep_gemm') is None; import sglang.srt.layers.moe.token_dispatcher.deepep, sglang.srt.lora.lora_manager"
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    HF_HOME=/outputs/cache/huggingface \
    TRITON_CACHE_DIR=/outputs/cache/triton
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
