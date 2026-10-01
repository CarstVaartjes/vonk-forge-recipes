# Qwen3.8-27B EXL3 3.5bpw + MTP / DFlash2 drafting on one DGX Spark (GB10 / SM121).
#
# Upstream (MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw, a deployment kit) ships
# no container: its start.sh builds a venv on the host, installs PyTorch from
# the cu130 index, pip-installs the MiaAI-Lab exllamav3 fork from git HEAD
# (compiling the CUDA extension for 12.0;12.1) and serves tools/serve_openai.py.
# This image performs the same installation against the digest-pinned official
# vLLM arm64 image (used only as a torch + CUDA 13 toolchain base; vLLM itself
# is not used) and a fork commit pinned by SHA. The upstream launcher scripts
# are vendored for reference and are not run; exl3-serve replaces start.sh.
FROM docker.io/vllm/vllm-openai@sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw" \
      org.opencontainers.image.revision="1242187390f780dba907e30659f1639fd0b491c8" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.exllamav3.source="https://github.com/MiaAI-Lab/exllamav3" \
      io.vonk.exllamav3.revision="63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed"

ARG EXLLAMAV3_COMMIT=63b32f001d7b2cfed3b3e3aaf25f534ba53cc7ed
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="12.0;12.1" \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH}

# The engine, as upstream's start.sh installs it (pip install --no-build-isolation
# from the fork). The vLLM image keeps the CUDA 13 headers under nvidia/cu13
# rather than /usr/local/cuda/include. The image's torch/triton stay as they
# are (constraints), so pip only adds the engine's missing dependencies.
RUN set -eux; \
    mkdir -p /tmp/exllamav3; \
    curl -fsSL "https://github.com/MiaAI-Lab/exllamav3/archive/${EXLLAMAV3_COMMIT}.tar.gz" \
      | tar -xz -C /tmp/exllamav3 --strip-components=1; \
    python3 -c "from pathlib import Path; assert (Path('/tmp/exllamav3')/'exllamav3/version.py').is_file()"; \
    PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    export CPATH="${PY_SITE}/nvidia/cu13/include${CPATH:+:$CPATH}"; \
    export CPLUS_INCLUDE_PATH="${PY_SITE}/nvidia/cu13/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"; \
    export C_INCLUDE_PATH="${PY_SITE}/nvidia/cu13/include${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"; \
    python3 -m pip freeze 2>/dev/null | grep -i -E '^(torch|triton)==' > /tmp/constraints.txt; \
    cat /tmp/constraints.txt; \
    cd /tmp/exllamav3; \
    MAX_JOBS=8 python3 -m pip install --no-build-isolation --no-cache-dir -c /tmp/constraints.txt . aiohttp huggingface_hub; \
    python3 -c "import torch; import exllamav3_ext; print('exllamav3_ext', exllamav3_ext.__file__)"; \
    rm -rf /tmp/exllamav3 /tmp/constraints.txt /root/.cache

# Upstream server, byte-identical to the vendored copy.
COPY tools/serve_openai.py /opt/exl3/tools/serve_openai.py
COPY --chmod=0755 exl3-serve /opt/vonk/bin/exllamav3-serve
COPY verify-runtime.py /opt/vonk/verify-runtime.py

RUN python3 /opt/vonk/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner=10001 --group=10001 /outputs /outputs/cache

ENV HF_HUB_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache/xdg \
    XDG_CONFIG_HOME=/outputs/cache/config \
    HF_HOME=/outputs/cache/huggingface \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TORCH_HOME=/outputs/cache/torch \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    CUDA_CACHE_PATH=/outputs/cache/cuda

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
