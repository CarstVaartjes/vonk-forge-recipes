FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark" \
      org.opencontainers.image.revision="6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Upstream ran vllm/vllm-openai:nightly-8a728663 (CUDA 13.0.2), a tag that is gone.
# Its commit's CUDA 13 build is the hash-locked wheel in requirements.lock, installed
# system-wide on the CUDA 13 base (packages as of 2026-09-05, NCCL 2.30.7 as in the
# creator's image; see requirements.in and overrides.txt). The CUDA 12.9 build of the same commit carries
# its SM12x kernels as sm_120a only, which GB10 (sm_121) cannot load.
ENV DEBIAN_FRONTEND=noninteractive CUDA_HOME=/usr/local/cuda PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates build-essential python3 python3-dev python3-pip libnuma1 libibverbs1 ibverbs-providers librdmacm1 && rm -rf /var/lib/apt/lists/*
COPY requirements.lock /tmp/requirements.lock
RUN python3 -m pip install --ignore-installed --no-deps --require-hashes --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://flashinfer.ai/whl --extra-index-url https://flashinfer.ai/whl/cu130 -r /tmp/requirements.lock && rm /tmp/requirements.lock
# Upstream bind-mounts these files over the vLLM nightly's own copies at every
# launch (launch/qwen38fn-nvidia-tp*.sh). The same exact files are baked in here.
COPY --chmod=0755 verify-runtime.py /opt/vonk/verify-runtime.py
RUN python3 /opt/vonk/verify-runtime.py base
# libcuda.so.1 is a link-time dependency of vLLM's kernels; the CUDA stub lets the
# imports run without a driver. The API server import fails the build on a
# dependency API break. The fatbin scan fails the build for any library
# whose SM12x kernels cannot load on GB10.
RUN mkdir /tmp/cuda-stub && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stub/libcuda.so.1 \
    && LD_LIBRARY_PATH=/tmp/cuda-stub python3 -c "import torch, vllm._C_stable_libtorch, vllm.entrypoints.openai.api_server" \
    && rm -r /tmp/cuda-stub \
    && python3 /opt/vonk/verify-runtime.py kernels

COPY patch/ple_layer.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ple_layer.py
COPY patch/ple_mmap.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple_mmap.py
COPY patch/model_state.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/model_state.py
COPY patch/mtp_draft_vocab.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/mtp.py
COPY patch/compilation.py /usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py
COPY patch/upstream-overlays/ops_ple.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/ple.py
COPY patch/upstream-overlays/ops_qsa.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ops/qsa.py
COPY patch/upstream-overlays/qsa.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/qsa.py
COPY patch/upstream-overlays/platforms_interface.py /usr/local/lib/python3.12/dist-packages/vllm/platforms/interface.py
COPY patch/upstream-overlays/modelopt.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py
COPY LICENSE.upstream NOTICE.upstream /opt/vonk/
COPY --chmod=0755 qwen38-tonyd2wild-serve.py /opt/vonk/bin/vllm

RUN python3 /opt/vonk/verify-runtime.py overlays \
    && install --directory --mode=0755 /outputs \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && chown -R 10001:10001 /outputs /opt/vonk

ENV VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600

WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
