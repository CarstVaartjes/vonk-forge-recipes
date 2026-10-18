FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04@sha256:5dc1bca23d05bd37b011be68ec470c03b403a5da07ec3a86e41af9470e9d0cc6
LABEL org.opencontainers.image.source="https://github.com/vcruz305/Nemotron-3.5-Bolt-74B-A9B-24GB-vLLM-recipe" \
      org.opencontainers.image.revision="2fe482f67d5fb8859acac8d22b6670ef453b28a9"
ENV DEBIAN_FRONTEND=noninteractive CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.1 \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} MAX_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates build-essential python3 python3-dev python3-venv libnuma1 libibverbs1 && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://wheels.vllm.ai/77154ca6116ae34cbfe314beef1016bde9b63989/cu130 --extra-index-url https://flashinfer.ai/whl --extra-index-url https://flashinfer.ai/whl/cu130 -r /tmp/requirements.lock
# FlashInfer minimal mode installs the version shim without the other GPU families.
# The matching SM120f provider is already hash-locked in requirements.lock.
COPY kernel-shim.lock /tmp/kernel-shim.lock
RUN pip install --no-cache-dir --no-deps --require-hashes --index-url https://flashinfer.ai/whl/cu130 -r /tmp/kernel-shim.lock
COPY --chmod=0755 vllm-wrapper /opt/vonk/bin/vllm
RUN python -c "import vllm, b12x, flashinfer; import importlib.metadata as m; assert m.version('flashinfer-jit-cache-sm120f') == '0.7.0.post1+cu130'" && groupadd --gid 10001 vonk && useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
