FROM ghcr.io/drowzeys/vllm-024-puzzle-a4q@sha256:a859a90363d1fee9a4ea37ce507b082fa353e6ac0fb2ddab2fc3295dbd37fe8d

LABEL org.opencontainers.image.source="https://github.com/drowzeys/vLLm-0.24-optimized-Tencent-Hy3-295B-A21B-NVFP4-A4Q-MTP2-4bit-KV-256K-ctx-2x-DGX-Spark" \
      org.opencontainers.image.revision="7f9e9d82365bbb8b3dc5ec151c3689755153edb2" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's published :fi614 image (vLLM 0.24 aeon lineage + jethac FlashInfer 0.6.13
# with the A4Q wiring). overlays/ are upstream's Hy3 parser and MTP-loader files, which
# upstream's hy3-a4q-serve.sh bind-mounts over the installed vLLM; here they are copied to
# the same paths. scripts/ is upstream's launcher set, carried for provenance.
COPY overlays/hy_v3_tool_parser.py /usr/local/lib/python3.12/site-packages/vllm/tool_parsers/hy_v3_tool_parser.py
COPY overlays/hy_v3_reasoning_parser.py /usr/local/lib/python3.12/site-packages/vllm/reasoning/hy_v3_reasoning_parser.py
COPY overlays/hy_v3_mtp.py /usr/local/lib/python3.12/site-packages/vllm/model_executor/models/hy_v3_mtp.py
COPY scripts /opt/vonk/upstream-scripts
COPY LICENSE /opt/vonk/licenses/LICENSE
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

# The two launch-time edits upstream's script makes inside the container: use the
# system NCCL 2 and tolerate a missing FlashInfer all-reduce import.
RUN ln -sf /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/local/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2 2>/dev/null; \
    sed -i 's/^except ImportError:/except Exception:/' /usr/local/lib/python3.12/site-packages/vllm/distributed/device_communicators/flashinfer_all_reduce.py 2>/dev/null; \
    command -v vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_ENGINE_READY_TIMEOUT_S=3600 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    FLASHINFER_WORKSPACE_BASE=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch-extensions

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
