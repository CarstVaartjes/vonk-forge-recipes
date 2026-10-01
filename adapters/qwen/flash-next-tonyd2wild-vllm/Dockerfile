FROM docker.io/vllm/vllm-openai@sha256:c4a8820479f1c012dcb350300220eb82845a484937c9fc635cb2c5489dd3b74f

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark" \
      org.opencontainers.image.revision="6ad1c8f15cbab1ababd2048e8e5f94094dbfc4a0" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Upstream bind-mounts these files over the vLLM nightly's own copies at every
# launch (launch/qwen38fn-nvidia-tp*.sh). The same exact files are baked in here.
COPY --chmod=0755 verify-runtime.py /opt/vonk/verify-runtime.py
RUN python3 /opt/vonk/verify-runtime.py base

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
