FROM ghcr.io/drowzeys/keys-heretic-minimax-h3-sol-engine-spark@sha256:1789d539f2dfc4a3f061e5577fcc680b36ae3e75e0c01f3303df79100f5dd153
LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-heretic-MiniMax-H3-sol-engine-more-speed-upgrades-upscaler-finish-Single-DGX-Spark" \
      org.opencontainers.image.revision="7e8304fd8de9741119e704d1aa347534346c3a48" \
      ai.vonkforge.runtime-interface="v1"
COPY vonk_agent_protocol-4.1.0-py3-none-any.whl /opt/vonk/vonk_agent_protocol-4.1.0-py3-none-any.whl
RUN python3 -m pip install --no-cache-dir /opt/vonk/vonk_agent_protocol-4.1.0-py3-none-any.whl
COPY workflows /opt/vonk/source/workflows
COPY runtime_smoke.py /opt/vonk/runtime_smoke.py
RUN cd /opt/comfy/ComfyUI && PYTHONPATH=/opt/comfy/ComfyUI python3 /opt/vonk/runtime_smoke.py
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY --chmod=0555 comfyui_job.py /opt/vonk/bin/comfyui-job
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache TMPDIR=/outputs/cache/tmp PYTHONUNBUFFERED=1
USER 10001:10001
WORKDIR /opt/comfy/ComfyUI
ENTRYPOINT []
