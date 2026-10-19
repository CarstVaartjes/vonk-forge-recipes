FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-Wasserman-Slate-AI-Studio-powered-by-TensorFold-built-with-integrated-Fish-Audio-Drama3" \
      org.opencontainers.image.revision="1e8677e7c9151836de375e3cc6306ed6e4170c65" \
      ai.vonkforge.runtime-interface="v1"
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 \
 && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://codeload.github.com/drowzeys/TensorFold/tar.gz/1f7ecc75ded5df6da151d4d5eeee87b35161e3c2 -o /tmp/engine.tar.gz \
 && echo 'fe408b7ac227d8976a3050db95239cadcb7cdca1e906fb76adfd20a219485dfd  /tmp/engine.tar.gz' | sha256sum -c - \
 && curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz -o /tmp/zig.tar.xz \
 && echo '9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz' | sha256sum -c - \
 && mkdir -p /tmp/engine /opt/tensorfold /opt/vonk/bin /opt/vonk/source/workflows \
 && tar -xzf /tmp/engine.tar.gz --strip-components=1 -C /tmp/engine \
 && tar -xJf /tmp/zig.tar.xz -C /tmp \
 && cd /tmp/engine \
 && /tmp/zig-aarch64-linux-0.17.0/zig build tf-h3 -j2 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc \
 && cp zig-out/lib/libtf_h3.so LICENSE /opt/tensorfold/ \
 && rm -rf /tmp/engine* /tmp/zig* /root/.cache/zig
RUN curl -fsSL https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/5a665890ded5020c282f63c4ea2dd9e083e025ff -o /tmp/comfy.tar.gz \
 && echo '0ea1a0522bcd51fa6a48b4944d01cf8eadc7e9115510e682be2d2e772cdbb96f  /tmp/comfy.tar.gz' | sha256sum -c - \
 && mkdir -p /opt/comfyui \
 && tar -xzf /tmp/comfy.tar.gz --strip-components=1 -C /opt/comfyui \
 && rm /tmp/comfy.tar.gz \
 && python3 -m pip install --no-cache-dir -r /opt/comfyui/requirements.txt
COPY vonk_agent_protocol-4.1.0-py3-none-any.whl /opt/vonk/vonk_agent_protocol-4.1.0-py3-none-any.whl
RUN python3 -m pip install --no-cache-dir /opt/vonk/vonk_agent_protocol-4.1.0-py3-none-any.whl
COPY tensorfold_h3 /opt/comfyui/custom_nodes/tensorfold_h3
COPY workflows /opt/vonk/source/workflows
COPY runtime_smoke.py /opt/vonk/runtime_smoke.py
RUN cd /opt/comfyui && PYTHONPATH=/opt/comfyui python3 /opt/vonk/runtime_smoke.py
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY --chmod=0555 comfyui_job.py /opt/vonk/bin/comfyui-job
ENV TF_H3_LIB=/opt/tensorfold/libtf_h3.so HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache TMPDIR=/outputs/cache/tmp PYTHONUNBUFFERED=1
USER 10001:10001
WORKDIR /opt/comfyui
ENTRYPOINT []
