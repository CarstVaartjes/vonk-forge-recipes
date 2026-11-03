FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/drowzeys/TensorFold" \
      org.opencontainers.image.revision="8cd08c8350c02cf3d7d8eee5f1c1d9dfa22fa703" \
      ai.vonkforge.runtime-interface="v1"
RUN curl -fsSL https://codeload.github.com/drowzeys/TensorFold/tar.gz/8cd08c8350c02cf3d7d8eee5f1c1d9dfa22fa703 -o /tmp/music.tar.gz \
 && echo '5065371d4235012ca0fd2ec23d3fc554248f0b4c330f307a2d660f1ba8f0a32b  /tmp/music.tar.gz' | sha256sum -c - \
 && curl -fsSL https://ziglang.org/download/0.17.0/zig-aarch64-linux-0.17.0.tar.xz -o /tmp/zig.tar.xz \
 && echo '9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8  /tmp/zig.tar.xz' | sha256sum -c - \
 && mkdir -p /tmp/music /opt/vonk/bin /opt/tensorfold \
 && tar -xzf /tmp/music.tar.gz --strip-components=1 -C /tmp/music \
 && tar -xJf /tmp/zig.tar.xz -C /tmp \
 && cd /tmp/music \
 && /tmp/zig-aarch64-linux-0.17.0/zig build music3-generate -j2 -Doptimize=ReleaseFast -Dnvcc=/usr/local/cuda/bin/nvcc -Dsm=121 \
 && cp zig-out/bin/music3-generate /opt/tensorfold/music3-generate \
 && cp LICENSE /opt/tensorfold/LICENSE \
 && rm -rf /tmp/music* /tmp/zig* /root/.cache/zig
COPY vonk_agent_protocol-4.1.0-py3-none-any.whl /opt/vonk/vonk_agent_protocol-4.1.0-py3-none-any.whl
RUN python3 -m pip install --no-cache-dir /opt/vonk/vonk_agent_protocol-4.1.0-py3-none-any.whl \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY MUSIC3-LICENSE /opt/tensorfold/MUSIC3-LICENSE
COPY job.py /opt/vonk/job.py
COPY --chmod=0755 launcher /opt/vonk/bin/pytorch-pipeline
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
