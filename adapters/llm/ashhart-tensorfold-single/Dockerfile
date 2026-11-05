# TensorFold v1.2.3 (Apache-2.0, commit 6ec5fa4c): Ash Hart's native Zig engine as released, the linux-aarch64 archive
# installed as the creator's Linux instructions install it (README "Linux with an NVIDIA GPU": download, sha256sum -c,
# unpack, keep its directories together). CUDA fatbins (sm_80, sm_86, sm_120, sm_121) are embedded in the binary, so
# nothing is compiled at start; lib/ holds libtfgrammar for structured output and required tool calls. The NVIDIA
# driver (R580 or newer) supplies libcuda.so.1; the base supplies the CUDA 13 cuBLASLt that --vision loads.
# This layer only adds the Controller launcher wrapper at the platform's path and the non-root user.
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3

LABEL org.opencontainers.image.source="https://github.com/ashhart/TensorFold" \
      org.opencontainers.image.revision="6ec5fa4cb233f3606acbdad5bec433f0f8758f96" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

RUN curl -fsSL -o /tmp/tensorfold.tar.gz https://github.com/ashhart/TensorFold/releases/download/v1.2.3/tensorfold-1.2.3-linux-aarch64.tar.gz \
    && echo "02bb060e2f42bc4dcc0d3c6800022046b10420e0237ebde2519343703f4f68f9  /tmp/tensorfold.tar.gz" | sha256sum -c - \
    && mkdir /opt/tensorfold \
    && tar -xzf /tmp/tensorfold.tar.gz -C /opt/tensorfold --strip-components=1 \
    && rm /tmp/tensorfold.tar.gz \
    && test "$(cat /opt/tensorfold/VERSION)" = 1.2.3 \
    && /opt/tensorfold/bin/tensorfold-native --version

COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi

# The engine writes its measured window costs under $HOME/.cache/tensorfold and the driver its compute cache, both under
# the platform's writable /outputs/cache.
ENV HF_HUB_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    CUDA_CACHE_PATH=/outputs/cache/cuda \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
