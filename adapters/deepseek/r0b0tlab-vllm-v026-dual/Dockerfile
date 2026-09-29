FROM ghcr.io/r0b0tlab/deepseek-v4-flash-dspark-v026-sm121@sha256:ef852781efd6278cb3d908a4298874fb8b34fe84f0e1a03181ed415ec233a9d4

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/DeepSeek-V4-Flash-DSpark-v026-SM121" \
      org.opencontainers.image.revision="d8393134b49f7e92ffd64b9670b5d500fc6478cb" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's frozen native-v11 image (vLLM 0.26.0+dspark.sm121.3, arm64, CUDA 13.0,
# registry digest above, image ID sha256:2dfe24a3...). Its /usr/local/bin/vllm is the engine;
# this layer only adds the placement wrapper and the non-root user.
COPY LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/vonk/licenses/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN test -x /usr/local/bin/vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
RUN python3 -c 'import getpass, os, pwd; assert os.getuid() == 10001; assert pwd.getpwuid(10001).pw_uid == 10001; assert getpass.getuser()'
ENTRYPOINT []
