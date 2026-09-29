FROM ghcr.io/r0b0tlab/vllm-minimax-m27-nvfp4-blackwell@sha256:9ee93af94a508bf7ef6125f6bf655c27b9082a892296d9db20ecdee6e360cb03

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/minimax-m27-nvfp4-gb10-benchmark" \
      org.opencontainers.image.revision="32c395f75ff44f744e6b27991d17775623de071a" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's published runtime image (arm64, NGC PyTorch 26.02 base, vLLM 0.21.1 source
# build with SM120/SM121 FP4 kernels; docker/Dockerfile in this directory is upstream's recipe for
# it). This layer only adds the placement wrapper (Ray head/worker) and the non-root user.
COPY LICENSE /opt/vonk/licenses/r0b0tlab-LICENSE
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
ENTRYPOINT []
