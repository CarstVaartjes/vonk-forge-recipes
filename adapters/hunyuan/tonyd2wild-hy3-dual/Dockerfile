FROM ghcr.io/drowzeys/vllm-node-tf5-glm52-b12x@sha256:e006935eb4f8266705f213c369de1eac8de7d20417254c5f234601a2fd56d481

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark" \
      org.opencontainers.image.revision="285bc9177800114fe255274e4a002cd3d812d5b4" \
      org.opencontainers.image.licenses="Apache-2.0"

# Upstream ran vLLM 0.23.1rc1.dev190 (image tag vllm-node-tf5-glm52-b12x:probe-modded,
# not published). The pinned base is the published build of the same vLLM
# (0.23.1rc1.dev190+gab6660699, build-metadata.yaml in the image).
#
# kodelow/Hy3-NVFP4-W4A16 emits ":opensource"-suffixed special tokens while vLLM's
# hy_v3 parsers hardcode the bare forms. Upstream rewrites the two parser files at
# every start (scripts/serve-hy3-tools.sh); the same substitutions are applied once
# here because the runtime root is read-only.
COPY --chmod=0755 apply-parser-patch.sh /opt/vonk/build/apply-parser-patch.sh
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY verify-runtime.py /opt/vonk/build/verify-runtime.py

RUN /opt/vonk/build/apply-parser-patch.sh \
    && python3 /opt/vonk/build/verify-runtime.py \
    && rm -rf /opt/vonk/build \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs \
    && test -x /opt/vonk/bin/vllm

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
