FROM ghcr.io/r0b0tlab/sglang-mimo26-env@sha256:84857252a1a9b4196702154ae38eb3cfafdf4cd832795eba18ac2772f8a83f1e

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/mimo26-nvfp4-sm121" \
      org.opencontainers.image.revision="683239d7280880af3bfdce6500ea7349e473b8da" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# Parent: upstream's published preferred image (tag 20260922-582389ce-mtp-mm, arm64): SGLang
# nightly-dev 582389ce + torchcodec + the marlin-skip modelopt_quant loader patch + the three
# MTP/multimodal patch files (docker/Dockerfile.mtp-mm and patches/ in this directory are
# upstream's recipe for it). This layer only adds the placement wrapper and the non-root user.
COPY LICENSE NOTICE.md /opt/vonk/licenses/
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve

RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache \
    && test -x /opt/vonk/bin/sglang-serve

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
