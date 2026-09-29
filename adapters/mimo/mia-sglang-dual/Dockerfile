FROM docker.io/lmsysorg/sglang@sha256:9e1fb4c395b9c406136e10aa445b8784d06bca3839623b52cbe4a3b231a157a8

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/MiMo-V2.6-Flash-2x-DGX-Sparks" \
      org.opencontainers.image.revision="201be3e743d215291feee09fe470822ec453c63d" \
      org.opencontainers.image.licenses="AGPL-3.0-only AND Apache-2.0" \
      mimo26.base="lmsysorg/sglang:nightly-dev-cu13-20260921-0f6761b5"

# The base is lmsysorg/sglang nightly-dev-cu13-20260921-0f6761b5 (arm64), the
# first image with sgl-project/sglang#40448 (MiMo-V2 MXFP4 experts kept packed).
# boot.py refuses to launch on an engine that lacks it.
#
# Upstream installs ffmpeg with apt and an unpinned torchcodec with pip, then
# asserts they import. This base already ships ffmpeg, so only torchcodec is
# installed, pinned by wheel hash (0.16.0 was the newest release on the
# upstream commit date). torchaudio comes with the base.
COPY requirements-audio.txt /opt/mimo26/requirements-audio.txt
COPY boot.py /opt/mimo26/boot.py
COPY patches/sitecustomize.py /opt/mimo26/patches/sitecustomize.py
COPY --chmod=0755 sglang-serve /opt/vonk/bin/sglang-serve
COPY verify-runtime.py /opt/mimo26/verify-runtime.py

RUN pip install --no-cache-dir --no-deps --require-hashes -r /opt/mimo26/requirements-audio.txt \
    && python3 -c "import torchaudio; from torchcodec.decoders import AudioDecoder; import shutil; assert shutil.which('ffprobe'), 'ffprobe missing'" \
    && python3 /opt/mimo26/verify-runtime.py \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton

ENV MODEL_PATH=/models/target \
    STATE_PATH=/state \
    OFFLOAD_MODE=off \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    SGLANG_DISABLE_USAGE_REPORT=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
