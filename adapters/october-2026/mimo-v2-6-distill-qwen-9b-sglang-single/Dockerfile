FROM docker.io/lmsysorg/sglang@sha256:06441be3de19a816ddbac917535e920cbfc7fc43171abc08417394f43634ffb6
RUN install --directory --mode=0755 /opt/vonk/bin /inputs \
    && install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs
COPY --chmod=0755 launcher /opt/vonk/bin/sglang-serve
ENV HOME=/state HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
