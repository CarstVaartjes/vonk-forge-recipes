FROM ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark" \
      org.opencontainers.image.revision="7963d432bfe717f2a7249a53fb4c0673c239c03e" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      io.vonk.mia-source-archive-sha256="342b20c7245f8e46e39b16d679d71f2bd772e12bce2c681e52b979b16f2f730e" \
      io.vonk.anemll-source-revision="47503f8e38dadd4dededca798150db2619594fce" \
      io.vonk.pre-patch-tree-sha256="4a1ee8ac6eaefb8dfc9e3d15792daf9154dbaee0acab6295f1e4b95c8397c432" \
      io.vonk.patch-tree-sha256="8079b5e2a300e12c55ec7ccb5182d89f4966de5e2dcf3d5d0917eca5164343f1" \
      io.vonk.dspark-loader-sha256="efe33c32d37ed7f26d869d94626f1415906d31218ec0ee44d79bb2b815b8cf39" \
      io.vonk.vision-hotfix-sha256="882c26ed30e1e2f611bd902bd2ee63853f4eeea3eb3ca23137b2adf8b27449e8"

COPY encoding /opt/vonk-build/encoding
COPY patches /opt/vonk-build/patches
COPY patches/vision_exp /opt/dspark-patches/vision_exp
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 apply-build-patches.py verify-dspark-runtime.py verify-patched-tree.py /opt/vonk-build/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN install --directory --owner 10001 --group 10001 /state /outputs \
    && python3 /opt/vonk-build/verify-dspark-runtime.py \
    && python3 /opt/vonk-build/verify-patched-tree.py \
        /usr/local/lib/python3.12/dist-packages/vllm 4a1ee8ac6eaefb8dfc9e3d15792daf9154dbaee0acab6295f1e4b95c8397c432 \
    && python3 /opt/vonk-build/apply-build-patches.py \
    && python3 /opt/vonk-build/patches/hotfix-vllm-issue136-xgrammar-termination.py --status \
    && python3 /opt/vonk-build/verify-patched-tree.py \
        /usr/local/lib/python3.12/dist-packages/vllm 8079b5e2a300e12c55ec7ccb5182d89f4966de5e2dcf3d5d0917eca5164343f1 \
    && rm -rf /opt/vonk-build \
    && test -x /opt/vonk/bin/vllm

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
