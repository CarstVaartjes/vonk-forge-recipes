FROM ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark" \
      org.opencontainers.image.revision="de230b45bc49e88ea92af898650b6bbb92f0e266" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      io.vonk.mia-source-archive-sha256="3bcd0b4c70677bf7e2d1f2a790315624edf0a8cba1920f33bd8daa944804fced" \
      io.vonk.anemll-source-revision="47503f8e38dadd4dededca798150db2619594fce" \
      io.vonk.pre-patch-tree-sha256="4a1ee8ac6eaefb8dfc9e3d15792daf9154dbaee0acab6295f1e4b95c8397c432" \
      io.vonk.patch-tree-sha256="8be1b911a35d798a8cda5c548ed071ef6605d63c2bb40951a6e9c81e65d7872c" \
      io.vonk.dspark-loader-sha256="efe33c32d37ed7f26d869d94626f1415906d31218ec0ee44d79bb2b815b8cf39" \
      io.vonk.vision-hotfix-sha256="ee4176ce9b0c50cf5624e85c21046c406a47d92baf60a52174cecd027f7e26fe"

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
        /usr/local/lib/python3.12/dist-packages/vllm 8be1b911a35d798a8cda5c548ed071ef6605d63c2bb40951a6e9c81e65d7872c \
    && rm -rf /opt/vonk-build \
    && test -x /opt/vonk/bin/vllm

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
