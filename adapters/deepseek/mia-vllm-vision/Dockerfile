FROM ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark" \
      org.opencontainers.image.revision="c444d7032957f5a5437261d5366fd06b27a01760" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      io.vonk.mia-source-archive-sha256="576f8b616c59e254abea1252a9becb2985216edc0d26b98198a21ff48c7085ee" \
      io.vonk.anemll-source-revision="47503f8e38dadd4dededca798150db2619594fce" \
      io.vonk.pre-patch-tree-sha256="e600de8bf5d9ed4c4016d4bf51c2e20f8a4795dbd5bce146953c9032105f094a" \
      io.vonk.patch-tree-sha256="f6a04bf3e9856d1ea3d339d15635e16f32b833ca8ace46366e999a8999cf878f" \
      io.vonk.dspark-loader-sha256="efe33c32d37ed7f26d869d94626f1415906d31218ec0ee44d79bb2b815b8cf39" \
      io.vonk.vision-hotfix-sha256="0f9ae3c8d07f29fff6c5bb27c3ced4e7cd3d7b87724a1300e9b6386469ed9362"

COPY encoding /opt/vonk-build/encoding
COPY patches /opt/vonk-build/patches
COPY patches/vision_exp /opt/dspark-patches/vision_exp
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 apply-build-patches.py verify-dspark-runtime.py verify-patched-tree.py /opt/vonk-build/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN install --directory --owner 10001 --group 10001 /state /outputs \
    && python3 /opt/vonk-build/verify-dspark-runtime.py \
    && python3 /opt/vonk-build/verify-patched-tree.py \
        /usr/local/lib/python3.12/dist-packages/vllm e600de8bf5d9ed4c4016d4bf51c2e20f8a4795dbd5bce146953c9032105f094a \
    && python3 /opt/vonk-build/apply-build-patches.py \
    && python3 /opt/vonk-build/patches/hotfix-vllm-issue136-xgrammar-termination.py --status \
    && python3 /opt/vonk-build/verify-patched-tree.py \
        /usr/local/lib/python3.12/dist-packages/vllm f6a04bf3e9856d1ea3d339d15635e16f32b833ca8ace46366e999a8999cf878f \
    && rm -rf /opt/vonk-build \
    && test -x /opt/vonk/bin/vllm

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
