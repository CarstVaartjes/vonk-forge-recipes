FROM ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark" \
      org.opencontainers.image.revision="d1b76251535daef578d8751b04b39c29ad7ecdf9" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      io.vonk.mia-source-archive-sha256="51249cd81009a57b420d48e3d3e1e406427e6c37fe984e1631c52bf37143faee" \
      io.vonk.anemll-source-revision="47503f8e38dadd4dededca798150db2619594fce" \
      io.vonk.pre-patch-tree-sha256="c95903b2148fee727e4e88ffa9b0e65d86239f36bcbec33e352c9a5b92580a03" \
      io.vonk.patch-tree-sha256="25e036c025a68c01de1408450f3ba6d2d368cfab016ea3ba7cc4d4ca63b9a3ba" \
      io.vonk.dspark-source-sha256="fdbfd9e57b052c7ef9584ad628fc66efe7df3735150e624097f2f330891c4f97" \
      io.vonk.dspark-patched-sha256="e5fbedb76142b69041ec827dc37726c4b8fb14b8fdc9f6923be4dc220ce32afe"

COPY encoding /opt/vonk-build/encoding
COPY patches /opt/vonk-build/patches
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 apply-build-patches.py verify-patched-tree.py /opt/vonk-build/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN install --directory --owner 10001 --group 10001 /state /outputs \
    && python3 /opt/vonk-build/verify-patched-tree.py \
        /usr/local/lib/python3.12/dist-packages/vllm c95903b2148fee727e4e88ffa9b0e65d86239f36bcbec33e352c9a5b92580a03 \
    && python3 /opt/vonk-build/apply-build-patches.py \
    && python3 /opt/vonk-build/verify-patched-tree.py \
        /usr/local/lib/python3.12/dist-packages/vllm 25e036c025a68c01de1408450f3ba6d2d368cfab016ea3ba7cc4d4ca63b9a3ba \
    && rm -rf /opt/vonk-build \
    && test -x /opt/vonk/bin/vllm

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
