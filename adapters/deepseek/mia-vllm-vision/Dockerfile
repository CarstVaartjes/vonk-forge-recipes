FROM ghcr.io/anemll/dspark-vllm-gx10@sha256:a83948492cf13df455170fb42885f5ef4db54fefe0feff0f841ecbff464ac9d8

LABEL org.opencontainers.image.source="https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark" \
      org.opencontainers.image.revision="a8b4636710068fd30edb98e1365d567086cc208e" \
      org.opencontainers.image.licenses="MIT AND Apache-2.0" \
      ai.vonkforge.runtime-interface="v1" \
      io.vonk.mia-source-archive-sha256="e72b7e5cb2a11a607b0991098ecdfea685d2100af67eaa7cd88f1e4e1c2e1105" \
      io.vonk.anemll-source-revision="47503f8e38dadd4dededca798150db2619594fce" \
      io.vonk.dspark-loader-sha256="efe33c32d37ed7f26d869d94626f1415906d31218ec0ee44d79bb2b815b8cf39" \
      io.vonk.vision-hotfix-sha256="0f9ae3c8d07f29fff6c5bb27c3ced4e7cd3d7b87724a1300e9b6386469ed9362"

COPY encoding /opt/vonk-build/encoding
COPY patches /opt/vonk-build/patches
COPY patches/vision_exp /opt/dspark-patches/vision_exp
COPY licenses /opt/vonk/licenses
COPY --chmod=0755 apply-build-patches.py verify-dspark-runtime.py /opt/vonk-build/
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN install --directory --mode=0770 --owner=10001 --group=10001 /state /outputs \
    && python3 /opt/vonk-build/verify-dspark-runtime.py \
    && python3 /opt/vonk-build/apply-build-patches.py \
    && python3 /opt/vonk-build/patches/hotfix-vllm-issue136-xgrammar-termination.py --status \
    && rm -rf /opt/vonk-build \
    && test -x /opt/vonk/bin/vllm \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    HOME=/state
WORKDIR /state
# Refresh both trust stores after upstream installs; fail the build on stale roots.
COPY requirements-tls.lock verify-runtime-tls.py /tmp/runtime-tls/
RUN apt-get update \
    && apt-get install -y --no-install-recommends --reinstall ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir --no-deps --require-hashes -r /tmp/runtime-tls/requirements-tls.lock \
    && python3 /tmp/runtime-tls/verify-runtime-tls.py \
    && rm -rf /tmp/runtime-tls

USER 10001:10001
RUN python3 -c 'import getpass, os, pwd; assert os.getuid() == 10001; assert pwd.getpwuid(10001).pw_uid == 10001; assert getpass.getuser()'
ENTRYPOINT []
