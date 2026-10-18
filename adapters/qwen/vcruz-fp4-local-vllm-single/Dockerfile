FROM eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595
LABEL org.opencontainers.image.source="https://github.com/vcruz305/Qwen3.8-Flash-Next-FP4-Spark-recipe" \
      org.opencontainers.image.revision="0d727c4ec145dadc14ac311508c4bb8c577e1a7a" \
      ai.vonkforge.runtime-interface="v1"
COPY LICENSE.upstream /opt/vonk/LICENSE.upstream
COPY --chmod=0755 vllm-serve /opt/vonk/bin/vllm
RUN python3 -c "import vllm" \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
