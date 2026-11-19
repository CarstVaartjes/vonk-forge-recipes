FROM nvcr.io/nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:d44d6dc249c2c8330d0b380c595a514f0e86137fcc717fe7a4389a25720ade37 AS build

# libibverbs-dev: ds4_tp.c includes <infiniband/verbs.h> only when it is present (the creator's RDMA transport).
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libibverbs-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/src
# Upstream commit archive with gguf-tools/imatrix, gguf-tools/quality-testing/data and speed-bench removed (unused by make cuda-spark; keeps the bundle small).
COPY vendor/ds4-b3f50180ec2c7324892be1f50bc83555ccb8d5e7.tar.gz /tmp/ds4.tar.gz
RUN test "$(uname -m)" = aarch64 \
    && echo "93e93dd013ee2ec95b87ac588a8a5907d7e237c0c35b805ee89c4f6b68bec6f2  /tmp/ds4.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/ds4.tar.gz \
    && mv "ds4-b3f50180ec2c7324892be1f50bc83555ccb8d5e7" ds4 \
    && make -C ds4 cuda-spark \
    && test -x ds4/ds4-server \
    && test -x ds4/ds4

FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/antirez/ds4" \
      org.opencontainers.image.revision="b3f50180ec2c7324892be1f50bc83555ccb8d5e7" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.source-archive-sha256="93e93dd013ee2ec95b87ac588a8a5907d7e237c0c35b805ee89c4f6b68bec6f2"

# ds4 loads libibverbs.so.1 at run time (dlopen); ibverbs-providers carries the ConnectX-7 (mlx5) provider.
RUN apt-get update \
    && apt-get install --no-install-recommends --yes libibverbs1 ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 vonk \
    && useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home \
        --shell /usr/sbin/nologin vonk \
    && install --directory --owner 10001 --group 10001 /opt/vonk/bin /state
# ds4-server is the coordinator (serves HTTP); the ds4 CLI is the only binary that runs --role worker.
COPY --from=build --chmod=0755 /opt/src/ds4/ds4-server /opt/vonk/bin/ds4-server
COPY --from=build --chmod=0755 /opt/src/ds4/ds4 /opt/vonk/bin/ds4
COPY --from=build /opt/src/ds4/LICENSE /opt/vonk/LICENSE.ds4
COPY --chmod=0755 ds4-serve /opt/vonk/bin/ds4-serve

ENV HOME=/state \
    HF_HUB_OFFLINE=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
