FROM nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d AS build

WORKDIR /opt/src
COPY vendor/ds4-c1d4597a80e300b803dc642519718f2c999589da.tar.gz /tmp/ds4.tar.gz
RUN test "$(uname -m)" = aarch64 \
    && echo "39d409a06e03f6eb153c0a4026779138aa6185811ddc3e76c8d89505dceac36f  /tmp/ds4.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/ds4.tar.gz \
    && mv "ds4-c1d4597a80e300b803dc642519718f2c999589da" ds4 \
    && make -C ds4 cuda-spark \
    && test -x ds4/ds4-server

FROM nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04@sha256:36050649ad1acc5d3de2c26620191c25850fb12a5771b6c22996033003d952e4

LABEL org.opencontainers.image.source="https://github.com/antirez/ds4" \
      org.opencontainers.image.revision="c1d4597a80e300b803dc642519718f2c999589da" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.source-archive-sha256="39d409a06e03f6eb153c0a4026779138aa6185811ddc3e76c8d89505dceac36f"

RUN groupadd --gid 10001 vonk \
    && useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home \
        --shell /usr/sbin/nologin vonk \
    && install --directory --owner 10001 --group 10001 /opt/vonk/bin /state
COPY --from=build --chmod=0755 /opt/src/ds4/ds4-server /opt/vonk/bin/ds4-serve
COPY --from=build /opt/src/ds4/LICENSE /opt/vonk/LICENSE.ds4

ENV HOME=/state \
    HF_HUB_OFFLINE=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
