FROM nvcr.io/nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:d44d6dc249c2c8330d0b380c595a514f0e86137fcc717fe7a4389a25720ade37 AS build

WORKDIR /opt/src
COPY vendor/ds4-f4d03f6cf9f11c1e7b630bcb160853acfba7c52a.tar.gz /tmp/ds4.tar.gz
RUN test "$(uname -m)" = aarch64 \
    && echo "d7964a627d4f4ed490a5f20704fff7e4021398289a941ffe292ac28dfcb46d37  /tmp/ds4.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/ds4.tar.gz \
    && mv "ds4-f4d03f6" ds4 \
    && make -C ds4 cuda-spark \
    && test -x ds4/ds4-server

FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/antirez/ds4" \
      org.opencontainers.image.revision="f4d03f6cf9f11c1e7b630bcb160853acfba7c52a" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.source-archive-sha256="d7964a627d4f4ed490a5f20704fff7e4021398289a941ffe292ac28dfcb46d37"

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
