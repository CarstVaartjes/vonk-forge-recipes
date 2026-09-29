FROM nvcr.io/nvidia/cuda:13.4.1-devel-ubuntu24.04@sha256:d44d6dc249c2c8330d0b380c595a514f0e86137fcc717fe7a4389a25720ade37 AS build

WORKDIR /opt/src
# antirez/ds4 commit archive with gguf-tools/imatrix, gguf-tools/quality-testing/data and speed-bench removed
# (unused by make cuda-spark), then drowzeys' eight DSpark-port / GB10 patches applied in order
# (upstream README: git checkout 80ebbc3; git am patches/*.patch; make cuda-spark).
COPY vendor/ds4-80ebbc396aee40eedc1d829222f3362d10fa4c6c.tar.gz /tmp/ds4.tar.gz
COPY patches /tmp/patches
RUN test "$(uname -m)" = aarch64 \
    && echo "feb85fa05c617045e9eef1124db90d1e1337b994522da315dbcb9bc0ca3f97f8  /tmp/ds4.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/ds4.tar.gz \
    && mv "ds4-80ebbc396aee40eedc1d829222f3362d10fa4c6c" ds4 \
    && cd ds4 \
    && for patch in /tmp/patches/*.patch; do patch -p1 --no-backup-if-mismatch < "$patch" || exit 1; done \
    && make cuda-spark \
    && test -x ds4-server

FROM nvcr.io/nvidia/cuda:13.4.1-runtime-ubuntu24.04@sha256:2eaf346843c93ae617a718818f4a804e2c46c0d97a85392212ec5b310ee57962

LABEL org.opencontainers.image.source="https://github.com/drowzeys/keys-Antirez-Dwarf-Star-DSV4-on-DGX-Spark-GB10-Q2bit" \
      org.opencontainers.image.revision="98b73b7caa342fffd7871b5e77c803b7e087c67b" \
      org.opencontainers.image.licenses="MIT" \
      io.vonk.ds4-base-revision="80ebbc396aee40eedc1d829222f3362d10fa4c6c" \
      io.vonk.source-archive-sha256="feb85fa05c617045e9eef1124db90d1e1337b994522da315dbcb9bc0ca3f97f8"

RUN groupadd --gid 10001 vonk \
    && useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home \
        --shell /usr/sbin/nologin vonk \
    && install --directory --owner 10001 --group 10001 /opt/vonk/bin /state
COPY --from=build --chmod=0755 /opt/src/ds4/ds4-server /opt/vonk/bin/ds4-serve
COPY --from=build /opt/src/ds4/LICENSE /opt/vonk/LICENSE.ds4
COPY LICENSE /opt/vonk/LICENSE.drowzeys

ENV HOME=/state \
    HF_HUB_OFFLINE=1
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
