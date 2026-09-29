FROM ghcr.io/r0b0tlab/nemotron-3.5-lightning-atlas-sm121@sha256:d00a317bbe8ada857836c89bde54b75449d2b91086594db7cce1be27683483e4

LABEL org.opencontainers.image.source="https://github.com/r0b0tlab/nemotron-3.5-lightning-atlas-sm121" \
      org.opencontainers.image.revision="4ecde533459d6fc1407ad9f889e3a13ce872d91f" \
      org.opencontainers.image.licenses="AGPL-3.0-only AND MIT" \
      ai.vonkforge.runtime-interface="v1"

# Base: upstream's immutable Atlas runtime image (native Rust/CUDA `spark` binary
# sha256:1111f86b..., Atlas source 3f929293; Avarok-Cybersecurity's Atlas is AGPL-3.0-only),
# used unchanged; only the user and licence copy are ours. Weights are mounted at runtime.
USER root
COPY LICENSE.upstream /opt/vonk/licenses/LICENSE.upstream
RUN if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache

ENV HOME=/state \
    XDG_CACHE_HOME=/outputs/cache
WORKDIR /state
USER 10001:10001
ENTRYPOINT []
