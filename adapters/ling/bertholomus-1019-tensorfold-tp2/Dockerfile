# bertholomus's Ling-3.1-flash-1019 TensorFold TP2 recipe v0.2.0 (ling-3.1-tensorfold-tp2-2xgb10 6088f472). The creator
# publishes no image: README "Run it" serves the engine (bertholomus/TensorFold ling-3.1-tp2 at 72f16e48, locked in
# kit-lock.json) from its source tree on PYTHONPATH inside nvcr.io/nvidia/pytorch:26.07-py3, which is what this image
# does. The engine's CUDA extensions are JIT-built on the Spark for the GPU present (tensorfold/cuda/build.py passes
# -gencode=arch=compute_121,code=sm_121), by the kit's serve/prebuild.py before each start, so the image ships no
# prebuilt device code and there is nothing for a build-time sm_121 kernel scan to inspect.
FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/bertholomus/ling-3.1-tensorfold-tp2-2xgb10" \
      org.opencontainers.image.revision="6088f47241c434e4263043a97a25784b03b0d1df" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"
RUN curl -fsSL https://codeload.github.com/bertholomus/TensorFold/tar.gz/72f16e48fc501f31d0d1239459cb53cd5e43d36a -o /tmp/engine.tar.gz \
 && curl -fsSL https://codeload.github.com/bertholomus/ling-3.1-tensorfold-tp2-2xgb10/tar.gz/6088f47241c434e4263043a97a25784b03b0d1df -o /tmp/kit.tar.gz \
 && echo "c85dc45e52f47973298d9c63634aae05576c93acf6f0a9e52fe09b72787cd25f  /tmp/engine.tar.gz" | sha256sum -c - \
 && echo "3367f2d4ee3a6b83ffbf32bd5dea3c43e4c922ffca7c23c451663bf8e9a1a9e6  /tmp/kit.tar.gz" | sha256sum -c - \
 && mkdir -p /opt/tensorfold /opt/tensorfold/kit /opt/vonk/bin \
 && tar -xzf /tmp/engine.tar.gz --strip-components=1 -C /opt/tensorfold \
 && tar -xzf /tmp/kit.tar.gz --strip-components=1 -C /opt/tensorfold/kit \
 && rm -rf /opt/tensorfold/tests /opt/tensorfold/assets /tmp/*.tar.gz
COPY requirements.lock /opt/tensorfold/requirements.lock
# The serve path imports tokenizers and jinja2 on first use; import them and the Ling modules here so a missing
# dependency fails the build instead of the start.
RUN python3 -m pip install --no-deps --require-hashes --no-cache-dir -r /opt/tensorfold/requirements.lock \
 && PYTHONPATH=/opt/tensorfold/src python3 -c 'import tokenizers, jinja2, tensorfold.cli, tensorfold.cuda.server, tensorfold.families.bailing_v3, tensorfold.families.bailing_v3.cuda.app, tensorfold.families.bailing_v3.cuda.engine, tensorfold.families.glm5_next.cuda.split' \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
ENV PYTHONPATH=/opt/tensorfold/src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
