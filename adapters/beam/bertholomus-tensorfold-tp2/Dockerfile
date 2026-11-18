# bertholomus's Beam-1.0-501B-A23B two-Spark TensorFold recipe v0.2.0, its tf/ lane:
# https://github.com/bertholomus/beam-1.0-2x-dgx-spark/tree/bdfa59fa326be3cafb675aba3ff2a7984ab7a20f/tf
# (the repository's gb10/ vLLM lane is a different engine and not used here). The creator publishes no image:
# tf/beam-tf-serve.sh runs the stock nvcr.io/nvidia/pytorch:26.07-py3 container and installs the TensorFold fork
# (bertholomus/TensorFold, branch beam-1.0-tp2, at the REV of tf/beam-tf.env, d00201b4, locked in kit-lock.json)
# with `pip install --no-deps --no-build-isolation`, which this image does once at build time, after our one patch
# (patches/: an integral number such as 17.0 is an integer tool parameter; upstream PR follows the hardware pass).
# The engine's CUDA extensions are JIT builds for the GPU present (tensorfold/cuda/build.py: one -gencode for the
# current device), as the creator's tf/beam-tf-build.sh builds them before the ranks start; tensorfold-serve runs that
# same build before serving. No kernel image is built here, so there is no build-time sm_121 scan.
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/bertholomus/beam-1.0-2x-dgx-spark" \
      org.opencontainers.image.revision="bdfa59fa326be3cafb675aba3ff2a7984ab7a20f" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT" \
      ai.vonkforge.runtime-interface="v1"
COPY patches /tmp/patches
RUN curl -fsSL https://codeload.github.com/bertholomus/TensorFold/tar.gz/d00201b420a81e467c681dc7fe8ca903e24d076b -o /tmp/engine.tar.gz \
 && echo "5677dfb8f2cc8ac79d310557cc4f8bbef23826bdd86abad6c3557aadd0617999  /tmp/engine.tar.gz" | sha256sum -c - \
 && mkdir /tmp/engine \
 && tar -xzf /tmp/engine.tar.gz --strip-components=1 -C /tmp/engine \
 && (cd /tmp/engine && for p in /tmp/patches/*.patch; do patch -p1 --batch --fuzz=0 < "$p"; done) \
 && python3 -m pip install --no-deps --no-build-isolation --no-cache-dir /tmp/engine \
 && python3 -c "import tensorfold.cli, tensorfold.families.beam.cuda.app, tensorfold.families.beam.cuda.engine, tokenizers, safetensors, jinja2, numpy, triton" \
 && tensorfold --version \
 && rm -rf /tmp/engine /tmp/engine.tar.gz /tmp/patches \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi
COPY --chmod=0755 tensorfold-serve /opt/vonk/bin/tensorfold-serve
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    HOME=/outputs/cache/home \
    XDG_CACHE_HOME=/outputs/cache \
    TORCH_EXTENSIONS_DIR=/outputs/cache/torch_extensions \
    TRITON_CACHE_DIR=/outputs/cache/triton \
    TMPDIR=/outputs/cache/tmp
WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
