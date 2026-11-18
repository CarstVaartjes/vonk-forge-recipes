# bertholomus's Mistral Large 4 TensorFold TP4 recipe v0.2.0 (mistral-large-4-tensorfold-tp4-4xgb10 de33c39c). The
# creator publishes no image: its README runs the engine checkout (bertholomus/TensorFold mistral-large-4-tp4 at the
# README's commit f4a30b3e, locked in kit-lock.json) from source in the stock nvcr.io/nvidia/pytorch:26.07-py3
# container (PYTHONPATH=src, tools/ml4/cp4_serve.sh), with no pip installs, which is what this image does.
# serve/prebuild.py JIT-builds the engine's CUDA extensions for the GPU present (tensorfold.cuda.build reads the
# device's compute capability), so they cannot be built on a GPU-less image runner: the launcher runs that same pinned
# prebuild.py on each rank before the server starts, into the platform's writable cache, as the creator runs it before
# the first start. The image therefore carries no prebuilt kernels to scan.
FROM nvcr.io/nvidia/pytorch@sha256:7531d90bcbe0e43e1f7363029c7e145ce90eebeb494a7b4695fdba0329d7c3c3
LABEL org.opencontainers.image.source="https://github.com/bertholomus/mistral-large-4-tensorfold-tp4-4xgb10" \
      org.opencontainers.image.revision="de33c39c0f6fe3b29e5234628cb959a317204249" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"
RUN curl -fsSL https://codeload.github.com/bertholomus/TensorFold/tar.gz/f4a30b3ecc990ec760c4322653df9a6e49d91a70 -o /tmp/engine.tar.gz \
 && curl -fsSL https://codeload.github.com/bertholomus/mistral-large-4-tensorfold-tp4-4xgb10/tar.gz/de33c39c0f6fe3b29e5234628cb959a317204249 -o /tmp/kit.tar.gz \
 && echo "59c71eca41336214360d865241c8ae13451b329e36d39548275b6cd7ca85d25b  /tmp/engine.tar.gz" | sha256sum -c - \
 && echo "9d83bf4b0f73425ce9a10d844645ad9581ef8e945adee5107eba15b926af76e2  /tmp/kit.tar.gz" | sha256sum -c - \
 && mkdir -p /opt/tensorfold /tmp/recipe /opt/vonk/bin \
 && tar -xzf /tmp/engine.tar.gz --strip-components=1 -C /opt/tensorfold \
 && tar -xzf /tmp/kit.tar.gz --strip-components=1 -C /tmp/recipe \
 && cp /tmp/recipe/serve/prebuild.py /opt/tensorfold/prebuild.py \
 && cp /tmp/recipe/LICENSE /opt/tensorfold/RECIPE-LICENSE \
 && rm -rf /tmp/recipe /tmp/*.tar.gz
# The engine runs from its source tree, as cp4_serve.sh runs it; the CUDA sources the extensions build from stay there.
ENV PYTHONPATH=/opt/tensorfold/src
RUN python3 -c "import tensorfold.cuda.server, tensorfold.families.mistral_large4.cuda.engine" \
 && for f in families/mistral_large4/cuda/mla_rows.cu families/mistral_large4/cuda/score4.cu \
      families/mistral_large4/cuda/score8.cu families/mistral_large4/cuda/select_step.cu cuda/exl3/experts.cu \
      cuda/exl3/linear.cu cuda/exl3/l2_prefetch.cu cuda/rdma_gather.cu; do test -s /opt/tensorfold/src/tensorfold/$f; done \
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
