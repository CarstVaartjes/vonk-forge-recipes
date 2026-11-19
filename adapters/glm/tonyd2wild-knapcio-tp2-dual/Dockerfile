# tonyd2wild's two-Spark port (d061f26a, no licence file) of knapcio's GLM-5.3-Flash NVFP4 DFlash2 stack
# (knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 at 770d115, MIT; vLLM-derived files Apache-2.0). The stack's image is
# knapcio's Dockerfile.roce on tonyd2wild's published vLLM image ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2
# (the kit's default BASE_IMAGE, pinned by digest, which is the tag's digest today); that build (the b12x RoCEnante subset
# and the glm_roce shim) is reproduced stage for stage below. start.sh bind-mounts the overlay files over the image's
# vLLM; here they are installed into the image at the same paths instead, and the overlay tree sits at /overlay/overlay
# where the profile's absolute paths expect it. /cache, the kit's persistent cache mount, is a link to the platform's
# writable cache. The Controller placement wrapper (which also builds the kit's lossless8 target and FP8 drafter views
# from the immutable Models at start) takes the platform's launcher path. Base: the linux/arm64 image
# ghcr.io/tonyd2wild/vllm-glm53-flash, by digest.
# Stage 1: compile the RDMA proxy (b12x comm/roce/_roce_proxy.c, plain C over libibverbs). Any apt install happens
# here only; the final image gets the .so and keeps its own rdma-core untouched.
FROM ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6 AS proxy
COPY roce/b12x/b12x /opt/glm-roce-b12x/b12x
RUN set -eu; \
    if [ -f /usr/include/infiniband/verbs.h ] && command -v gcc >/dev/null; then echo "verbs.h + gcc present"; else \
      apt-get update; \
      v=$(dpkg-query -W -f='${Version}' libibverbs1 2>/dev/null || true); \
      apt-get install -y --no-install-recommends "libibverbs-dev=$v" gcc libc6-dev \
        || apt-get install -y --no-install-recommends libibverbs-dev gcc libc6-dev; \
    fi; \
    cd /opt/glm-roce-b12x && B12X_ROCE_CACHE_DIR=/proxy-cache python3 -c "from b12x.comm.roce import _proxy; print('proxy', _proxy._build())"

# Stage 2: the serving image.
FROM ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6

LABEL org.opencontainers.image.source="https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark" \
      org.opencontainers.image.revision="d061f26ad3ec5c3c04f64aad7516dbe62a8aa6af" \
      io.vonk.knapcio.revision="770d1153062aa916b06591411c61d5c593ea0f03" \
      ai.vonkforge.runtime-interface="v1"

ARG ROCE_DEFAULT=0
ARG ROCE_HCA=rocep1s0f0,roceP2p1s0f0
ARG ROCE_SPIN_LIMIT=300000000
ARG RUN_TESTS=1
ARG VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm

# The shim brings its own b12x; refuse to shadow one the base already ships.
RUN python3 -c "import importlib.util as u, sys; s = u.find_spec('b12x'); sys.exit(f'base image already has b12x at {s.origin}: reconcile before adding RoCEnante') if s else print('no b12x in base: ok')"

COPY roce/b12x/b12x /opt/glm-roce-b12x/b12x
COPY roce/b12x/LICENSE roce/b12x/PROVENANCE.json /opt/glm-roce-b12x/
COPY roce/glm_roce /opt/glm-roce/glm_roce
COPY roce/tests /opt/glm-roce/tests
COPY roce/glm_roce.pth /opt/glm-roce/glm_roce.pth
# Install the .pth next to vllm, i.e. in a directory site.py is known to process.
RUN python3 -c "import importlib.util as u, os, shutil; d = os.path.dirname(os.path.dirname(u.find_spec('vllm').origin)); shutil.copy('/opt/glm-roce/glm_roce.pth', d); print('glm_roce.pth ->', d)"

# Runtime defaults, as in the kit's Dockerfile.roce (a run-time variable overrides any of them). The kit also sets
# B12X_ROCE_CACHE_DIR to the prebuilt proxy directory below; the root filesystem is read-only here, so the wrapper
# copies that directory under /outputs/cache and points the variable there.
ENV GLM_ROCE_ALLREDUCE=${ROCE_DEFAULT} \
    B12X_ROCE_HCA=${ROCE_HCA} \
    B12X_ROCE_SPIN_LIMIT=${ROCE_SPIN_LIMIT}

COPY --from=proxy /proxy-cache /opt/glm-roce-b12x/proxy-cache
RUN chmod -R a+rX /opt/glm-roce /opt/glm-roce-b12x

RUN if [ "$RUN_TESTS" = 1 ]; then cd /opt/glm-roce \
 && env -u GLM_ROCE_ALLREDUCE CUDA_VISIBLE_DEVICES= python3 tests/test_glm_roce_cpu.py \
 && env -u GLM_ROCE_ALLREDUCE CUDA_VISIBLE_DEVICES= B12X_ROCE_CACHE_DIR=/opt/glm-roce-b12x/proxy-cache python3 tests/test_glm_roce_image.py; fi

# The kit's overlay tree (python hooks, tables, CUDA sources that JIT-build into /cache on first use), the scheduler
# policy the profile names, and the two conversion scripts the wrapper runs.
COPY overlay /overlay/overlay
COPY profiles/levers_policy.json /overlay/profiles/levers_policy.json
COPY scripts/glm_quant_mix.py scripts/drafter_fp8.py /overlay/scripts/

# start.sh's bind mounts over the image's vLLM (GLM5NEXT_PATCH=1, GLM_KPOOL_FIX=1, GLM_ARGMAX_CLAMP=1 and the always-on
# parser, prefix-cache, indexer and MoE-table files), installed at the same paths.
RUN set -eu; o=/overlay/overlay; \
    install -D -m 0644 $o/sparse_attn_indexer_kpool.py $VLLM_PKG/model_executor/layers/sparse_attn_indexer_kpool.py; \
    install -D -m 0644 $o/glm47_moe.py $VLLM_PKG/parser/glm47_moe.py; \
    install -D -m 0644 $o/abstract_parser.py $VLLM_PKG/parser/abstract_parser.py; \
    install -D -m 0644 $o/kv_cache_coordinator.py $VLLM_PKG/v1/core/kv_cache_coordinator.py; \
    install -D -m 0644 "$o/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json" "$VLLM_PKG/model_executor/layers/fused_moe/configs/E=288,N=512,device_name=NVIDIA_GB10,dtype=fp8_w8a8,block_shape=[128,128].json"; \
    install -D -m 0644 $o/glm5next_kda.py $VLLM_PKG/models/glm5next/nvidia/kda.py; \
    install -D -m 0644 $o/glm5next_model.py $VLLM_PKG/models/glm5next/nvidia/model.py; \
    install -D -m 0644 $o/glm5next_kpool_compress.py $VLLM_PKG/models/glm5next/nvidia/ops/kpool_compress.py; \
    install -D -m 0644 $o/glm5next_attention.py $VLLM_PKG/models/glm5next/nvidia/attention.py; \
    install -D -m 0644 $o/mla_indexer.py $VLLM_PKG/v1/attention/backends/mla/indexer.py; \
    install -D -m 0644 $o/mamba_hybrid.py $VLLM_PKG/v1/worker/gpu/model_states/mamba_hybrid.py; \
    install -D -m 0644 $o/gumbel.py $VLLM_PKG/v1/worker/gpu/sample/gumbel.py; \
    install -D -m 0644 $o/rejection_sampler_utils.py $VLLM_PKG/v1/worker/gpu/spec_decode/rejection_sampler_utils.py; \
    chmod -R a+rX /overlay

COPY licenses /opt/vonk/licenses/knapcio
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm

RUN { test -x /opt/vllm/.venv/bin/vllm || test -x /usr/local/bin/vllm; } \
    && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
    && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /state --no-create-home --shell /usr/sbin/nologin vonk; fi \
    && install --directory --mode=0755 /inputs \
    && install --directory --owner 10001 --group 10001 /state /outputs /outputs/cache /outputs/cache/triton \
    && ln -s /outputs/cache /cache

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/state \
    XDG_CACHE_HOME=/outputs/cache \
    VLLM_CACHE_ROOT=/outputs/cache/vllm \
    TRITON_CACHE_DIR=/outputs/cache/triton

WORKDIR /state
USER 10001:10001
ENTRYPOINT []
