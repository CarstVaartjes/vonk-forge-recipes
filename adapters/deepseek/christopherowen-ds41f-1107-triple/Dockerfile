# christopherowen/spark-ds41f release spark-ds41f-1126 (890f8654bb484cb4b5006f7e9c26d50cb60c3610) on three DGX Sparks.
# The creator publishes no image: `bin/spark build prepare` clones and patches every upstream on the host and `bin/spark build image` hands
# docker/Dockerfile the exported contexts. This file is that build, from source, converted to the platform's static build policy and otherwise
# the release's Dockerfile:
#   * a `fetch` stage clones spark-ds41f at the release commit and runs its own spark_ds41f.build preparation against its own
#     upstreams.lock.json (each upstream at its pinned commit, its patch series applied with git am, the patch-set fingerprint, patch head and
#     tree checked against the source manifest), and its exports replace the named build contexts;
#   * every FROM is the literal base digest; the release's lock lists CUDA toolkit 13.5.37, so every stage the release starts from its
#     cuda-13-5 stage (the four builders and runtime-core) begins with that stage's instructions instead: its ENV and its RUN, whose
#     body is build/install-cuda-13-5; the chain runtime-core, runtime-core-cuda-13-5, runtime-tilelang and runtime-tilelang-sparknet
#     is one stage with the same instructions in the same order (the lock lists no B12X, so the release builds no `runtime` B12X stage; its
#     torch and runtime-no-flashinfer-kernels stages are not built either);
#   * every heredoc body is a file under build/ (byte for byte except the two edits below), run as `RUN python3 /tmp/<file>`;
#   * the optional github_token secret mount is dropped; the CUDA toolkit packages are bind-mounted from the fetch stage's export as the
#     release mounts its cuda-toolkit-debs context (no layer keeps them); the other build contexts became COPY --from=fetch;
#   * the apt io_uring library is not installed: vLLM's direct loader and Engram row reader compile against posix-uring (the recipe
#     library's tools/posix-uring), a header-only stand-in that does one pread per read, so default seccomp suffices; the creator's
#     "links liburing" self-check (build/check-runtime.py) asserts the opposite;
#   * image labels the release passes as build arguments are not set; the release's lock and manifest are copied to /opt/spark-ds41f instead;
#   * compile job counts are lowered for the hosted builder (4 vCPUs, 15 GiB of RAM); architecture flags are unchanged (12.1a, NCCL compute_121);
#   * the non-root user, the /opt/vonk/bin/vllm wrapper, ENTRYPOINT [] and the build-time sm_121 kernel scan are ours.

ARG BASE_IMAGE=docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf
ARG SPARK_DS41F_UPSTREAM=https://github.com/christopherowen/spark-ds41f
ARG SPARK_DS41F_REVISION=890f8654bb484cb4b5006f7e9c26d50cb60c3610
# Compile parallelism: the release sizes jobs from MemAvailable (8 GiB per job; 8 vLLM jobs on a GB10). The hosted builder has 4 vCPUs
# and 15 GiB of RAM plus swap, so 2.
ARG VLLM_COMPILE_JOBS=2
ARG VLLM_NVCC_THREADS=2
ARG NCCL_COMPILE_JOBS=2
ARG TILELANG_COMPILE_JOBS=2

# ── fetch: spark-ds41f's own build preparation at the release commit ───────────────────────────────
FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf AS fetch

USER root
ENV PYTHONDONTWRITEBYTECODE=1
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ARG BASE_IMAGE
ARG SPARK_DS41F_UPSTREAM
ARG SPARK_DS41F_REVISION
RUN set -eux; \
    git init --quiet /work/spark-ds41f; \
    cd /work/spark-ds41f; \
    git remote add origin "$SPARK_DS41F_UPSTREAM"; \
    git fetch --quiet --depth 1 --no-tags origin "$SPARK_DS41F_REVISION"; \
    git checkout --quiet --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "$SPARK_DS41F_REVISION"; \
    test -z "$(git status --porcelain --untracked-files=all)"

# Drives spark-ds41f's spark_ds41f.build functions; the only thing it substitutes is configuration(), which would otherwise need a site
# node map, with the release's lock and its tp3-triangle-4k shape.
COPY --chmod=0755 build/ds41f-prepare /usr/local/bin/ds41f-prepare
RUN ds41f-prepare --check-pins

# One layer per part, so a network failure repeats only that part; `--all` then reuses every verified part, exports the contexts, lists the
# vLLM deletions and checks everything again (build_problems).
RUN ds41f-prepare --only vllm
RUN ds41f-prepare --only nccl
RUN ds41f-prepare --only tilelang
RUN ds41f-prepare --only tile_kernels
RUN ds41f-prepare --only sparknet
RUN ds41f-prepare --only cutlass
RUN ds41f-prepare --only wheels
RUN ds41f-prepare --only cuda_toolkit
RUN set -eux; \
    ds41f-prepare --all; \
    mv "$(ds41f-prepare --directory)" /work/build; \
    mkdir -p /work/out; \
    ds41f-prepare --versions > /work/out/versions.env; \
    cat /work/out/versions.env

# CUDA 13.5 (nvcc/ptxas 13.5.37, libcudart 13.5.37, libcublasLt 13.10.0.20) over the
# base image's CUDA 13.0.2, from the lock's packages (requirements/cuda-13-5-sbsa.txt,
# verified by `bin/spark build prepare`). dpkg installs exactly that set with no
# network access, so a dependency the set lacks fails the build. /usr/local/cuda,
# /usr/local/cuda-13 and the pip nvidia/cu13 directory, which libtorch_cuda.so's
# DT_RPATH and cuda.pathfinder search ahead of /usr/local/cuda, are wired to
# /usr/local/cuda-13.5; nvcc reads TOP from its nvcc.profile because it does not
# resolve the nvidia/cu13/bin link it may be called through. The release builds this once as its cuda-13-5 stage; the
# platform pins every FROM by digest, so each stage below that the release starts from cuda-13-5 begins with the same ENV
# and RUN (build/install-cuda-13-5, with the packages bind-mounted from the fetch stage).
FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf AS vllm-native-builder
USER root

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_VERSION=13.5.0 \
    NV_CUDA_CUDART_VERSION=13.5.37-1 \
    NV_CUDA_LIB_VERSION=13.5.0-1 \
    NV_LIBCUBLAS_VERSION=13.10.0.20-1 \
    PATH=/usr/local/cuda-13.5/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.5/targets/sbsa-linux/lib:/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/lib64

COPY build/install-cuda-13-5 /tmp/install-cuda-13-5
RUN --mount=type=bind,from=fetch,source=/work/build/context/cuda-toolkit-debs,target=/tmp/cuda-toolkit-debs \
    sh /tmp/install-cuda-13-5 && rm /tmp/install-cuda-13-5

COPY --from=fetch /work/build/context/vllm-source/ /workspace/vllm/
# CUTLASS headers come from a host-side checkout of the tag vLLM pins.
COPY --from=fetch /work/build/context/cutlass-source/ /workspace/cutlass/
ENV VLLM_CUTLASS_SRC_DIR=/workspace/cutlass

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      cmake \
      git \
    && rm -rf /var/lib/apt/lists/*
ENV TORCH_CUDA_ARCH_LIST=12.1a
ARG VLLM_COMPILE_JOBS
ARG VLLM_NVCC_THREADS
# Only the two stable extensions are rebuilt; the external projects (DeepGEMM,
# FlashMLA, flash-attention, Triton kernels, ...) produce other native modules
# that are taken prebuilt from the base image. Skip their configure-time
# clones: GitHub intermittently answers anonymous clones from this network with
# 401. A configure retry remains for any residual fetch.
RUN set -eu; \
    python_path=$(python3 -c 'import sys; print(":".join(sys.path))'); \
    sed -i -E 's#^([[:space:]]*)include\(cmake/external_projects/[a-z0-9_]+\.cmake\)#\1message(STATUS "spark: external project skipped")#' \
      /workspace/vllm/CMakeLists.txt; \
    if grep -n 'include(cmake/external_projects/' /workspace/vllm/CMakeLists.txt; then exit 1; fi; \
    grep -q '#define CUTLASS_MAJOR 4' /workspace/cutlass/include/cutlass/version.h; \
    attempt=1; \
    until cmake -S /workspace/vllm -B /tmp/vllm-native -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DVLLM_TARGET_DEVICE=cuda \
      -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 \
      "-DVLLM_PYTHON_PATH=$python_path" \
      -DFETCHCONTENT_BASE_DIR=/workspace/vllm/.deps \
      -DCMAKE_JOB_POOL_COMPILE:STRING=compile \
      -DCMAKE_JOB_POOLS:STRING=compile=${VLLM_COMPILE_JOBS} \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
      -DNVCC_THREADS=${VLLM_NVCC_THREADS}; do \
      if [ "$attempt" -ge 8 ]; then exit 1; fi; \
      echo "cmake configure attempt $attempt failed; retrying in 45 s"; \
      attempt=$((attempt + 1)); sleep 45; \
    done; \
    cmake --build /tmp/vllm-native -j=${VLLM_COMPILE_JOBS} \
      --target _C_stable_libtorch _moe_C_stable_libtorch; \
    cmake --install /tmp/vllm-native --prefix /opt/vllm-native \
      --component _C_stable_libtorch; \
    cmake --install /tmp/vllm-native --prefix /opt/vllm-native \
      --component _moe_C_stable_libtorch; \
    test -f /opt/vllm-native/vllm/_C_stable_libtorch.abi3.so; \
    test -f /opt/vllm-native/vllm/_moe_C_stable_libtorch.abi3.so

# NCCL 2.30.7 (the base image's nvidia-nccl-cu13 wheel), rebuilt from the
# pinned release tag with patches/nccl for SM121 only. The runtime stage
# replaces the wheel's libnccl.so.2 with it.
FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf AS nccl-builder
USER root

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_VERSION=13.5.0 \
    NV_CUDA_CUDART_VERSION=13.5.37-1 \
    NV_CUDA_LIB_VERSION=13.5.0-1 \
    NV_LIBCUBLAS_VERSION=13.10.0.20-1 \
    PATH=/usr/local/cuda-13.5/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.5/targets/sbsa-linux/lib:/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/lib64

COPY build/install-cuda-13-5 /tmp/install-cuda-13-5
RUN --mount=type=bind,from=fetch,source=/work/build/context/cuda-toolkit-debs,target=/tmp/cuda-toolkit-debs \
    sh /tmp/install-cuda-13-5 && rm /tmp/install-cuda-13-5

COPY --from=fetch /work/build/context/nccl-source/ /workspace/nccl/
ARG NCCL_COMPILE_JOBS
RUN set -eux; \
    make -C /workspace/nccl -j${NCCL_COMPILE_JOBS} src.build \
      CUDA_HOME=/usr/local/cuda \
      NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"; \
    mkdir -p /opt/nccl; \
    cp -L /workspace/nccl/build/lib/libnccl.so.2 /opt/nccl/libnccl.so.2; \
    sha256sum /opt/nccl/libnccl.so.2 | cut -d" " -f1 > /opt/nccl/libnccl.so.2.sha256

# TileLang 0.1.15 from tile-ai/tilelang with patches/tilelang and its pinned
# submodules, and DeepSeek's TileKernels, as wheels for the TileLang kernel
# backend. TileLang's CUDA stubs leave kernel compilation to run time, so no
# GPU architecture is fixed here. The isolated build environment is held to
# the base image's z3-solver, which TileLang links and loads at run time.
# TileKernels takes its version from setuptools-scm, which an export without
# Git metadata cannot supply.
FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf AS tilelang-builder
USER root

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_VERSION=13.5.0 \
    NV_CUDA_CUDART_VERSION=13.5.37-1 \
    NV_CUDA_LIB_VERSION=13.5.0-1 \
    NV_LIBCUBLAS_VERSION=13.10.0.20-1 \
    PATH=/usr/local/cuda-13.5/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.5/targets/sbsa-linux/lib:/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/lib64

COPY build/install-cuda-13-5 /tmp/install-cuda-13-5
RUN --mount=type=bind,from=fetch,source=/work/build/context/cuda-toolkit-debs,target=/tmp/cuda-toolkit-debs \
    sh /tmp/install-cuda-13-5 && rm /tmp/install-cuda-13-5

COPY --from=fetch /work/build/context/tilelang-source/ /workspace/tilelang/
COPY --from=fetch /work/build/context/tile_kernels-source/ /workspace/tile_kernels/
ARG TILELANG_COMPILE_JOBS
COPY --from=fetch /work/out/versions.env /tmp/versions.env
RUN set -eux; \
    . /tmp/versions.env; \
    z3=$(python3 -c 'import importlib.metadata as m; print(m.version("z3-solver"))'); \
    printf 'z3-solver==%s\n' "$z3" > /tmp/tilelang-build-constraints.txt; \
    export PIP_CONSTRAINT=/tmp/tilelang-build-constraints.txt; \
    export PIP_BUILD_CONSTRAINT=/tmp/tilelang-build-constraints.txt; \
    CMAKE_BUILD_PARALLEL_LEVEL=${TILELANG_COMPILE_JOBS} \
    CMAKE_ARGS="-DUSE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc" \
    NO_VERSION_LABEL=1 \
      python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /opt/tilelang \
        /workspace/tilelang; \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_TILE_KERNELS=${TILE_KERNELS_VERSION} \
      python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /opt/tilelang \
        /workspace/tile_kernels; \
    ls /opt/tilelang/tilelang-*.whl /opt/tilelang/tile_kernels-*.whl

# sparknet (christopherowen/dgx-spark-networking) 0.6.0, with no local patches: the
# one-shot RoCE collectives, used by the TileLang family instead of
# b12x.comm.roce. Pure Python plus a C proxy.
FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf AS sparknet-builder
USER root

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_VERSION=13.5.0 \
    NV_CUDA_CUDART_VERSION=13.5.37-1 \
    NV_CUDA_LIB_VERSION=13.5.0-1 \
    NV_LIBCUBLAS_VERSION=13.10.0.20-1 \
    PATH=/usr/local/cuda-13.5/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.5/targets/sbsa-linux/lib:/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/lib64

COPY build/install-cuda-13-5 /tmp/install-cuda-13-5
RUN --mount=type=bind,from=fetch,source=/work/build/context/cuda-toolkit-debs,target=/tmp/cuda-toolkit-debs \
    sh /tmp/install-cuda-13-5 && rm /tmp/install-cuda-13-5

COPY --from=fetch /work/build/context/sparknet-source/ /workspace/sparknet/
RUN python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /opt/sparknet /workspace/sparknet && \
    ls /opt/sparknet/dgx_spark_networking-*.whl

# Everything the image holds but B12X: the candidate vLLM tree with its two
# rebuilt native extensions, the patched NCCL and the hash-locked CuTe DSL wheels.
# While the lock lists B12X (spark-ds41f-1025 and earlier), `runtime` adds it on
# top; a lock without it builds the TileLang runtime on this stage directly.

FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf AS runtime
USER root

ENV CUDA_HOME=/usr/local/cuda \
    CUDA_VERSION=13.5.0 \
    NV_CUDA_CUDART_VERSION=13.5.37-1 \
    NV_CUDA_LIB_VERSION=13.5.0-1 \
    NV_LIBCUBLAS_VERSION=13.10.0.20-1 \
    PATH=/usr/local/cuda-13.5/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.5/targets/sbsa-linux/lib:/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda/lib64

COPY build/install-cuda-13-5 /tmp/install-cuda-13-5
RUN --mount=type=bind,from=fetch,source=/work/build/context/cuda-toolkit-debs,target=/tmp/cuda-toolkit-debs \
    sh /tmp/install-cuda-13-5 && rm /tmp/install-cuda-13-5

# No usage reporting: vLLM's usage stats (stats.vllm.ai) stay off in every image and
# every container started from it, whatever the deployment's environment.
ENV VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1


USER root

COPY --from=fetch /work/build/context/vllm-source/ /opt/spark-ds41f/candidate/vllm/
COPY --from=vllm-native-builder \
  /opt/vllm-native/vllm/_C_stable_libtorch.abi3.so \
  /opt/spark-ds41f/candidate/vllm/vllm/_C_stable_libtorch.abi3.so
COPY --from=vllm-native-builder \
  /opt/vllm-native/vllm/_moe_C_stable_libtorch.abi3.so \
  /opt/spark-ds41f/candidate/vllm/vllm/_moe_C_stable_libtorch.abi3.so
COPY --from=fetch /work/build/context/cutlass-dsl-wheels/ /tmp/cutlass-dsl-wheels/
COPY --from=nccl-builder /opt/nccl/ /opt/spark-ds41f/nccl/
RUN set -eux; \
    target=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2; \
    test -f "$target"; \
    install -m 0755 /opt/spark-ds41f/nccl/libnccl.so.2 "$target"; \
    rm /opt/spark-ds41f/nccl/libnccl.so.2

# vLLM's direct loader and Engram disk row reader build native helpers against `pkg-config liburing`, prebuilt below into
# /opt/spark-ds41f/native so no compiler runs on first boot. The sandbox blocks io_uring, so that answer is posix-uring: a header-only
# stand-in doing one pread per read (tools/posix-uring in the recipe library, a byte-identical copy here).
COPY posix-uring /opt/vonk/posix-uring
ENV PKG_CONFIG_PATH=/opt/vonk/posix-uring
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Keep install-only generated modules and matching native extensions from the
# base install without overwriting any candidate source file, then remove the
# files the candidate deliberately deletes relative to the base image's
# canonical commit (listed by `bin/spark build prepare` from git).
COPY --from=fetch /work/build/context/vllm-deletions/deleted.txt /tmp/vllm-deleted.txt
RUN set -eux; \
    installed=$(python3 -c \
      'from pathlib import Path; import vllm; print(Path(vllm.__file__).parent)'); \
    cp -a --no-clobber "$installed"/. /opt/spark-ds41f/candidate/vllm/vllm/; \
    while read -r relative; do \
      [ -n "$relative" ] || continue; \
      rm -f "/opt/spark-ds41f/candidate/vllm/vllm/$relative"; \
      echo "kept deletion: $relative"; \
    done < /tmp/vllm-deleted.txt; \
    rm -f /tmp/vllm-deleted.txt; \
    test -f /opt/spark-ds41f/candidate/vllm/vllm/third_party/flashmla/flash_mla_interface.py

# The nightly base installs CuTe DSL 4.7.1, which vLLM and quack-kernels require.
# Reinstall the hash-locked wheel set (the same version).
RUN set -eux; \
    python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      /tmp/cutlass-dsl-wheels/*.whl; \
    rm -rf /tmp/cutlass-dsl-wheels
COPY build/check-cutlass-consumers.py /tmp/check-cutlass-consumers.py
RUN python3 /tmp/check-cutlass-consumers.py && rm /tmp/check-cutlass-consumers.py

ENV PYTHONPATH=/opt/spark-ds41f/candidate/vllm
# The prebuilt helpers go to vLLM's default prebuilt root, /opt/spark-ds41f/native (no VLLM_NATIVE_CACHE_DIR ENV line: the writable-state
# check cannot tell a build-time cache from state).

# NCCL_VERSION_CODE of the lock's NCCL tag (v2.30.7-1, the release's default).
ARG NCCL_VERSION_CODE=23007
COPY build/check-runtime.py /tmp/check-runtime.py
RUN LIBRARY_PATH=/usr/local/cuda/lib64/stubs python3 /tmp/check-runtime.py && rm /tmp/check-runtime.py

# The CUDA 13.5 build's runtime-core (runtime-core-cuda-13-5): the CuTe DSL wheel reinstall above rewrites files in nvidia/cu13 (it put a
# 13.2 libnvrtc.so.13 back), so wire it to CUDA 13.5 again, then check the toolchain and the libraries torch, cuda.pathfinder and
# ctypes resolve. The lock lists no B12X, so the TileLang runtime builds on this stage directly.
LABEL local.spark-ds41f.cuda.version="13.5.37"

RUN /usr/local/sbin/spark-wire-cuda-13-5
COPY build/check-cuda-13-5.py /tmp/check-cuda-13-5.py
RUN python3 /tmp/check-cuda-13-5.py && rm /tmp/check-cuda-13-5.py

# The TileLang kernel backend: the patched TileLang replaces the base image's
# tilelang 0.1.12 (vLLM's pin), and TileKernels is added, both without
# dependencies. Every requirement must already hold; no shared package moves
# back to suit one consumer (docs/upstreams.md). It builds on `runtime` while
# the lock lists B12X and on `runtime-core` once it does not
# (TILELANG_BASE_STAGE, set by `bin/spark build image`).

COPY --from=tilelang-builder /opt/tilelang/ /tmp/tilelang-wheels/
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      /tmp/tilelang-wheels/tilelang-*.whl /tmp/tilelang-wheels/tile_kernels-*.whl
COPY --from=fetch /work/out/versions.env /tmp/versions.env
COPY build/check-tilelang.py /tmp/check-tilelang.py
RUN . /tmp/versions.env \
 && TILELANG_VERSION="$TILELANG_VERSION" TILE_KERNELS_VERSION="$TILE_KERNELS_VERSION" python3 /tmp/check-tilelang.py \
 && rm -rf /tmp/check-tilelang.py /tmp/tilelang-wheels

# The TileLang runtime with sparknet's collectives. The proxy is built here,
# into SPARKNET_ROCE_CACHE_DIR, so no compiler runs when serving starts; the
# one-shot kernels compile on a node's first prepare into the mounted TileLang
# cache (sparknet 0.4 has TileLang kernels only; earlier pins also had CuTe).

# The proxy is built into /opt/sparknet/roce here so no compiler runs when serving starts; the recipe's environment names the same directory
# (an ENV line here would trip the writable-state check, which cannot tell a built cache from state).
COPY --from=sparknet-builder /opt/sparknet/ /tmp/sparknet-wheels/
RUN python3 -m pip install --no-cache-dir --no-deps /tmp/sparknet-wheels/dgx_spark_networking-*.whl \
 && rm -rf /tmp/sparknet-wheels
COPY build/check-sparknet.py /tmp/check-sparknet.py
RUN . /tmp/versions.env \
 && SPARKNET_ROCE_CACHE_DIR=/opt/sparknet/roce SPARKNET_VERSION="$SPARKNET_VERSION" python3 /tmp/check-sparknet.py \
 && rm /tmp/check-sparknet.py /tmp/versions.env

LABEL org.opencontainers.image.source="https://github.com/christopherowen/spark-ds41f" \
      org.opencontainers.image.revision="890f8654bb484cb4b5006f7e9c26d50cb60c3610" \
      org.opencontainers.image.licenses="Apache-2.0" \
      ai.vonkforge.runtime-interface="v1"

# The release's upstream lock, for the record of what was built.
COPY --from=fetch /work/spark-ds41f/upstreams.lock.json /opt/spark-ds41f/upstreams.lock.json
COPY --from=fetch /work/spark-ds41f/LICENSE /opt/spark-ds41f/LICENSE

# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or sm_121a; every file with
# device code needs a loadable image; each file's listing is echoed). Covered: the two vLLM native extensions this build compiles and the
# patched NCCL, each required to carry a loadable image. Not covered: the base image's other kernels (FlashInfer, DeepGEMM, FlashMLA, Triton
# kernels), which the creator serves as the base image publishes them, and TileLang kernels, which compile at run time.
COPY scan-sm121.py /opt/scan-sm121.py
RUN python3 /opt/scan-sm121.py \
      --require /opt/spark-ds41f/candidate/vllm/vllm/_C_stable_libtorch.abi3.so \
      --require /opt/spark-ds41f/candidate/vllm/vllm/_moe_C_stable_libtorch.abi3.so \
      --require /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 \
 && rm /opt/scan-sm121.py

COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY LICENSE /opt/vonk/LICENSE.upstream

RUN chmod -R a+rX /opt/vonk \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
 && install --directory --owner=10001 --group=10001 /outputs /outputs/cache

ENV HOME=/outputs/cache/home

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
