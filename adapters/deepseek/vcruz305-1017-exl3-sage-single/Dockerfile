# DeepSeek-V4.1-Flash-1017 SAGE EXL3 1.62 bpw on one DGX Spark (vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe 8036aa74).
# The creator's base image (dsv41-1017-ds41f:r6w) is distributed only as an 8.8 GB Hugging Face tar.zst that the platform cannot
# pull, so this file is the kit's own from-source path: justincasey's base/docker/Dockerfile (spark-ds41f r6w, upstreams pinned at
# 2dc95329) followed by vcruz305's serving layer (docker/Dockerfile: vllm-exl3, exllamav3 mixed-K, TP1 decode tiles, the
# dsv41_exl3_shim quantization name). Both files are converted to the platform's static build policy and otherwise unchanged:
#   * every FROM is the literal base digest, and each linear stage chain (runtime, runtime-tilelang, runtime-tilelang-sparknet,
#     dsv41, the serving layer) is one stage with the same instructions in the same order; the dsv41-vllm stage is folded into fetch
#     (it only adds its own export directory and mutates src/vllm, which no other stage reads);
#   * every heredoc body is a file under build/ (byte for byte), run as `RUN python3 /tmp/<file>`; ds41f-prepare is COPY'd;
#   * the optional github_token secret mount is dropped (anonymous clones; the cmake configure retry stays);
#   * RUN --mount=type=bind,from=X becomes COPY --from=X into /tmp and a RUN;
#   * the apt io_uring library is not installed: the engines compile against posix-uring (tools/posix-uring), a header-only
#     stand-in that does one pread per read, so default seccomp suffices; the creators' "links liburing" self-checks become
#     "links no liburing" (build/check-b12x-runtime.py, build/check-posix-uring.py);
#   * the compile job counts are lowered for the hosted builder (4 vCPUs, 15 GiB RAM); architecture flags are unchanged (12.1a,
#     NCCL compute_121);
#   * the non-root user, the /opt/vonk/bin/vllm wrapper and ENTRYPOINT [] are ours, as is the sm_121 kernel scan.
# The fetch stage runs spark-ds41f's own bin/spark build prepare against its r6w lock: each upstream is fetched at its pinned
# commit, its patch series is applied with git am, and the patch-set fingerprint, patch head and tree must equal the manifest's.
#
# Resource use (spark-ds41f's build policy): a source build of vLLM native extensions, TileLang/TVM and NCCL; about 100 GB of disk.

# ── pins ──────────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf
ARG BASE_IMAGE_NAME=docker.io/vllm/vllm-openai
ARG BASE_IMAGE_DIGEST=sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf
ARG VLLM_BASE_SOURCE_REVISION=af1c01499b289be555c475669ba50a88e96d846e

ARG SPARK_DS41F_UPSTREAM=https://github.com/christopherowen/spark-ds41f
ARG SPARK_DS41F_REVISION=2dc95329610188451a5c5f5af98e5c392144809c
ARG SPARK_DS41F_LOCK=experiments/2026-10-18-r6w-swiglu/upstreams.lock.json
ARG SPARK_DS41F_PROFILE=experiments/2026-10-18-r6w-swiglu/r6w-tp4-1017.json

ARG VLLM_UPSTREAM=https://github.com/local-inference-lab/vllm
ARG VLLM_REF=integration/karmic-kraken-beta
ARG VLLM_REVISION=19f2c20ed4d64ae22bd009ceacadd85e2a81e354
ARG VLLM_SERIES=experiments/2026-10-18-r6w-swiglu/vllm/series
ARG VLLM_PATCHSET_SHA256=1879bb379390dcbd0f75a198dd92453afeaaf64f5d97018a5e79b936d20902e3
ARG VLLM_PATCH_HEAD=600fc0ebfa240c804d5f5e3e0116ce6b9207839c
ARG VLLM_TREE=8564da0b9a0f19d70ff75decb15ff89101886279
ARG VLLM_CANONICAL_UPSTREAM=https://github.com/vllm-project/vllm

ARG B12X_UPSTREAM=https://github.com/local-inference-lab/b12x
ARG B12X_REF=integration/karmic-kraken-beta
ARG B12X_REVISION=236ddff04a5f5064084b46464800b70b34ec7d8c
ARG B12X_SERIES=experiments/2026-10-08-lil-rebase/b12x/series
ARG B12X_PATCHSET_SHA256=3b0eb04a5870302937679badb436bfc9aa4afadd19333eb96d33148004ea1f09
ARG B12X_PATCH_HEAD=69e5795e2141fbb5fa2b3af096d51e754939bb41
ARG B12X_TREE=1a647a311ca55d45070ab83d60668d9c8fb8886c

ARG NCCL_UPSTREAM=https://github.com/NVIDIA/nccl
ARG NCCL_REF=refs/tags/v2.30.7-1
ARG NCCL_REVISION=73cf112295c33aee2b895f329f592f2a9b4b0f97
ARG NCCL_SERIES=experiments/2026-10-03-balanced-policy/nccl-adaptive/series
ARG NCCL_PATCHSET_SHA256=06f133b1709622d64b0ab87fc32f29d167f4617ca9386117596008d400a8d40c
ARG NCCL_PATCH_HEAD=eeacf1c6343f0499e4e7ab20ec7eecfb6f62eda6
ARG NCCL_TREE=6af10aa730a64fcee8781d62340a9c9184e79469

ARG TILELANG_UPSTREAM=https://github.com/tile-ai/tilelang
ARG TILELANG_REF=main
ARG TILELANG_REVISION=b95ee4ffb52ae26e15619cad38c6f41e7e0a2b46
ARG TILELANG_SERIES=experiments/2026-10-17-r6u/tilelang/series
ARG TILELANG_PATCHSET_SHA256=7d097972b85ae6817f2e7314ed3446f9d61d3e77691c2d0097dc18548d11ee07
ARG TILELANG_PATCH_HEAD=a0e66d71fe7617216f2de5d86c3b088595367387
ARG TILELANG_TREE=222b895d04ee6cd75179155d06e0c0d450636f51
ARG TILELANG_VERSION=0.1.15

ARG TILE_KERNELS_UPSTREAM=https://github.com/deepseek-ai/TileKernels
ARG TILE_KERNELS_REF=main
ARG TILE_KERNELS_REVISION=66258df6175d2f630ffecb04c5ab66bff8a2ae6a
ARG TILE_KERNELS_SERIES=patches/tile_kernels/series
ARG TILE_KERNELS_PATCHSET_SHA256=3fed91bbd083d38b08de1bd7ebe25dbc7aec9d36f4e2b8f7adbfb497b128c22c
ARG TILE_KERNELS_PATCH_HEAD=edfa64d018bc64006fcb07c95c32fa196363c0ae
ARG TILE_KERNELS_TREE=cb2fa41c8af290204c81c547ed678ebf957cca53
ARG TILE_KERNELS_VERSION=2.0.0

ARG SPARKNET_UPSTREAM=https://github.com/christopherowen/dgx-spark-networking
ARG SPARKNET_REF=timeout-stop
ARG SPARKNET_REVISION=3bc8d10104a5012244a0424d7dfc1107a72eafab
ARG SPARKNET_SERIES=-
ARG SPARKNET_PATCHSET_SHA256=e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
ARG SPARKNET_PATCH_HEAD=3bc8d10104a5012244a0424d7dfc1107a72eafab
ARG SPARKNET_TREE=f72599fa212e894d67ee114553e628ab05daf2da
ARG SPARKNET_VERSION=0.3.0

ARG CUTLASS_UPSTREAM=https://github.com/NVIDIA/cutlass
ARG CUTLASS_REVISION=cb4247394dd82148787aed73e5dc7cef33cbf862
ARG CUTLASS_RELEASE_TAG=v4.7.1

# Ours, applied in this order after his vLLM series (Apache-2.0, this repository). docker/ must hold exactly
# these patches, DSV41_VLLM_POST_OVERLAY_PATCHES, LINEAR_BUDGET_PATCH and KV_PERSIST_PATCH below.
ARG DSV41_VLLM_PATCHES="0001-deepseek-v41-reasoning-effort-minimal-medium.patch 0002-model-loader-weight-snapshots.patch"

# Then Christopher Owen's lossless MoE scale-codes overlay (spark-ds41f experiments/2026-10-18-scale-codes,
# Apache-2.0): the routed experts' UE8M0 scales kept as a base exponent per row and 4-bit codes, which the
# grouped GEMMs expand into the same scale words (outputs bit-identical to r6w's). Three whole files over
# r6w's (r6v with 0181), at the newest commit touching overlay/, each pinned by sha256, committed with a
# fixed author, committer and date; SCALE_CODES_PATCH_HEAD/TREE are the head and tree after that commit.
ARG SCALE_CODES_REVISION=a250ace70424475c04b0919fc9a46f23c19e51da
ARG SCALE_CODES_OVERLAY=experiments/2026-10-18-scale-codes/overlay
ARG SCALE_CODES_FUSED_MOE_TILELANG_SHA256=fdb6f54d02859e7d7e1aee5fb9f73c5f0a2a9ac26ded4044911b8f10e5b25ddb
ARG SCALE_CODES_TILELANG_GEMM_SHA256=1b6b2c878fb79293d3cce291a27002d6d966718218d69a7461bf99a38a38556c
ARG SCALE_CODES_TILELANG_MOE_SHA256=b446d2fe12bc6c6785f8593148c21fb7036369e749488437f90f335b1f78bf6d
ARG SCALE_CODES_PATCH_HEAD=2847b6a14d611fbffdb43b9456288b77dcdbad5e
ARG SCALE_CODES_TREE=3302edb15da7502a82ac70f03dd9cd83b3f3e211

# Then ours again (Apache-2.0, this repository), applied in this order after the overlay commit:
# 0003 ports vllm-project/vllm#60945 (Chauncey), no lookahead past the last logit in block verification's
# residual mass; 0004 ports vllm-project/vllm#61161 (DimensionSTP), fresh acceptance draws per draft step
# under block verification; 0005 keeps one CED decoder row for a non-final prefill chunk no later window
# reads (DSV41_CED_SKIP_NONFINAL=0 turns it off); 0006 drops the pre-collapse MTP buffer DSpark never reads.
ARG DSV41_VLLM_POST_OVERLAY_PATCHES="0003-spec-decode-block-residual-mass-lookahead-bound.patch 0004-spec-decode-block-verification-step-seeds.patch 0005-deepseek-v41-ced-skip-unread-nonfinal-chunks.patch 0006-deepseek-v41-dspark-no-mtp-hidden-buffer.patch"

# Then Christopher Owen's DSpark profile-replays overlay (spark-ds41f
# experiments/2026-10-18-dspark-profile-replays, Apache-2.0; his vLLM branch dspark-profile-replays 6bc1fc400,
# not published): adaptive verification times each graph-captured shape in 25 sweeps instead of 5 (the eager
# tail keeps 5), cutting the 16-96-row cost slopes' build-to-build noise 4-5x, and its pin key names the
# replay counts, so 5-replay curves are never reused. One whole file over r6w's (untouched by 0001-0006), at
# the commit adding overlay/, pinned by sha256, committed with a fixed author, committer and date;
# PROFILE_REPLAYS_PATCH_HEAD/TREE are the head and tree after that commit.
ARG PROFILE_REPLAYS_REVISION=2ce3644fa361290b0024f0ee784b17b63c8b2882
ARG PROFILE_REPLAYS_OVERLAY=experiments/2026-10-18-dspark-profile-replays/overlay
ARG PROFILE_REPLAYS_ADAPTIVE_VERIFICATION_SHA256=f9f3876ff36cda8d8f4bcc1dcfcd10715bc8bb3b2a5d691565e05614f316e07c
ARG PROFILE_REPLAYS_PATCH_HEAD=a71931204cc1b49d7596ff12c570fc4ab2f86435
ARG PROFILE_REPLAYS_TREE=68c0d5fbba65c850490bfc1e125b95a774740de7

# Then Christopher Owen's DSpark draft budget by a fixed linear step cost (spark-ds41f
# experiments/2026-10-18-dspark-linear-budget, its overlay at LINEAR_BUDGET_OVERLAY_REVISION; promoted in r6x as
# vLLM 0184, experiments/2026-10-18-r6x/vllm at LINEAR_BUDGET_REVISION; Apache-2.0): each step's draft budget
# maximises expected tokens over 1 + row_cost * rows, row_cost the speculative config's
# adaptive_verification_row_cost (default 0.02; his screen also ran 0.035), in place of the boot-timed cost
# curves. It deletes the boot profile (so the profile-replays code above), the pinned curves
# (SPARK3_DSPARK_COST_DIR) and adaptive_verification_cost_scale. His overlay's whole model_runner.py would
# drop 0006's None guard, so 0184 is applied as a patch rebased onto the tree above (his authorship and date;
# the one conflict is the deleted 25-replay profile): adaptive_verification.py and speculative.py come out
# byte for byte his overlay's (sha256 pins below), model_runner.py is his plus 0006's guard.
# LINEAR_BUDGET_PATCH_HEAD/TREE are the head and tree after it.
ARG LINEAR_BUDGET_REVISION=03e8775c9f2c9a791bb49a2f4d97575020ec0f7b
ARG LINEAR_BUDGET_OVERLAY_REVISION=53ef3558f4ca56f244cfb1566061199ad9a8975f
ARG LINEAR_BUDGET_PATCH=0007-dspark-adaptive-verification-linear-row-cost-budget.patch
ARG LINEAR_BUDGET_ADAPTIVE_VERIFICATION_SHA256=cffcb7cd663d15f206688a17559e0424c24f145474ce0703635b2ae1d5f525c3
ARG LINEAR_BUDGET_SPECULATIVE_SHA256=62f0a2409d5c1a64ec1d364033cc2df0268e1aadccf390ac60a5be91ea1b88bd
ARG LINEAR_BUDGET_PATCH_HEAD=ef9987d130261b4e35126e9364b6befe77752a82
ARG LINEAR_BUDGET_TREE=0e9522300cd782c328ce4c7e053f55c99008a4c7

# Then ours (Apache-2.0, this repository): the prefix KV cache persists across restarts. With
# DSV41_KV_PERSIST=1 (default 0) `POST /v1/dsv41/kv_snapshot` saves every rank's KV backing buffer and the
# scheduler's prefix-cache table under /cache/dsv41/kv-snapshots, and the next start restores them after
# warmup, on every rank or none. DSV41_VLLM_PATCH_HEAD/TREE are the serving vLLM's head and tree after it.
ARG KV_PERSIST_PATCH=0008-v1-engine-persistent-prefix-kv-cache.patch
ARG DSV41_VLLM_PATCH_HEAD=97a1e5068fada891c2016ce7e0e3193e872a909e
ARG DSV41_VLLM_TREE=675d178dfd069b22dcfbdb44e3e8da73854e1a43

# Compile parallelism: spark-ds41f sizes jobs from MemAvailable (8 GiB per job, at most 10, two NVCC
# threads each; 8 on a GB10). The hosted builder has 4 vCPUs and 15 GiB of RAM plus swap, so 2.
ARG VLLM_COMPILE_JOBS=2
ARG VLLM_NVCC_THREADS=2
ARG TILELANG_COMPILE_JOBS=2
ARG NCCL_COMPILE_JOBS=2

# vcruz305's serving layer pins (docker/Dockerfile of the kit; scripts/env.sh declares the same refs).
ARG PLUGIN_REPO=https://github.com/vcruz305/vllm-exl3.git
ARG PLUGIN_REF=30e8b941376ce2d43bc8438439937c4809106635
ARG FORK_REPO=https://github.com/vcruz305/exllamav3.git
ARG FORK_REF=909ad8564fcb9e070caca8a65d380c741681e79c
ARG MOE_TILESIZE_K=16

# ── fetch: spark-ds41f's own build preparation at the pinned commit ─────────────────────────────────
FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf AS fetch

USER root
ENV PYTHONDONTWRITEBYTECODE=1
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

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

# Drives spark-ds41f's bin/spark functions; the only thing it substitutes is configuration(), which
# would otherwise need a site node map, with the pinned lock.
COPY --chmod=0755 build/ds41f-prepare /usr/local/bin/ds41f-prepare

ARG SPARK_DS41F_LOCK
ARG SPARK_DS41F_PROFILE
ARG BASE_IMAGE
ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_DIGEST
ARG VLLM_BASE_SOURCE_REVISION
ARG VLLM_UPSTREAM
ARG VLLM_REF
ARG VLLM_REVISION
ARG VLLM_SERIES
ARG VLLM_PATCHSET_SHA256
ARG VLLM_PATCH_HEAD
ARG VLLM_TREE
ARG VLLM_CANONICAL_UPSTREAM
ARG B12X_UPSTREAM
ARG B12X_REF
ARG B12X_REVISION
ARG B12X_SERIES
ARG B12X_PATCHSET_SHA256
ARG B12X_PATCH_HEAD
ARG B12X_TREE
ARG NCCL_UPSTREAM
ARG NCCL_REF
ARG NCCL_REVISION
ARG NCCL_SERIES
ARG NCCL_PATCHSET_SHA256
ARG NCCL_PATCH_HEAD
ARG NCCL_TREE
ARG TILELANG_UPSTREAM
ARG TILELANG_REF
ARG TILELANG_REVISION
ARG TILELANG_SERIES
ARG TILELANG_PATCHSET_SHA256
ARG TILELANG_PATCH_HEAD
ARG TILELANG_TREE
ARG TILELANG_VERSION
ARG TILE_KERNELS_UPSTREAM
ARG TILE_KERNELS_REF
ARG TILE_KERNELS_REVISION
ARG TILE_KERNELS_SERIES
ARG TILE_KERNELS_PATCHSET_SHA256
ARG TILE_KERNELS_PATCH_HEAD
ARG TILE_KERNELS_TREE
ARG TILE_KERNELS_VERSION
ARG SPARKNET_UPSTREAM
ARG SPARKNET_REF
ARG SPARKNET_REVISION
ARG SPARKNET_SERIES
ARG SPARKNET_PATCHSET_SHA256
ARG SPARKNET_PATCH_HEAD
ARG SPARKNET_TREE
ARG SPARKNET_VERSION
ARG CUTLASS_UPSTREAM
ARG CUTLASS_REVISION
ARG CUTLASS_RELEASE_TAG
RUN ds41f-prepare --check-pins

# One layer per part, so a network failure repeats only that part; `--all` then reuses every verified
# part, exports the contexts, lists the vLLM deletions and checks everything again (build_problems).
RUN ds41f-prepare --only vllm
RUN ds41f-prepare --only b12x
RUN ds41f-prepare --only nccl
RUN ds41f-prepare --only tilelang
RUN ds41f-prepare --only tile_kernels
RUN ds41f-prepare --only sparknet
RUN ds41f-prepare --only cutlass
RUN ds41f-prepare --only wheels
RUN set -eux; \
    ds41f-prepare --all; \
    mv "$(ds41f-prepare --directory)" /work/build; \
    mkdir -p /work/out; \
    ds41f-prepare --smoke > /work/out/smoke.py; \
    printf '%s\n' "$SPARK_DS41F_REVISION" > /work/out/REVISION


# ── (folded into fetch) dsv41-vllm: our patches, the scale-codes and profile-replays overlays, 0007 and 0008 on his vLLM series
ARG SPARK_DS41F_UPSTREAM
ARG SCALE_CODES_REVISION
RUN set -eux; \
    git init --quiet /work/scale-codes; \
    cd /work/scale-codes; \
    git remote add origin "$SPARK_DS41F_UPSTREAM"; \
    git fetch --quiet --depth 1 --no-tags origin "$SCALE_CODES_REVISION"; \
    git checkout --quiet --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "$SCALE_CODES_REVISION"; \
    test -z "$(git status --porcelain --untracked-files=all)"

ARG PROFILE_REPLAYS_REVISION
RUN set -eux; \
    git init --quiet /work/profile-replays; \
    cd /work/profile-replays; \
    git remote add origin "$SPARK_DS41F_UPSTREAM"; \
    git fetch --quiet --depth 1 --no-tags origin "$PROFILE_REPLAYS_REVISION"; \
    git checkout --quiet --detach FETCH_HEAD; \
    test "$(git rev-parse HEAD)" = "$PROFILE_REPLAYS_REVISION"; \
    test -z "$(git status --porcelain --untracked-files=all)"

ARG SPARK_DS41F_LOCK
ARG VLLM_PATCH_HEAD
ARG DSV41_VLLM_PATCHES
ARG DSV41_VLLM_POST_OVERLAY_PATCHES
ARG SCALE_CODES_OVERLAY
ARG SCALE_CODES_FUSED_MOE_TILELANG_SHA256
ARG SCALE_CODES_TILELANG_GEMM_SHA256
ARG SCALE_CODES_TILELANG_MOE_SHA256
ARG SCALE_CODES_PATCH_HEAD
ARG SCALE_CODES_TREE
ARG PROFILE_REPLAYS_OVERLAY
ARG PROFILE_REPLAYS_ADAPTIVE_VERIFICATION_SHA256
ARG PROFILE_REPLAYS_PATCH_HEAD
ARG PROFILE_REPLAYS_TREE
ARG LINEAR_BUDGET_PATCH
ARG LINEAR_BUDGET_PATCH_HEAD
ARG LINEAR_BUDGET_TREE
ARG KV_PERSIST_PATCH
ARG DSV41_VLLM_PATCH_HEAD
ARG DSV41_VLLM_TREE
COPY base/docker/*.patch /work/dsv41/
RUN set -eux; \
    test "$(cd /work/dsv41 && LC_ALL=C ls | tr '\n' ' ')" = "$DSV41_VLLM_PATCHES $DSV41_VLLM_POST_OVERLAY_PATCHES $LINEAR_BUDGET_PATCH $KV_PERSIST_PATCH "; \
    cd /work/build/src/vllm; \
    test "$(git rev-parse HEAD)" = "$VLLM_PATCH_HEAD"; \
    for patch in $DSV41_VLLM_PATCHES; do \
      git -c user.name="Justin Casey" -c user.email="4348807+justincasey@users.noreply.github.com" \
        -c commit.gpgsign=false am --quiet --committer-date-is-author-date "/work/dsv41/$patch"; \
    done; \
    overlay="vllm/model_executor/layers/fused_moe/tilelang.py $SCALE_CODES_FUSED_MOE_TILELANG_SHA256 \
      vllm/models/deepseek_v4_1/tilelang/gemm.py $SCALE_CODES_TILELANG_GEMM_SHA256 \
      vllm/models/deepseek_v4_1/tilelang/moe.py $SCALE_CODES_TILELANG_MOE_SHA256"; \
    set -- $overlay; \
    while [ "$#" -gt 0 ]; do \
      test -z "$(git diff --name-only "$VLLM_PATCH_HEAD" HEAD -- "$1")"; \
      echo "$2  /work/scale-codes/$SCALE_CODES_OVERLAY/$1" | sha256sum --check --strict; \
      cp "/work/scale-codes/$SCALE_CODES_OVERLAY/$1" "$1"; \
      git add -- "$1"; \
      shift 2; \
    done; \
    GIT_AUTHOR_NAME="Christopher Owen" GIT_AUTHOR_EMAIL="christopherowen@users.noreply.github.com" \
    GIT_AUTHOR_DATE="2026-10-18T09:26:40+02:00" GIT_COMMITTER_NAME="Justin Casey" \
    GIT_COMMITTER_EMAIL="4348807+justincasey@users.noreply.github.com" \
    GIT_COMMITTER_DATE="2026-10-18T09:26:40+02:00" \
      git -c commit.gpgsign=false commit --quiet \
        -m "DeepSeek V4.1 MoE: the routed experts' scales as row bases and 4-bit codes" \
        -m "Christopher Owen's scale-codes overlay, copied unmodified from spark-ds41f $SCALE_CODES_REVISION ($SCALE_CODES_OVERLAY): the three files over r6w's (r6v with 0181). The grouped GEMMs expand the codes into the same scale words, so every output bit is r6w's, from half the scale bytes."; \
    test "$(git rev-parse 'HEAD^{tree}')" = "$SCALE_CODES_TREE"; \
    test "$(git rev-parse HEAD)" = "$SCALE_CODES_PATCH_HEAD"; \
    for patch in $DSV41_VLLM_POST_OVERLAY_PATCHES; do \
      git -c user.name="Justin Casey" -c user.email="4348807+justincasey@users.noreply.github.com" \
        -c commit.gpgsign=false am --quiet --committer-date-is-author-date "/work/dsv41/$patch"; \
    done; \
    replays=vllm/v1/worker/gpu/spec_decode/adaptive_verification.py; \
    test -z "$(git diff --name-only "$VLLM_PATCH_HEAD" HEAD -- "$replays")"; \
    echo "$PROFILE_REPLAYS_ADAPTIVE_VERIFICATION_SHA256  /work/profile-replays/$PROFILE_REPLAYS_OVERLAY/$replays" \
      | sha256sum --check --strict; \
    cp "/work/profile-replays/$PROFILE_REPLAYS_OVERLAY/$replays" "$replays"; \
    git add -- "$replays"; \
    GIT_AUTHOR_NAME="Christopher Owen" GIT_AUTHOR_EMAIL="christopherowen@users.noreply.github.com" \
    GIT_AUTHOR_DATE="2026-10-18T10:07:44+02:00" GIT_COMMITTER_NAME="Justin Casey" \
    GIT_COMMITTER_EMAIL="4348807+justincasey@users.noreply.github.com" \
    GIT_COMMITTER_DATE="2026-10-18T10:07:44+02:00" \
      git -c commit.gpgsign=false commit --quiet \
        -m "DSpark cost profile: 25 replays of each decode shape (5 of the eager tail)" \
        -m "Christopher Owen's profile-replays overlay, copied unmodified from spark-ds41f $PROFILE_REPLAYS_REVISION ($PROFILE_REPLAYS_OVERLAY): adaptive_verification.py over r6w's (r6v's). Each graph-captured shape is timed in 25 sweeps instead of 5, the eager tail keeps 5, and the pin key names the replay counts, so curves timed with fewer replays are not reused."; \
    test "$(git rev-parse 'HEAD^{tree}')" = "$PROFILE_REPLAYS_TREE"; \
    test "$(git rev-parse HEAD)" = "$PROFILE_REPLAYS_PATCH_HEAD"; \
    git -c user.name="Justin Casey" -c user.email="4348807+justincasey@users.noreply.github.com" \
      -c commit.gpgsign=false am --quiet --committer-date-is-author-date "/work/dsv41/$LINEAR_BUDGET_PATCH"; \
    test "$(git rev-parse 'HEAD^{tree}')" = "$LINEAR_BUDGET_TREE"; \
    test "$(git rev-parse HEAD)" = "$LINEAR_BUDGET_PATCH_HEAD"; \
    git -c user.name="Justin Casey" -c user.email="4348807+justincasey@users.noreply.github.com" \
      -c commit.gpgsign=false am --quiet --committer-date-is-author-date "/work/dsv41/$KV_PERSIST_PATCH"; \
    git diff --check "$VLLM_PATCH_HEAD" HEAD; \
    test -z "$(git status --porcelain --untracked-files=all)"; \
    test "$(git rev-parse 'HEAD^{tree}')" = "$DSV41_VLLM_TREE"; \
    test "$(git rev-parse HEAD)" = "$DSV41_VLLM_PATCH_HEAD"; \
    test -z "$(git diff --name-only --diff-filter=D "$VLLM_PATCH_HEAD" HEAD)"; \
    ds41f-prepare --export /work/build/src/vllm /work/build/context/dsv41-vllm-source

# ── spark-ds41f docker/Dockerfile @ SPARK_DS41F_REVISION, contexts from `fetch` ────────────────────
# Canonical ARM64 nightly af1c0149, the same runtime base as the 2026-09-20
# weekend image. Local Inference Lab's karmic-kraken-beta vLLM sits seven
# commits after it with no native-source changes in between, so only the two

# stable CUDA extensions are rebuilt from the overlaid source.
FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf AS vllm-native-builder

USER root

COPY --from=fetch /work/build/context/vllm-source/ /workspace/vllm/
# CUTLASS headers come from a checkout of the tag vLLM pins.
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

COPY --from=fetch /work/build/context/tilelang-source/ /workspace/tilelang/
COPY --from=fetch /work/build/context/tile_kernels-source/ /workspace/tile_kernels/
ARG TILELANG_COMPILE_JOBS
ARG TILE_KERNELS_VERSION
RUN set -eux; \
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

# sparknet (christopherowen/dgx-spark-networking): the one-shot RoCE
# collectives vendored from B12X with spark's patches 0001-0011, used by the
# TileLang family instead of b12x.comm.roce. Pure Python plus a C proxy.
FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf AS sparknet-builder

COPY --from=fetch /work/build/context/sparknet-source/ /workspace/sparknet/
RUN python3 -m pip wheel --no-cache-dir --no-deps --wheel-dir /opt/sparknet /workspace/sparknet && \
    ls /opt/sparknet/dgx_spark_networking-*.whl

FROM docker.io/vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf AS runtime

ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_DIGEST
ARG VLLM_UPSTREAM
ARG VLLM_REF
ARG VLLM_REVISION
ARG VLLM_PATCH_HEAD
ARG VLLM_TREE
ARG B12X_UPSTREAM
ARG B12X_REF
ARG B12X_REVISION
ARG B12X_PATCH_HEAD
ARG B12X_TREE
ARG NCCL_UPSTREAM
ARG NCCL_REF
ARG NCCL_REVISION
ARG NCCL_PATCH_HEAD
ARG NCCL_TREE
ARG SPARK_DS41F_REVISION

LABEL org.opencontainers.image.source="https://github.com/christopherowen/spark-ds41f" \
      org.opencontainers.image.revision="${SPARK_DS41F_REVISION}" \
      org.opencontainers.image.base.name="${BASE_IMAGE_NAME}" \
      org.opencontainers.image.base.digest="${BASE_IMAGE_DIGEST}" \
      local.spark3.vllm.upstream="${VLLM_UPSTREAM}" \
      local.spark3.vllm.ref="${VLLM_REF}" \
      local.spark3.vllm.revision="${VLLM_REVISION}" \
      local.spark3.vllm.patch-head="${VLLM_PATCH_HEAD}" \
      local.spark3.vllm.tree="${VLLM_TREE}" \
      local.spark3.b12x.upstream="${B12X_UPSTREAM}" \
      local.spark3.b12x.ref="${B12X_REF}" \
      local.spark3.b12x.revision="${B12X_REVISION}" \
      local.spark3.b12x.patch-head="${B12X_PATCH_HEAD}" \
      local.spark3.b12x.tree="${B12X_TREE}" \
      local.spark3.nccl.upstream="${NCCL_UPSTREAM}" \
      local.spark3.nccl.ref="${NCCL_REF}" \
      local.spark3.nccl.revision="${NCCL_REVISION}" \
      local.spark3.nccl.patch-head="${NCCL_PATCH_HEAD}" \
      local.spark3.nccl.tree="${NCCL_TREE}"

USER root

# The serving vLLM is his r6w tree plus our patches, the scale-codes and profile-replays overlays, 0007 (his
# DSpark linear draft budget) and 0008 (our persistent prefix KV cache) (local.dsv41.vllm.* labels); the native
# extensions are built from his tree, which these Python-only changes leave unchanged.
COPY --from=fetch /work/build/context/dsv41-vllm-source/ /opt/spark3/candidate/vllm/
COPY --from=fetch /work/build/context/b12x-source/ /opt/spark3/candidate/b12x/
COPY --from=vllm-native-builder \
  /opt/vllm-native/vllm/_C_stable_libtorch.abi3.so \
  /opt/spark3/candidate/vllm/vllm/_C_stable_libtorch.abi3.so
COPY --from=vllm-native-builder \
  /opt/vllm-native/vllm/_moe_C_stable_libtorch.abi3.so \
  /opt/spark3/candidate/vllm/vllm/_moe_C_stable_libtorch.abi3.so
COPY --from=fetch /work/build/context/cutlass-dsl-wheels/ /tmp/cutlass-dsl-wheels/
COPY --from=nccl-builder /opt/nccl/ /opt/spark3/nccl/
RUN set -eux; \
    target=/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2; \
    test -f "$target"; \
    install -m 0755 /opt/spark3/nccl/libnccl.so.2 "$target"; \
    rm /opt/spark3/nccl/libnccl.so.2

# B12X and vLLM build native row readers at first use against `pkg-config liburing`. The sandbox blocks io_uring, so that answer is
# posix-uring: a header-only stand-in doing one pread per read (tools/posix-uring in the recipe library, byte-identical copy here).
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
    cp -a --no-clobber "$installed"/. /opt/spark3/candidate/vllm/vllm/; \
    while read -r relative; do \
      [ -n "$relative" ] || continue; \
      rm -f "/opt/spark3/candidate/vllm/vllm/$relative"; \
      echo "kept deletion: $relative"; \
    done < /tmp/vllm-deleted.txt; \
    rm -f /tmp/vllm-deleted.txt; \
    test -f /opt/spark3/candidate/vllm/vllm/third_party/flashmla/flash_mla_interface.py

# The nightly base installs CuTe DSL 4.7.1, which vLLM and quack-kernels require;
# patches/b12x moves B12X's exact pin to it. Reinstall the hash-locked wheel set
# (the same version) and B12X without dependencies.
RUN set -eux; \
    python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      /tmp/cutlass-dsl-wheels/*.whl; \
    rm -rf /tmp/cutlass-dsl-wheels; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps \
      /opt/spark3/candidate/b12x
COPY build/check-b12x-requirements.py /tmp/check-b12x-requirements.py
RUN python3 /tmp/check-b12x-requirements.py && rm /tmp/check-b12x-requirements.py

ENV PYTHONPATH=/opt/spark3/candidate/vllm:/opt/spark3/candidate/b12x

COPY build/check-b12x-runtime.py build/check-posix-uring.py /tmp/
RUN LIBRARY_PATH=/usr/local/cuda/lib64/stubs python3 /tmp/check-b12x-runtime.py \
 && LIBRARY_PATH=/usr/local/cuda/lib64/stubs python3 /tmp/check-posix-uring.py \
 && rm /tmp/check-b12x-runtime.py /tmp/check-posix-uring.py

# The TileLang kernel backend: the patched TileLang replaces the base image's
# tilelang 0.1.12 (vLLM's pin), and TileKernels is added, both without
# dependencies. Every requirement must already hold; no shared package moves
# back to suit one consumer (docs/upstreams.md).
ARG TILELANG_UPSTREAM
ARG TILELANG_REF
ARG TILELANG_REVISION
ARG TILELANG_PATCH_HEAD
ARG TILELANG_TREE
ARG TILELANG_VERSION
ARG TILE_KERNELS_UPSTREAM
ARG TILE_KERNELS_REF
ARG TILE_KERNELS_REVISION
ARG TILE_KERNELS_PATCH_HEAD
ARG TILE_KERNELS_TREE
ARG TILE_KERNELS_VERSION

LABEL local.spark3.tilelang.upstream="${TILELANG_UPSTREAM}" \
      local.spark3.tilelang.ref="${TILELANG_REF}" \
      local.spark3.tilelang.revision="${TILELANG_REVISION}" \
      local.spark3.tilelang.patch-head="${TILELANG_PATCH_HEAD}" \
      local.spark3.tilelang.tree="${TILELANG_TREE}" \
      local.spark3.tile_kernels.upstream="${TILE_KERNELS_UPSTREAM}" \
      local.spark3.tile_kernels.ref="${TILE_KERNELS_REF}" \
      local.spark3.tile_kernels.revision="${TILE_KERNELS_REVISION}" \
      local.spark3.tile_kernels.patch-head="${TILE_KERNELS_PATCH_HEAD}" \
      local.spark3.tile_kernels.tree="${TILE_KERNELS_TREE}"

COPY --from=tilelang-builder /opt/tilelang/ /tmp/tilelang-wheels/
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      /tmp/tilelang-wheels/tilelang-*.whl /tmp/tilelang-wheels/tile_kernels-*.whl
COPY build/check-tilelang.py /tmp/check-tilelang.py
RUN TILELANG_VERSION=${TILELANG_VERSION} TILE_KERNELS_VERSION=${TILE_KERNELS_VERSION} \
    python3 /tmp/check-tilelang.py \
 && rm -rf /tmp/check-tilelang.py /tmp/tilelang-wheels

# The TileLang runtime with sparknet's collectives. The proxy is built here,
# into SPARKNET_ROCE_CACHE_DIR, so no compiler runs when serving starts; the
# one-shot kernels compile on a node's first prepare into the mounted TileLang
# cache (sparknet 0.4 has TileLang kernels only; earlier pins also had CuTe).
ARG SPARKNET_UPSTREAM
ARG SPARKNET_REF
ARG SPARKNET_REVISION
ARG SPARKNET_PATCH_HEAD
ARG SPARKNET_TREE
ARG SPARKNET_VERSION

LABEL local.spark3.sparknet.upstream="${SPARKNET_UPSTREAM}" \
      local.spark3.sparknet.ref="${SPARKNET_REF}" \
      local.spark3.sparknet.revision="${SPARKNET_REVISION}" \
      local.spark3.sparknet.patch-head="${SPARKNET_PATCH_HEAD}" \
      local.spark3.sparknet.tree="${SPARKNET_TREE}"

COPY --from=sparknet-builder /opt/sparknet/ /tmp/sparknet-wheels/
RUN python3 -m pip install --no-cache-dir --no-deps /tmp/sparknet-wheels/dgx_spark_networking-*.whl \
 && rm -rf /tmp/sparknet-wheels
COPY build/check-sparknet.py /tmp/check-sparknet.py
# The proxy is built into /opt/sparknet/roce here so no compiler runs when serving starts; the recipe's environment names the same
# directory (read-only at run time; an ENV line here would trip the writable-state check, which cannot tell a built cache from state).
RUN SPARKNET_ROCE_CACHE_DIR=/opt/sparknet/roce SPARKNET_VERSION=${SPARKNET_VERSION} python3 /tmp/check-sparknet.py \
 && rm /tmp/check-sparknet.py

# ── dsv41: the image (ENTRYPOINT stays the base image's ["vllm", "serve"]) ─────────────────────────
ARG SPARK_DS41F_UPSTREAM
ARG SPARK_DS41F_REVISION
ARG DSV41_VLLM_PATCHES
ARG DSV41_VLLM_POST_OVERLAY_PATCHES
ARG DSV41_VLLM_PATCH_HEAD
ARG DSV41_VLLM_TREE
ARG SCALE_CODES_REVISION
ARG SCALE_CODES_OVERLAY
ARG SCALE_CODES_FUSED_MOE_TILELANG_SHA256
ARG SCALE_CODES_TILELANG_GEMM_SHA256
ARG SCALE_CODES_TILELANG_MOE_SHA256
ARG PROFILE_REPLAYS_REVISION
ARG PROFILE_REPLAYS_OVERLAY
ARG LINEAR_BUDGET_REVISION
ARG LINEAR_BUDGET_OVERLAY_REVISION
ARG LINEAR_BUDGET_PATCH
ARG LINEAR_BUDGET_ADAPTIVE_VERIFICATION_SHA256
ARG LINEAR_BUDGET_SPECULATIVE_SHA256
ARG KV_PERSIST_PATCH

LABEL org.opencontainers.image.title="dsv41-1017-ds41f" \
      org.opencontainers.image.description="DeepSeek-V4.1-Flash-1017, TP4 on a four-DGX-Spark RoCE ring: spark-ds41f r6w plus the dsv41-1017 patches (reasoning-effort aliases, weight snapshots, vLLM #60945 and #61161 block verification fixes, CED decoder skip of unread prefill chunks, no DSpark MTP buffer, persistent prefix KV cache), spark-ds41f's MoE scale-codes overlay and its DSpark draft budget by a fixed linear step cost (r6x's vLLM 0184)" \
      org.opencontainers.image.licenses="Apache-2.0" \
      local.dsv41.engine="spark-ds41f r6w" \
      local.dsv41.spark-ds41f.upstream="${SPARK_DS41F_UPSTREAM}" \
      local.dsv41.spark-ds41f.revision="${SPARK_DS41F_REVISION}" \
      local.dsv41.vllm.patches="${DSV41_VLLM_PATCHES}" \
      local.dsv41.vllm.post-overlay-patches="${DSV41_VLLM_POST_OVERLAY_PATCHES}" \
      local.dsv41.vllm.scale-codes.revision="${SCALE_CODES_REVISION}" \
      local.dsv41.vllm.scale-codes.overlay="${SCALE_CODES_OVERLAY}" \
      local.dsv41.vllm.profile-replays.revision="${PROFILE_REPLAYS_REVISION}" \
      local.dsv41.vllm.profile-replays.overlay="${PROFILE_REPLAYS_OVERLAY}" \
      local.dsv41.vllm.linear-budget.revision="${LINEAR_BUDGET_REVISION}" \
      local.dsv41.vllm.linear-budget.overlay-revision="${LINEAR_BUDGET_OVERLAY_REVISION}" \
      local.dsv41.vllm.linear-budget.patch="${LINEAR_BUDGET_PATCH}" \
      local.dsv41.vllm.kv-persist.patch="${KV_PERSIST_PATCH}" \
      local.dsv41.vllm.patch-head="${DSV41_VLLM_PATCH_HEAD}" \
      local.dsv41.vllm.tree="${DSV41_VLLM_TREE}"

# spark-ds41f's host-side helpers at the pinned commit, unmodified, for the controller to extract
# (memguard.sh / memguard-start.sh), and the GPU import smoke its `build image` runs after a build:
#   docker run --rm -i --gpus all --memory=16g --memory-swap=16g --ulimit=core=0 \
#     --entrypoint /usr/bin/python3 <image> /opt/spark-ds41f/smoke.py
COPY --from=fetch /work/spark-ds41f/scripts/ /opt/spark-ds41f/scripts/
COPY --from=fetch /work/spark-ds41f/LICENSE /opt/spark-ds41f/LICENSE
COPY --from=fetch /work/out/smoke.py /work/out/REVISION /opt/spark-ds41f/

# The patched TileLang carries the SM120-family block-scaled MMA on SM121 (patches/tilelang 0001-0003),
# and the serving vLLM carries our reasoning-effort aliases, the weight-snapshot loader hook, the
# scale-codes overlay's three files byte for byte, our patches 0003-0006, and the linear-budget overlay's
# adaptive_verification.py and speculative.py byte for byte (0007, which replaced the profile-replays file) with
# its model runner change beside 0006's guard, and our KV snapshot hooks (0008); none of these files is
# importable without a GPU, so all are read as files.
COPY build/check-dsv41.py /tmp/check-dsv41.py
RUN SCALE_CODES_FUSED_MOE_TILELANG_SHA256=${SCALE_CODES_FUSED_MOE_TILELANG_SHA256} \
    SCALE_CODES_TILELANG_GEMM_SHA256=${SCALE_CODES_TILELANG_GEMM_SHA256} \
    SCALE_CODES_TILELANG_MOE_SHA256=${SCALE_CODES_TILELANG_MOE_SHA256} \
    LINEAR_BUDGET_ADAPTIVE_VERIFICATION_SHA256=${LINEAR_BUDGET_ADAPTIVE_VERIFICATION_SHA256} \
    LINEAR_BUDGET_SPECULATIVE_SHA256=${LINEAR_BUDGET_SPECULATIVE_SHA256} \
    python3 /tmp/check-dsv41.py \
 && rm /tmp/check-dsv41.py

# ── vcruz305's serving layer (docker/Dockerfile of the kit) on the engine above ─────────────────────
# vllm-exl3 (Exl3MoEMethod / ATS loader), his exllamav3 fork (mixed-K unified MoE, aarch64 guards), the TP1 decode tile edits and
# the dsv41_exl3_shim that registers quantization exl3_dsv41. Nothing else in the engine is replaced.
ARG PLUGIN_REPO
ARG PLUGIN_REF
ARG FORK_REPO
ARG FORK_REF
ARG MOE_TILESIZE_K

# The two compiles below run with MAX_JOBS=2 (the kit's 8 lowered for the hosted builder); the image keeps the kit's MAX_JOBS=8
# afterwards. The architecture list is unchanged.
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.1a \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN set -eux; \
    if ! command -v git >/dev/null 2>&1; then \
        apt-get update; \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git; \
        rm -rf /var/lib/apt/lists/*; \
    fi; \
    git clone "$PLUGIN_REPO" /opt/vllm-exl3; \
    git -C /opt/vllm-exl3 checkout "$PLUGIN_REF"; \
    test "$(git -C /opt/vllm-exl3 rev-parse HEAD)" = "$PLUGIN_REF"

# The served MoE kernel instance is compiled at tile-K 16, a #define in the fork. At the pinned FORK_REF the header already reads 16;
# the build still rewrites it to MOE_TILESIZE_K and asserts the result, so the build argument is never silently ignored.
RUN set -eux; \
    git clone "$FORK_REPO" /opt/exllamav3; \
    git -C /opt/exllamav3 checkout "$FORK_REF"; \
    test "$(git -C /opt/exllamav3 rev-parse HEAD)" = "$FORK_REF"; \
    PATCHER=/opt/vllm-exl3/tools/patch_exllamav3_aarch64.py; \
    if [ -f "$PATCHER" ]; then \
        sed -i 's|raise SystemExit(f"expected one x86 pause block in {path}")|print(f"already guarded (x86 pause): {path}")|' "$PATCHER"; \
        grep -q 'already guarded (x86 pause)' "$PATCHER"; \
        python3 "$PATCHER" /opt/exllamav3/exllamav3/exllamav3_ext; \
    fi; \
    F=/opt/exllamav3/exllamav3/exllamav3_ext/quant/exl3_moe_common.cuh; \
    sed -i -E "s/#define MOE_TILESIZE_K [0-9]+/#define MOE_TILESIZE_K ${MOE_TILESIZE_K}/" "$F"; \
    grep -n "define MOE_TILESIZE_K" "$F"; \
    grep -qx "#define MOE_TILESIZE_K ${MOE_TILESIZE_K}" "$F"; \
    CUSPARSE_DIR="$(find /usr/local/lib/python3*/dist-packages/nvidia -name cusparse.h -exec dirname {} \; 2>/dev/null | head -1)"; \
    if [ -z "$CUSPARSE_DIR" ]; then echo "ERROR: cusparse.h not found under NVIDIA Python packages" >&2; exit 2; fi; \
    export CPATH="$CUSPARSE_DIR${CPATH:+:$CPATH}"; \
    cd /opt/exllamav3; \
    MAX_JOBS=2 python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

RUN set -eux; \
    CUSPARSE_DIR="$(find /usr/local/lib/python3*/dist-packages/nvidia -name cusparse.h -exec dirname {} \; 2>/dev/null | head -1)"; \
    export CPATH="$CUSPARSE_DIR${CPATH:+:$CPATH}"; \
    export EXL3_EXT_INCLUDE=/opt/exllamav3/exllamav3/exllamav3_ext; \
    cd /opt/vllm-exl3; \
    MAX_JOBS=2 python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

ENV MAX_JOBS=8

COPY docker/tp1_decode_tiles.py /opt/dsv41-tools/tp1_decode_tiles.py
RUN python3 /opt/dsv41-tools/tp1_decode_tiles.py \
        /opt/spark3/candidate/vllm/vllm/models/deepseek_v4_1/tilelang

COPY shim /opt/dsv41-shim
RUN set -eux; \
    cd /opt/dsv41-shim; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps .

# Build-time sm_121 kernel scan (scan-sm121.py: cubins sm_121[af] / sm_120[f], PTX not arch-specific and <= 121, or sm_121a; every
# file with device code needs a loadable image; each file's listing is echoed). Covered: the two extensions this layer compiles,
# exllamav3_ext and vllm_exl3_c, and the engine's own compiled outputs (vLLM's two stable extensions and the patched NCCL), each
# required to carry a loadable image. Not covered: the base image's other kernels (FlashInfer, DeepGEMM, FlashMLA, Triton kernels),
# which the creator serves as the base image publishes them, and TileLang kernels, which compile at run time.
COPY scan-sm121.py /opt/scan-sm121.py
RUN set -eux; \
    cd /tmp; \
    EXT_EXL3="$(python3 -c 'import torch, exllamav3_ext; print(exllamav3_ext.__file__)')"; \
    EXT_PLUGIN="$(python3 -c 'import torch, vllm_exl3_c; print(vllm_exl3_c.__file__)')"; \
    python3 /opt/scan-sm121.py --require "$EXT_EXL3" --require "$EXT_PLUGIN" \
      --require /opt/spark3/candidate/vllm/vllm/_C_stable_libtorch.abi3.so \
      --require /opt/spark3/candidate/vllm/vllm/_moe_C_stable_libtorch.abi3.so \
      --require /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2; \
    rm /opt/scan-sm121.py

LABEL local.dsv41.exl3.base="spark-ds41f r6w (this file's engine stages)" \
      local.dsv41.exl3.vllm-exl3="${PLUGIN_REF}" \
      local.dsv41.exl3.exllamav3="${FORK_REF}" \
      local.dsv41.exl3.moe-tilesize-k="${MOE_TILESIZE_K}" \
      local.dsv41.exl3.shim="dsv41_exl3_shim:register (quantization exl3_dsv41)" \
      org.opencontainers.image.source="https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe" \
      org.opencontainers.image.revision="8036aa741e2c2328035ed546f1f844b6797775f6" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT AND AGPL-3.0-only" \
      ai.vonkforge.runtime-interface="v1"

# The cold-expert alias builder and its shipped selection, run by the launcher into the persistent cache.
COPY tools/ats_build_alias.py configs/cold-alias-selection.json /opt/vonk/tools/
COPY --chmod=0755 vllm-launcher.py /opt/vonk/bin/vllm
COPY LICENSE /opt/vonk/LICENSE.kit
COPY base/LICENSE base/NOTICE /opt/vonk/licenses/spark-ds41f-r6w-base/

RUN chmod -R a+rX /opt/vonk \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
 && install --directory --owner=10001 --group=10001 /outputs /outputs/cache

ENV VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    HOME=/outputs/cache/home

WORKDIR /tmp
USER 10001:10001
ENTRYPOINT []
