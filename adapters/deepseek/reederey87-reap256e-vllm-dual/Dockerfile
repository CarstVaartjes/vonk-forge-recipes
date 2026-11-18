# Reederey87 DeepSeek-V4.1-Flash REAP-256E for two DGX Sparks (vLLM TP=2, NVMe Engram), Vonk Forge packaging.
# https://github.com/Reederey87/deepseek-v41-flash-reap-2x-dgx-spark at 4ccddc8801757e3f26b162688ddd6bcdd0668bdb
#
# The build steps below are upstream's Dockerfile (the image dsv41-reap-selfbuild) with these differences: the
# base is named by its linux/arm64 manifest digest (upstream names the manifest-index digest of
# vllm/vllm-openai:deepseekv41-flash-0909, whose arm64 manifest is the one pinned here, as its comments record);
# only the overlay files the engine and the patches use are copied (upstream's build-time test suites, the host
# preflight scripts and start.sh are not part of the image); the image ENV keeps the Engram overlay off so
# build-time interpreters do not load it, and the recipe turns it on for the served processes; and the ENTRYPOINT
# is replaced by the Controller launcher. The creator publishes no image.
#
# Kernel-contract patches (each fails the build when its anchor has drifted):
#   1. patch_swa_page_size.py            64-token SWA cache pages (FlashInfer's SM120 sparse decode needs 64).
#   2. patch_flashinfer_prefill_pbsx.py  32-state compressed-page instantiation in the sparse-MLA prefill kernel,
#      mask_flashinfer_aot.py            and removal of the prebuilt AOT binary that would ignore that edit; the
#      build_flashinfer.py (ours)        patched module is compiled here at image build (the creator's
#                                        scripts/build_flashinfer.sh prebuilds it offline for the same reason: an
#                                        in-engine nvcc build raced memory and reset nodes) and copied into the
#                                        emptied AOT slot, so FlashInfer loads it with no lock, ninja or writes
#                                        under the read-only root.
#   3. patch_v41_kernel_block_size.py    the V4.1 sparse-MLA backends accept a 64-token kernel block on SM121.
#   4. patch_sm12x_indexer.py            Triton FP8 paged-MQA logits fallback (sm12x_mqa.py) for 32-state pages.
# Engram: engram_install.py / engram_disk.py skip the 189 GiB Engram tables while loading and gather rows with
# POSIX os.pread from the mounted shards (buffered reads, no io_uring, no Direct I/O, no CUDA mmap).
FROM docker.io/vllm/vllm-openai@sha256:d84a123255b822fc22508635218000187221794f59c0694c33b0650d1e377d58

LABEL org.opencontainers.image.source="https://github.com/Reederey87/deepseek-v41-flash-reap-2x-dgx-spark" \
      org.opencontainers.image.revision="4ccddc8801757e3f26b162688ddd6bcdd0668bdb" \
      org.opencontainers.image.licenses="MIT" \
      ai.vonkforge.runtime-interface="v1"

ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=2 \
    CUTE_DSL_ARCH=sm_121a \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    DSV41_ENGRAM_DISK=0 \
    DSV41_ENGRAM_STRICT=1 \
    PYTHONPATH=/opt/dsv41/overlay \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

# This kit overlays a vLLM tree; it does not rebuild vLLM. The base must already contain the V4.1 model.
RUN python3 -c "import pathlib, vllm; p = pathlib.Path(vllm.__file__).parent / 'models' / 'deepseek_v4_1'; assert p.is_dir(), f'base image vLLM lacks V4.1: {p}'; print('base vLLM V4.1 OK', vllm.__version__, p)"

COPY overlay/ /opt/dsv41/overlay/

RUN python3 /opt/dsv41/overlay/patch_swa_page_size.py
RUN python3 /opt/dsv41/overlay/patch_flashinfer_prefill_pbsx.py \
 && python3 /opt/dsv41/overlay/mask_flashinfer_aot.py
COPY build_flashinfer.py /tmp/build_flashinfer.py
RUN python3 /tmp/build_flashinfer.py && rm /tmp/build_flashinfer.py
RUN python3 /opt/dsv41/overlay/patch_v41_kernel_block_size.py
RUN python3 /opt/dsv41/overlay/patch_sm12x_indexer.py

# Persistent install for every interpreter (EngineCore and TP workers included), active only while
# DSV41_ENGRAM_DISK=1. Written after the patch steps so it cannot fire during them.
RUN python3 -c "from pathlib import Path; Path('/usr/local/lib/python3.12/dist-packages/dsv41_engram.pth').write_text('import dsv41_boot\n')"

# Vonk Controller entrypoint: binds the placement rendezvous to vLLM's multi-node arguments and starts
# upstream's overlay/serve.py. Upstream's SSH/docker launcher (start.sh) is not used.
COPY --chmod=0755 vllm-wrapper.py /opt/vonk/bin/vllm
COPY LICENSE /opt/vonk/LICENSE.upstream

RUN chmod -R a+rX /opt/dsv41 \
 && install --directory --mode=0755 /outputs \
 && if ! getent group 10001 >/dev/null; then groupadd --gid 10001 vonk; fi \
 && if ! getent passwd 10001 >/dev/null; then useradd --uid 10001 --gid 10001 --home-dir /outputs/cache/home --no-create-home --shell /usr/sbin/nologin vonk; fi \
 && chown -R 10001:10001 /outputs /opt/vonk

ENV VLLM_NO_USAGE_STATS=1

WORKDIR /opt/vonk
USER 10001:10001
ENTRYPOINT []
