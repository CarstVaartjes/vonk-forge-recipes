#!/usr/bin/env bash
# Build-time form of the in-container setup that upstream's start.sh
# (container_setup_script) runs on every launch. Same commands, applied once so
# the image is read-only at run time. Like upstream, each step tolerates a
# missing target (the base image is the stock vllm/vllm-openai:v0.24.0).
set -u
DIST=/usr/local/lib/python3.12/dist-packages
rm -f "$DIST/nvidia/nccl/lib/libnccl.so.2" 2>/dev/null
ln -sf /usr/lib/aarch64-linux-gnu/libnccl.so.2 "$DIST/nvidia/nccl/lib/libnccl.so.2" 2>/dev/null
sed -i 's/^except ImportError:/except Exception:/' "$DIST/vllm/distributed/device_communicators/flashinfer_all_reduce.py" 2>/dev/null
sed -i 's/if lib_name in line:/if lib_name in line and "stub" not in line:/' "$DIST/flashinfer/comm/cuda_ipc.py" 2>/dev/null
exit 0
