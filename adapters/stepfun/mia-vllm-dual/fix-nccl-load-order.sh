#!/bin/bash
# Build-time form of the NCCL load-order fix in upstream templates/launch-*.sh:
# point the pip-installed NCCL at the image's system libnccl.so.2 when both exist.
set -euo pipefail

system_nccl=""
for candidate in \
  "/usr/lib/$(gcc -print-multiarch 2>/dev/null || true)/libnccl.so.2" \
  "/usr/lib/aarch64-linux-gnu/libnccl.so.2" \
  "/usr/lib/x86_64-linux-gnu/libnccl.so.2"; do
  if [[ -e "$candidate" ]] || [[ -L "$candidate" ]]; then
    system_nccl="$candidate"
    break
  fi
done
[[ -z "$system_nccl" ]] && system_nccl=$(ldconfig -p 2>/dev/null | awk '/libnccl\.so\.2/ && $NF ~ /^\/usr\/lib\//{print $NF;exit}' || true)
if [[ -z "$system_nccl" ]]; then
  echo "[nccl] No system libnccl.so.2 found; skipping."
  exit 0
fi

py_nccl=""
for candidate in \
  "/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2" \
  "/usr/local/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2" \
  "/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2" \
  "/opt/env/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"; do
  if [[ -e "$candidate" ]] || [[ -L "$candidate" ]]; then
    py_nccl="$candidate"
    break
  fi
done
if [[ -z "$py_nccl" ]]; then
  echo "[nccl] No pip-installed NCCL found; skipping."
  exit 0
fi

if [[ "$(readlink -f "$system_nccl")" == "$(readlink -f "$py_nccl")" ]]; then
  echo "[nccl] NCCL already points to the system library."
  exit 0
fi
mv "$py_nccl" "${py_nccl}.spark-vllm-backup"
ln -s "$system_nccl" "$py_nccl"
echo "[nccl] $py_nccl now points to $system_nccl"
