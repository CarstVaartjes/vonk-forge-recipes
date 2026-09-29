#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. docker/ray_cluster.env

IMAGE=step37-stepfun-ray:cu130-v0221-arm64
CONTAINER=${CONTAINER:-step37-vllm}
PY=${PY:-python}

echo "=== Step3.7 NVFP4 Ray head start ($HEAD_IP) ==="
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
rm -f /dev/shm/*ray* /dev/shm/*vllm* 2>/dev/null || true

docker run -d --name "$CONTAINER" \
  --gpus all --ipc=host --network=host \
  --entrypoint bash \
  -v /home/r0b0tdgx:/home/r0b0tdgx:ro \
  -v /home/r0b0tdgx/projects/step-37-flash-nvfp4-vllm-sm121-docker-plan/evidence:/work/evidence \
  -w /work \
  -e VLLM_HOST_IP="$HEAD_IP" \
  -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  -e TP_SOCKET_IFNAME="$TP_SOCKET_IFNAME" \
  -e RAY_memory_usage_threshold="$RAY_memory_usage_threshold" \
  -e RAY_memory_monitor_refresh_ms="$RAY_memory_monitor_refresh_ms" \
  "$IMAGE" -lc "ray start --head --node-ip-address='$HEAD_IP' --port=6379 --dashboard-host=0.0.0.0 --disable-usage-stats --block >/work/evidence/runtime/ray-head.log 2>&1"

sleep 3
docker logs "$CONTAINER" | tail -n 20
