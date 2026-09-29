#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
. docker/ray_cluster.env

REMOTE=${REMOTE:-gn100-89ac}
IMAGE=step37-stepfun-ray:cu130-v0221-arm64
CONTAINER=${CONTAINER:-step37-vllm-worker}

echo "=== Step3.7 NVFP4 Ray worker start ($REMOTE) ==="
ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE" "bash -lc '
  docker rm -f $CONTAINER >/dev/null 2>&1 || true
  rm -f /dev/shm/*ray* /dev/shm/*vllm* 2>/dev/null || true
'"

ssh "$REMOTE" "docker run -d --name $CONTAINER \
  --gpus all --ipc=host --network=host \
  --entrypoint bash \
  -v /home/r0b0tdgx:/home/r0b0tdgx:ro \
  -e VLLM_HOST_IP=$WORKER_IP \
  -e GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \
  -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME \
  -e TP_SOCKET_IFNAME=$TP_SOCKET_IFNAME \
  -e RAY_memory_usage_threshold=$RAY_memory_usage_threshold \
  -e RAY_memory_monitor_refresh_ms=$RAY_memory_monitor_refresh_ms \
  $IMAGE -lc 'ray start --address=${HEAD_IP}:6379 --node-ip-address=$WORKER_IP --disable-usage-stats --block >/tmp/step37-ray-worker.log 2>&1'"

sleep 5
echo "Worker container started on $REMOTE. Checking Ray status from head..."
