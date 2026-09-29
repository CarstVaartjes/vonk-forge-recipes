#!/usr/bin/env bash
set -euo pipefail
IMAGE=${IMAGE:-ghcr.io/r0b0tlab/vllm-diffusiongemma-26b-nvfp4-sm121:cu130-gemma4-arm64}
docker build -t "$IMAGE" -f docker/Dockerfile .
