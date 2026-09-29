# Reproduce — DeepSeek-V4-Flash on Dual GB10 (SM121)

## Quick start (prebuilt image — recommended)

Skip the build entirely. Pull the image on BOTH nodes:

```bash
docker pull ghcr.io/r0b0tlab/vllm-dsv4-flash-gb10:cu130-sm121-arm64-dda4668b
```

ARM64 / CUDA 13.0 / `sm_121a`, torch 2.11.0+cu130, vLLM pinned `dda4668b`.
You supply the model weights (149GB, see "Model" below). Then on the HEAD node:

```bash
# Edit the cluster vars at the top first (IPs, interface, HCA, model dir).
# Latency profile (best single-stream):
MAX_MODEL_LEN=200000 MAX_NUM_SEQS=2 MAX_BATCHED=4096 bash scripts/run-dsv4-gb10.sh
# Throughput profile (best aggregate):
MAX_MODEL_LEN=65536 MAX_NUM_SEQS=16 MAX_BATCHED=8192 bash scripts/run-dsv4-gb10.sh
```

Cold start ~6 min. Then `curl http://localhost:8000/health` and benchmark
(step 3 below). If you use the prebuilt image you can skip steps 1-2.

---

## Hardware
- 2× NVIDIA DGX Spark (GB10, SM121 Blackwell), 128GB unified memory each
- Direct QSFP56 200G link, RoCE/NCCL over CX-7 (HCA `rocep1s0f0`)
- node0 192.168.100.10 (head), node1 192.168.100.11 (worker)

## 1. Build vLLM at the pinned commit (native SM121 fast path)

```bash
cd spark-vllm-docker
./build-and-copy.sh \
  --vllm-ref dda4668b59567416f86956cfe7bbc1eab371a61e \
  --rebuild-vllm --gpu-arch 12.1a \
  -t vllm-node-dsv4-pinned -j 20
```

Then copy the image to the worker node (more reliable than `-c` under automation):

```bash
docker save vllm-node-dsv4-pinned:latest | ssh 192.168.100.11 'docker load'
```

### Build fix baked into the Dockerfile (runner stage)
Installing the vLLM wheel pulls `torch` from the default PyPI index (CPU build),
clobbering the cu130 build and removing `libtorch_cuda.so`. The Dockerfile
re-pins CUDA torch AFTER the wheel install:

```dockerfile
RUN uv pip install --reinstall torch==2.11.0 torchvision torchaudio triton \
      --index-url https://download.pytorch.org/whl/cu130 && \
    python3 -c "import torch; assert torch.version.cuda; import vllm._C"
```

## 2. Launch (no custom mod — native family-120 gating)

The launch script exports `CUDA_HOME=/usr/local/cuda` and puts torch/lib on
`LD_LIBRARY_PATH`, then runs a single `vllm serve` line. Key flags:

```
--tensor-parallel-size 2 --enable-expert-parallel
--kv-cache-dtype fp8 --block-size 256
--speculative-config '{"model":"/mnt/model","num_speculative_tokens":2,"method":"deepseek_mtp"}'
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
--tokenizer-mode deepseek_v4 --enable-prefix-caching --load-format safetensors
```

NCCL: `NCCL_IB_DISABLE=0`, `NCCL_IB_HCA=rocep1s0f0`,
`{GLOO,TP,NCCL}_SOCKET_IFNAME=enp1s0f0np0`.

Profiles:
- Latency: `MAX_MODEL_LEN=200000 MAX_NUM_SEQS=2 MAX_BATCHED=4096`
- Throughput: `MAX_MODEL_LEN=65536 MAX_NUM_SEQS=16 MAX_BATCHED=8192`

### Launch gotchas
- `docker restart` BOTH containers before each launch attempt — port 29501
  (TCPStore) hits EADDRINUSE from zombie processes otherwise.
- Launch the worker (node1) first, then the head (node0).

## 3. Benchmark

```bash
python3 scripts/quality_smoke.py      # must be 5/5
python3 scripts/bench_dsv4.py --concurrencies 1,2,4,8,16 --max-tokens 256 --runs 3 --out results/run.json
```

Headline metric = c=1 single-stream per-request decode tok/s. All numbers are
the median of 3 runs.
