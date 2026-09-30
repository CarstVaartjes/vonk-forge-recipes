# Qwen3.6-27B (NVFP4) DFlash for DGX Spark

[![vLLM](https://img.shields.io/badge/vLLM-nightly-blue)](https://github.com/vllm-project/vllm)
[![Model](https://img.shields.io/badge/model-Qwen3.6--27B-informational)](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](LICENSE)
[![ARM64](https://img.shields.io/badge/arch-arm64-lightgrey)](#)
[![DFlash](https://img.shields.io/badge/speculative-dflash-orange)](https://github.com/nbasyl/DFlash)

A production-ready vLLM deployment wrapper for **[Qwen3.6-27B](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)** — an NVFP4-quantized dense model (~27B params) with **DFlash speculative decoding**.

This repo bundles a ready-to-run Docker container, a custom chat template, and start/stop scripts so you can spin up a fully OpenAI-compatible inference server in minutes.

<p>
<a href="https://x.com/MiaAI_lab" target="_blank">
  <img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" />
</a>
</p>
<p>
<a href='https://ko-fi.com/Z8Z3SPLOD' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
</p>

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Model** | Qwen3.6-27B-NVFP4 — NVFP4 quantised dense (~27B params) |
| **Vision** | Supports image input (multimodal) |
| **Quantization** | ModelOpt (`--quantization modelopt`) |
| **Inference Engine** | vLLM (nightly aarch64) with Flash Attention backend |
| **Speculative Decoding** | DFlash, draft model `z-lab/Qwen3.6-27B-DFlash`, 10 speculative tokens |
| **Context Window** | Up to **262 144 tokens** (256K) |
| **OpenAI-Compatible API** | `/v1/chat/completions`, `/v1/completions`, `/v1/models` |
| **Served Model Name** | `qwen36-27b-nvidia-nvfp4-dflash` |
| **Tool Use** | Qwen3-coder tool-call parser, auto tool choice enabled |
| **Thinking/Reasoning** | CoT / chain-of-thought with `<thinking>` block support (configurable) |
| **Reasoning Parser** | Qwen3-specific parser via `--reasoning-parser qwen3` |
| **Streaming** | Full SSE streaming support |
| **Prefix Caching** | Enabled via `--enable-prefix-caching` |
| **Chunked Prefill** | Enabled via `--enable-chunked-prefill` |
| **KV Cache** | bfloat16 (`--kv-cache-dtype bfloat16`) |
| **Custom Chat Template** | Full Jinja template with tool use and thinking support |
| **ARM64 Native** | vLLM nightly aarch64 image |

---

## 📊 Performance

![benchmark](benchmark.png)

### Decode Benchmark (Concurrency Scaling)

256 tokens output · 1–4 concurrent streams · total 33.7 s

| Concurrency | TTFT | Streams | Aggregate (tok/s) | Stream (tok/s) |
|---|---|---|---|---|
| ×1 | 133 ms | 1/1 | 45.6 | 45.6 |
| ×2 | 566 ms | 2/2 | 71.7 | 37.1 |
| ×3 | 718 ms | 3/3 | 86.0 | 33.9 |
| ×4 | 771 ms | 4/4 | 102.4 | 29.8 |

> **Aggregate** — total tok/s across all concurrent streams. **Stream** — per-stream average.

---

### Deep-Context Throughput

Deep-context throughput benchmark (salted prompts to avoid prefix cache):

| Config | Value |
|---|---|
| **base_url** | `http://127.0.0.1:8888` |
| **model** | `qwen36-27b-nvidia-nvfp4-dflash` |
| **input_lens** | `[2048, 8192, 16384, 32768, 65536]` |
| **output_len** | `256` |
| **concurrency** | `1` |
| **runs** | `1` |
| **reuse_prompt** | `False` (salted prompts, avoid prefix cache) |
| **chars_per_token** | ~5.94 |

### Benchmark Results

| target_in | prompt_tok | out_tok | ttft (s) | prefill (tok/s) | decode (tok/s) | e2e_out (tok/s) | total (s) |
|---|---|---|---|---|---|---|---|
| 2048 | 1830 | 256 | 1.820 | 1005.4 | 45.3 | 34.2 | 7.48 |
| 8192 | 7142 | 256 | 6.594 | 1083.0 | 41.1 | 20.0 | 12.82 |
| 16384 | 14231 | 256 | 13.152 | 1082.0 | 38.1 | 12.9 | 19.87 |
| 32768 | 28399 | 256 | 26.359 | 1077.4 | 33.7 | 7.5 | 33.96 |
| 65536 | 56727 | 256 | 57.096 | 993.5 | 26.6 | 3.8 | 66.71 |

---

## 📈 Performance Notes

- **84 % GPU memory** is allocated for weights (`--gpu-memory-utilization 0.84`). The NVFP4 quantisation via ModelOpt keeps the ~27B-parameter model within reach of consumer GPUs.
- **DFlash speculative decoding** (10 draft tokens from `z-lab/Qwen3.6-27B-DFlash`) can provide significant speedup over standard autoregressive generation.
- **Max 4 concurrent sequences** is conservative; increase `--max-num-seqs` and `--max-num-batched-tokens` if your GPU has spare headroom.
- **Prefix caching** dramatically improves throughput for repeated prompts / system prompts.
- **language-model-only** mode skips vision pipeline, reducing memory overhead.

---

## 📋 Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                    Your Host Machine                  │
│                                                      │
│  start.sh / stop.sh                                  │
│  chat_template.jinja  ← Custom Jinja template        │
│  .cache/              ← HuggingFace + Triton cache   │
│  .vllm.log            ← Container log                │
│  .vllm.pid            ← Container ID                 │
│                                                      │
│  ┌──────────────────────────────────────────────┐    │
│  │  Docker Container                             │    │
│  │  vllm/vllm-openai:nightly-aarch64             │    │
│  │  Qwen3.6-27B-NVFP4                            │    │
│  │  vLLM Server ← OpenAI API on :8888           │    │
│  │  DFlash Draft: z-lab/Qwen3.6-27B-DFlash       │    │
│  └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────┘
```

The container exposes an **OpenAI-compatible REST API** at `http://0.0.0.0:8888/v1`, so any client that speaks the OpenAI protocol (langchain, llama-cpp-python, oapi, custom HTTP) can connect directly.

---

## 🛠️ Prerequisites

| Requirement | Minimum | Notes |
|---|---|---|
| **OS** | Ubuntu 22.04+ / Debian 12+ | ARM64 / aarch64 required |
| **GPU** | NVIDIA GPU with ≥ 40 GB VRAM | Tested on 48 GB+ (e.g. RTX 6000, A100) |
| **CUDA** | CUDA 12.x compatible | NVIDIA driver ≥ 535 |
| **Docker** | 24.0+ | With [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| **curl** | Any | Used for readiness probes |
| **Disk** | ~50 GB free | Model weights + caches |

---

## 🚀 Quick Start

### 1. Clone & Navigate

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.6-27B-NVFP4-DFlash-DGX-Spark
cd Qwen3.6-27B-NVFP4-DFlash-DGX-Spark
```

### 2. (Optional) Set HuggingFace Token

If the model repo (`nvidia/Qwen3.6-27B-NVFP4`) requires authenticated access:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### 3. Start the Server

```bash
./start.sh
```

This will:
1. Check for Docker and curl on PATH
2. Create cache directories (HuggingFace + Triton)
3. Verify `chat_template.jinja` exists
4. Remove any stale container with the same name
5. Pull the latest `vllm/vllm-openai:nightly-aarch64` image
6. Launch the container with `--gpus all` and `--privileged`
7. Stream logs to `.vllm.log`
8. Poll `/v1/models` until the server is ready
9. Print the OpenAI base URL on success

**Expected output:**
```
Starting vLLM container for nvidia/Qwen3.6-27B-NVFP4 (DFlash speculative decoding)
Image: vllm/vllm-openai:nightly-aarch64
Served model name: qwen36-27b-nvidia-nvfp4-dflash
Listening on 0.0.0.0:8888
Writing progress to .vllm.log
...
vLLM is ready
OpenAI base URL: http://0.0.0.0:8888/v1
vLLM is ready and responding; shell is now free.
```

### 4. Test It

```bash
# Quick health check
curl http://0.0.0.0:8888/v1/models | jq

# Chat completion
curl -s http://0.0.0.0:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen36-27b-nvidia-nvfp4-dflash",
    "messages": [{"role": "user", "content": "Explain quantization in one sentence."}],
    "max_tokens": 256,
    "stream": false
  }' | jq
```

### 5. Stop the Server

```bash
./stop.sh
```

---

## ⚙️ Configuration

All configurable options live in [`start.sh`](start.sh). Key variables:

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `nvidia/Qwen3.6-27B-NVFP4` | HuggingFace model identifier |
| `SERVED_MODEL_NAME` | `qwen36-27b-nvidia-nvfp4-dflash` | Name used in API requests |
| `IMAGE` | `vllm/vllm-openai:nightly-aarch64` | vLLM Docker image tag |
| `CONTAINER_NAME` | `qwen3.6-27b-nvidia-nvfp4-dflash` | Docker container name |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8888` | HTTP port |
| `HF_TOKEN` | env var | HuggingFace auth token |

### Model Inference Parameters

| Flag | Value | Description |
|---|---|---|
| `--tensor-parallel-size` | `1` | Single-GPU (adjust for multi-GPU) |
| `--trust-remote-code` | — | Required by Qwen models |
| `--quantization` | `modelopt` | ModelOpt quantization loader |
| `--attention-backend` | `flash_attn` | Flash Attention kernel backend |
| `--kv-cache-dtype` | `bfloat16` | KV cache precision |
| `--gpu-memory-utilization` | `0.84` | 84 % of GPU memory for model weights |
| `--max-model-len` | `262144` | 256K context window |
| `--max-num-seqs` | `4` | Max concurrent sequences |
| `--max-num-batched-tokens` | `8192` | Max tokens per batch |
| `--enable-chunked-prefill` | — | Improves throughput |
| `--enable-prefix-caching` | — | KV cache reuse across requests |
| `--speculative-config` | DFlash, 10 tokens, `z-lab/Qwen3.6-27B-DFlash` | DFlash speculative decoding |
| `--reasoning-parser` | `qwen3` | Qwen3 CoT parser |
| `--chat-template` | `chat_template.jinja` | Custom chat template |
| `--default-chat-template-kwargs` | `{"enable_thinking":true,"preserve_thinking":true}` | Thinking block behaviour |
| `--tool-call-parser` | `qwen3_coder` | Qwen3 tool-call format |
| `--enable-auto-tool-choice` | — | Auto tool selection |
| `--generation-config` | `vllm` | Use vLLM's own generation config |
| `--language-model-only` | — | Disable multi-modal (text only) |
| `--skip-mm-profiling` | — | Skip multi-modal profiling |

---

## 🧩 Custom Chat Template

The file [`chat_template.jinja`](chat_template.jinja) is a comprehensive Jinja2 template (**qwen3.6-froggeric-v20**) designed for Qwen3.6 with:

- Original author: https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates
- **System / Developer / User / Assistant / Tool messages** — Full OpenAI-style role support
- **Thinking / Reasoning Blocks** — Wraps chain-of-thought in `<thinking>...</thinking>`; toggleable via `enable_thinking` and `preserve_thinking` template kwargs
- **Tool Calling** — Serialises function calls into the `<tool_call>\n<function=...>\n</tool_call>` format with JSON parameter rendering
- **Error Recovery** — Consecutive tool-call error detection with ⚠️ retry warnings to the model
- **Auto-disabling Thinking** — When tools are active, thinking is automatically disabled (`auto_disable_thinking_with_tools`)
- **Content Truncation** — `max_tool_arg_chars` / `max_tool_response_chars` for length-limited serialisation
- **Multi-Step Tool Chains** — Detects when tool calls require follow-up turns and preserves reasoning context

### Template Kwargs

| Kwargs | Type | Default (in template) | Default (via CLI) | Description |
|---|---|---|---|---|
| `enable_thinking` | bool | `true` | `true` | Enable `<thinking>` blocks in generation |
| `preserve_thinking` | bool | `false` | `true` | Preserve thinking blocks from history |
| `auto_disable_thinking_with_tools` | bool | `false` | `false` | Disable thinking when tools are defined |

---

## 📁 Project Structure

```
Qwen3.6-27B-NVFP4-vLLM-DFlash/
├── README.md             ← This file
├── start.sh              ← Launch script (vLLM container + DFlash)
├── stop.sh               ← Stop & cleanup script
├── chat_template.jinja   ← Custom Jinja chat template (v20)
├── benchmark.png         ← Benchmark results screenshot
└── .gitignore            ← Git ignore rules

# Runtime artifacts (auto-created, gitignored):
#   .vllm.log     — Live container log
#   .vllm.pid     — Container ID file
#   .cache/       — HuggingFace downloads + Triton cache
```

---

## 🐳 Docker Details

| Property | Value |
|---|---|
| **Image** | `vllm/vllm-openai:nightly-aarch64` |
| **Container Name** | `qwen3.6-27b-nvidia-nvfp4-dflash` |
| **Network** | `host` mode (`--network host`) |
| **IPC** | `host` mode (`--ipc host`) |
| **Privileged** | Yes (`--privileged`) |
| **GPUs** | All (`--gpus all`) |
| **Environment** | `VLLM_TARGET_DEVICE=cuda`, `VLLM_FLOAT32_MATMUL_PRECISION=high`, `HF_HOME`, `TRITON_CACHE_DIR`, `HF_TOKEN` |
| **Volumes** | HF cache, Triton cache, chat template, working directory |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `docker is not on PATH` | Install Docker or add it to your `PATH` |
| `vLLM container exited before becoming ready` | Check `.vllm.log` for errors; ensure GPU drivers are installed |
| `Error: cannot access '...'` | Set `HF_TOKEN` and re-run `start.sh` |
| OOM errors | Reduce `--gpu-memory-utilization` or `--max-num-seqs` |
| Model weights not downloading | Verify HF token and network access; check `.cache/huggingface/` |
| Container won't stop | `docker rm -f qwen3.6-27b-nvidia-nvfp4-dflash` then re-run `stop.sh` |
| Served model name mismatch | Use `"model": "qwen36-27b-nvidia-nvfp4-dflash"` in your API requests |
| Template errors | Check `chat_template.jinja` syntax; refer to Jinja2 docs |

---

## 📝 License

- **Model weights:** Refer to the [HuggingFace repo](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4) for licensing details
- **DFlash draft model:** Refer to the [z-lab/Qwen3.6-27B-DFlash](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) repo
- **This codebase:** MIT License (or adjust as needed)

---

## 📚 Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [Qwen3.6-27B-NVFP4 on HuggingFace](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
- [z-lab/Qwen3.6-27B-DFlash (Draft Model)](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [DFlash](https://github.com/nbasyl/DFlash)
