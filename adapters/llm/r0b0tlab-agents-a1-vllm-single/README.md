# Agents-A1 NVFP4 SM121 vLLM Container

Reproducible SM121 / NVIDIA GB10 container recipe for serving the NVIDIA ModelOpt NVFP4 quantization of [`InternScience/Agents-A1`](https://huggingface.co/InternScience/Agents-A1) with vLLM-compatible OpenAI endpoints.

This repository is intentionally evidence-first: it documents the model artifact, the native NVFP4 runtime gates, the exact container entrypoint, and the commands used to prove the service is running.

## What this is

- **Base model:** `InternScience/Agents-A1`
- **Quantized checkpoint:** `r0b0tlab/Agents-A1-NVFP4`
- **Runtime target:** NVIDIA GB10 / SM121 (`compute capability 12.1`)
- **Quantization format:** NVIDIA ModelOpt NVFP4 / `modelopt`, group size 16
- **Serving API:** vLLM OpenAI-compatible server
- **Default mode:** text-only serving via `--language-model-only`

## What this is not

- It is not a fine-tune or behavior improvement over the base model.
- It is not a generic CUDA image for older GPUs.
- It is not a fallback/Marlin/emulation container. The publish gate requires native SM121 NVFP4 runtime evidence.

## Quantization policy

Agents-A1 is a Qwen3.5 MoE agentic model with hybrid linear-attention/full-attention layers and a vision tower. The first NVFP4 release uses an **MLP/MoE-only** ModelOpt policy:

- Quantized: MoE/MLP expert weights where ModelOpt marks them NVFP4.
- Preserved in BF16: `visual*`, `linear_attn*`, `self_attn*`, embeddings, `lm_head`, shared expert gates, and MTP-sensitive modules.

See the model card in `huggingface/README.md` for the exact exclusion list and Hugging Face metadata.

## Repository layout

```text
.
├── AGENTS.md                         # hard gates for future agents
├── README.md                         # this file
├── docker/Dockerfile                 # derivative SM121 runtime image
├── scripts/
│   ├── audit_runtime.py              # strict import / FP4 / artifact scan
│   ├── healthcheck.sh                # /v1/models healthcheck
│   └── start_agents_a1.sh            # container entrypoint
├── evidence/
│   ├── container-audit/              # build and runtime audit logs
│   └── runtime/                      # serving logs and API probes
├── benchmarks/                       # GSM8K, HumanEval, telemetry, concurrency evidence
└── huggingface/README.md             # model card used for upload
```

## Published artifacts

| Artifact | URL / digest |
|---|---|
| GitHub repo | https://github.com/r0b0tlab/agents-a1-nvfp4-sm121-vllm |
| Hugging Face model | https://huggingface.co/r0b0tlab/Agents-A1-NVFP4 |
| Container image | `ghcr.io/r0b0tlab/agents-a1-nvfp4-sm121-vllm:latest` |
| Image digest | `sha256:89a686b38a3831e540ecab17043f44df7bdc3cb49ee04f59e5b0e1b86c474edc` |
| Benchmark evidence | `benchmarks/agents-a1-nvfp4-gsm8k50-humaneval50-20260701T194211Z/` |

## Build

```bash
docker build \
  -t ghcr.io/r0b0tlab/agents-a1-nvfp4-sm121-vllm:latest \
  -f docker/Dockerfile .
```

The Dockerfile derives from the current r0b0tlab SM121 NVFP4 vLLM runtime, removes packaged fallback FlashAttention/Marlin artifacts that are not part of the publishable Agents-A1 text-only path, and patches the import guard so model modules that reference `fa_utils` still load while FlashInfer remains the active runtime backend.

## Audit the image

```bash
docker run --rm --gpus all \
  ghcr.io/r0b0tlab/agents-a1-nvfp4-sm121-vllm:latest audit
```

Expected gates:

- `cuda_capability` is `[12, 1]`
- `vllm._C`, `vllm._C_stable_libtorch`, `vllm._moe_C` import
- `cutlass_scaled_mm_supports_fp4(121)` is `true`
- no `_vllm_fa2_C`, `_vllm_fa3_C`, or `*marlin*.so` artifacts are packaged

## Serve local checkpoint

```bash
docker run --rm --gpus all --ipc=host \
  --name agents-a1-nvfp4-vllm \
  -p 18080:8000 \
  -v /home/r0b0tdgx/work/agents-a1-nvfp4/Agents-A1-NVFP4:/models/Agents-A1-NVFP4:ro \
  ghcr.io/r0b0tlab/agents-a1-nvfp4-sm121-vllm:latest
```

Default entrypoint args:

```text
--model /models/Agents-A1-NVFP4
--served-model-name agents-a1-nvfp4
--quantization modelopt
--kv-cache-dtype fp8
--attention-backend flashinfer
--moe-backend flashinfer_cutlass
--gpu-memory-utilization 0.70
--max-model-len 4096
--max-num-seqs 2
--max-num-batched-tokens 4096
--trust-remote-code
--language-model-only
--enforce-eager
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
--reasoning-parser qwen3
```

## Probe

```bash
curl -s http://127.0.0.1:18080/v1/models | python3 -m json.tool

curl -s http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"agents-a1-nvfp4",
    "messages":[{"role":"user","content":"Say READY and nothing else."}],
    "max_tokens":8
  }' | python3 -m json.tool
```

## Benchmark evidence

The live `agents-a1-nvfp4-vllm` container at `http://127.0.0.1:18080/v1` was evaluated with a lightweight benchmark suite rather than the full Hermes bench. The run preserved raw lm-eval outputs, direct HumanEval traces, telemetry, concurrency data, a summary, and `MANIFEST.sha256` under `benchmarks/agents-a1-nvfp4-gsm8k50-humaneval50-20260701T194211Z/`.

| Suite | Harness | Samples | Result | Notes |
|---|---:|---:|---:|---|
| GSM8K | lm-eval `gsm8k` | 50 | strict 98.00%, flexible 98.00% | `num_concurrent=2`, thinking disabled |
| HumanEval | direct OpenAI-compatible evaluator | 50 | 48/50 (96.00%) | code extracted/evaluated locally, `num_concurrent=2` |
| HumanEval | stock lm-eval `humaneval` | 50 | 0.00% | preserved as harness-interference evidence; stock stop rules truncate chat-model function output |

Concurrency sweep results use a short chat-completion prompt, three reps per level, and the mean of the last two reps:

| Concurrency | Success | Mean wall sec | Mean aggregate output tok/s |
|---:|---:|---:|---:|
| c1 | 3/3 | 0.171 | 17.52 |
| c2 | 6/6 | 0.178 | 33.72 |
| c4 | 12/12 | 0.387 | 31.21 |
| c8 | 24/24 | 0.751 | 32.14 |

Combined telemetry across GSM8K, HumanEval, and direct HumanEval:

| Metric | Avg | Max | Samples |
|---|---:|---:|---:|
| GPU power draw | 27.88 W | 36.00 W | 166 |
| GPU utilization | 70.75% | 96.00% | 166 |
| GPU temperature | 58.83°C | 65.00°C | 166 |

The benchmark helper scripts were also ad-hoc verified after the run: syntax checks passed, `direct_humaneval50.py` imported without executing `main()`, `extract_code()` handled fenced and completion-style outputs, `evaluate()` accepted a known-good candidate and rejected a known-bad candidate, benchmark artifact counts matched the expected 50+50 samples, and `/v1/models` still returned `agents-a1-nvfp4` with `max_model_len=4096`. This is focused ad-hoc verification, not a canonical repo-wide green suite claim.

## Environment overrides

| Variable | Default | Meaning |
|---|---:|---|
| `MODEL_ID` | `/models/Agents-A1-NVFP4` | HF ID or mounted checkpoint path |
| `SERVED_MODEL_NAME` | `agents-a1-nvfp4` | OpenAI model name |
| `PORT` | `8000` | Container HTTP port |
| `MAX_MODEL_LEN` | `4096` | Context length for smoke serving |
| `MAX_NUM_SEQS` | `2` | Conservative GB10 smoke default |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Leaves room for one MTP draft token per sequence |
| `GPU_MEMORY_UTILIZATION` | `0.70` | VRAM cap |
| `KV_CACHE_DTYPE` | `fp8` | KV cache dtype |
| `SPECULATIVE_CONFIG` | `{"method":"mtp","num_speculative_tokens":1}` | Enables the built-in Qwen3.5 MoE MTP drafter; set empty to disable |
| `EXTRA_ARGS` | empty | Additional vLLM args |

## MTP status

MTP/speculative decoding is enabled by default in the container via:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":1}'
```

Runtime evidence for the live endpoint is preserved under `evidence/mtp/`. The active-log markers to check are:

- `Resolved architecture: Qwen3_5MoeMTP`
- `speculative_config=SpeculativeConfig(method='mtp', model='/models/Agents-A1-NVFP4', num_spec_tokens=1)`
- `Loading drafter model...`
- `Detected MTP model. Sharing target model embedding weights with the draft model.`
- Prometheus `/metrics` exposes `vllm:spec_decode_num_drafts_total` and `vllm:spec_decode_num_draft_tokens_total`.

This is a live runtime gate, not just a config-field claim. Acceptance-rate metrics should be inspected before claiming an MTP throughput uplift.

## Hugging Face upload

The model card is staged at `huggingface/README.md`. To make the quantization appear under the base model's Quantizations section, the card must include both:

```yaml
base_model: InternScience/Agents-A1
tags:
  - base_model:InternScience/Agents-A1
  - base_model:quantized:InternScience/Agents-A1
```

Upload pattern:

```bash
hf repos create r0b0tlab/Agents-A1-NVFP4 --type model --public --exist-ok --token "$HF_TOKEN"
hf upload r0b0tlab/Agents-A1-NVFP4 /home/r0b0tdgx/work/agents-a1-nvfp4/Agents-A1-NVFP4 . --repo-type model --token "$HF_TOKEN"
```

## Credits

- InternScience for `Agents-A1`.
- NVIDIA for ModelOpt and NVFP4 tooling.
- vLLM, FlashInfer, CUTLASS, PyTorch, Hugging Face Transformers, and Safetensors for the runtime ecosystem.
- r0b0tlab for the GB10/SM121 validation and packaging.
