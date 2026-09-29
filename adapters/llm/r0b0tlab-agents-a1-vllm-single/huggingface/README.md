---
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
base_model: InternScience/Agents-A1
base_model_relation: quantized
quantized_by: r0b0tlab
tags:
  - nvfp4
  - modelopt
  - nvidia-modelopt
  - quantized
  - qwen3_5_moe
  - text-generation
  - image-text-to-text
  - base_model:InternScience/Agents-A1
  - base_model:quantized:InternScience/Agents-A1
---

# Agents-A1 NVFP4

This repository contains an NVIDIA ModelOpt NVFP4 quantization of [`InternScience/Agents-A1`](https://huggingface.co/InternScience/Agents-A1), a 35B Qwen3.5 MoE agentic model.

## Credits and Attribution

This NVFP4 checkpoint is derived from [`InternScience/Agents-A1`](https://huggingface.co/InternScience/Agents-A1).

- **Base model:** InternScience, for the Agents-A1 model, training recipe, technical report, and original BF16 Hugging Face release.
- **Quantization tooling:** NVIDIA, for **NVIDIA TensorRT Model Optimizer / NVIDIA ModelOpt**, used to produce the NVFP4 ModelOpt checkpoint.
- **Model architecture and runtime ecosystem:** Hugging Face Transformers, Safetensors, Accelerate, and the Hugging Face Hub.
- **Calibration data:** CNN/DailyMail via Hugging Face Datasets, used for text-path post-training calibration.
- **Inference ecosystem:** vLLM/SGLang compatibility is inherited from the Qwen3.5 MoE / ModelOpt NVFP4 ecosystem, subject to runtime support and validation.

## Quantization Summary

| Field | Value |
|---|---|
| Base model | `InternScience/Agents-A1` |
| Quantization tool | NVIDIA ModelOpt `0.44.0` |
| Quantization format | NVFP4 / ModelOpt FP4 |
| ModelOpt config | `mtq.NVFP4_MLP_ONLY_CFG` |
| Calibration data | `abisee/cnn_dailymail`, text-only calibration |
| Calibration sequence length | 1024 |
| Architecture | `Qwen3_5MoeForConditionalGeneration` |
| License | Apache-2.0, following the base model |

## Quantization Policy

Agents-A1 is a hybrid Qwen3.5 MoE model with 30 `linear_attention` layers, 10 full-attention layers, 256 experts per layer, and a vision tower. This checkpoint uses an MLP/MoE-only NVFP4 policy for the first release.

The following module families were explicitly excluded from NVFP4 quantization and preserved in BF16:

```json
[
  "*visual*",
  "*vision*",
  "*patch_embed*",
  "*pos_embed*",
  "*merger*",
  "*linear_attn*",
  "*linear_attention*",
  "*self_attn*",
  "*attn*",
  "*embed_tokens*",
  "*lm_head*",
  "*mtp*"
]
```

Rationale:

- The MoE/MLP expert layers are the largest parameter family and are the correct target for NVFP4 compression.
- The GDN/`linear_attn` path is not standard dense transformer attention and is excluded for compatibility.
- Vision modules are preserved to avoid multimodal degradation from text-only calibration.
- Embeddings, `lm_head`, and MTP-sensitive modules are preserved in BF16.

## Files

- `hf_quant_config.json` — ModelOpt quantization metadata used by compatible inference engines.
- `modelopt_exclusions.json` — exact exclusion list used during quantization.
- `config.json`, tokenizer, and processor files are copied from the base model and patched only as required for export consistency.

## Validation Status

This release is a quantized checkpoint, not a new fine-tune. It does **not** claim quality improvement over BF16.

Runtime smoke testing on NVIDIA GB10 / SM121 completed with the companion container recipe at [`r0b0tlab/agents-a1-nvfp4-sm121-vllm`](https://github.com/r0b0tlab/agents-a1-nvfp4-sm121-vllm).

Validated evidence includes:

1. Container audit on NVIDIA GB10 with CUDA capability `[12, 1]`.
2. vLLM extension imports: `vllm._C`, `vllm._C_stable_libtorch`, `vllm._moe_C`.
3. Native FP4 support checks: `cutlass_scaled_mm_supports_fp4(121)` and `(120)` return `true`.
4. Runtime log selection of `FlashInferCutlassNvFp4LinearKernel` and `FLASHINFER_CUTLASS` for NVFP4/MoE.
5. OpenAI-compatible `/v1/models` and `/v1/chat/completions` probes against the running container.
6. Lightweight live-container benchmark evidence in the companion repo: GSM8K 50-question lm-eval run at 98.00% exact match, direct HumanEval 50-question run at 48/50 (96.00%), c1/c2/c4/c8 concurrency sweep with 100% request success, and GPU telemetry including power draw.

### Benchmark snapshot

The benchmark run `agents-a1-nvfp4-gsm8k50-humaneval50-20260701T194211Z` used the live OpenAI-compatible endpoint at `http://127.0.0.1:18080/v1` with `chat_template_kwargs.enable_thinking=false` for scored requests.

| Suite | Harness | Samples | Result | Notes |
|---|---:|---:|---:|---|
| GSM8K | lm-eval `gsm8k` | 50 | strict 98.00%, flexible 98.00% | `num_concurrent=2` |
| HumanEval | direct OpenAI-compatible evaluator | 50 | 48/50 (96.00%) | code extracted/evaluated locally |
| HumanEval | stock lm-eval `humaneval` | 50 | 0.00% | preserved as harness-interference evidence; stock stop rules truncate chat-model output |

Combined telemetry across GSM8K, HumanEval, and direct HumanEval averaged 27.88 W GPU power draw, 70.75% GPU utilization, and 58.83°C, with maxima of 36.00 W, 96.00%, and 65.00°C over 166 telemetry samples. The c8 concurrency sweep completed 24/24 requests successfully. See the companion repo's `benchmarks/agents-a1-nvfp4-gsm8k50-humaneval50-20260701T194211Z/` directory for raw logs, samples, summaries, telemetry CSVs, and `MANIFEST.sha256`.

### SM121 container quick start

```bash
docker run --rm --gpus all --ipc=host \
  --name agents-a1-nvfp4-vllm \
  -p 18080:8000 \
  -e MODEL_ID=r0b0tlab/Agents-A1-NVFP4 \
  ghcr.io/r0b0tlab/agents-a1-nvfp4-sm121-vllm:latest
```

For fully pinned local reproduction, clone/download this model and mount it read-only:

```bash
docker run --rm --gpus all --ipc=host \
  --name agents-a1-nvfp4-vllm \
  -p 18080:8000 \
  -v /path/to/Agents-A1-NVFP4:/models/Agents-A1-NVFP4:ro \
  ghcr.io/r0b0tlab/agents-a1-nvfp4-sm121-vllm:latest
```

## Limitations

- Calibration is text-only; vision components are preserved in BF16 rather than calibrated.
- This card does not claim benchmark parity until BF16-vs-NVFP4 evaluations are published.
- Runtime support depends on the inference engine's ModelOpt/NVFP4 implementation.

## Citation

```bibtex
@misc{internscience_agents_a1_2026,
  title = {Agents-A1: Scaling the Horizon, Not the Parameters: Reaching Trillion-Parameter Performance with a 35B Agent},
  author = {InternScience},
  year = {2026},
  url = {https://huggingface.co/InternScience/Agents-A1}
}
```

## License

This quantized checkpoint follows the base model license, Apache-2.0. Users must also comply with the licenses and terms for the base model, calibration data, NVIDIA ModelOpt, Hugging Face libraries, and any inference runtime used.
