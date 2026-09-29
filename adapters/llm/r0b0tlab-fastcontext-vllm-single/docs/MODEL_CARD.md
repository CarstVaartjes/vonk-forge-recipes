# Model card (Hugging Face)

Canonical copy for [r0b0tlab/FastContext-1.0-4B-RL-NVFP4](https://huggingface.co/r0b0tlab/FastContext-1.0-4B-RL-NVFP4). Keep in sync when publishing README updates on the Hub.

## Overview

NVFP4 (W4A4) quantization of [microsoft/FastContext-1.0-4B-RL](https://huggingface.co/microsoft/FastContext-1.0-4B-RL) — repository-exploration subagent for coding agents.

## Credits

- **Base:** microsoft/FastContext-1.0-4B-RL (Microsoft, MIT). Qwen3-4B-Instruct base.
- **Quantization:** NVIDIA ModelOpt 0.44.0. Calibration: CNN/DailyMail (Apache 2.0).
- **Paper:** arXiv:2606.14066.
- **Quantization © 2026 r0b0tlab**; distributed under MIT.

## Quantization

| Property | Value |
|----------|-------|
| Method | NVFP4 W4A4, group_size=16 |
| Tool | ModelOpt `NVFP4_DEFAULT_CFG` |
| Calibration | 512 × 1024 tokens, batch 16 |
| Size | ~2.7 GB |
| `tie_word_embeddings` | true |
| `quant_method` | `modelopt_fp4` |

## Serving (vLLM)

```bash
vllm serve r0b0tlab/FastContext-1.0-4B-RL-NVFP4 \
  --quantization modelopt \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --port 30000
```

Requires NVFP4-capable GPU for native path.

## GB10 validation (reference)

| Metric | BF16 | NVFP4 |
|--------|------|-------|
| Decode | 22.8 tok/s | 66.3 tok/s |
| TTFT | 43 ms | 22 ms |
| Size | 7.6 GB | 2.7 GB |

## Limitations

- Post-hoc PTQ; minor quality regression possible.
- Hermes tool-call XML in content field; use FastContext CLI or compatible client.
- Benchmarks from single GB10 device; your results may vary.