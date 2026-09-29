# Quantization recipe

## Requirements

- NVIDIA GPU with CUDA
- Python 3.10+
- See `requirements-quantize.txt` (install **torch** matching your CUDA separately)

## Steps

1. Create venv and install deps.
2. Run:

```bash
python scripts/quantize_nvfp4.py \
  --model-path microsoft/FastContext-1.0-4B-RL \
  --output-dir ./FastContext-1.0-4B-RL-NVFP4
```

3. Verify:
   - `hf_quant_config.json` has `"quant_method": "modelopt_fp4"`
   - `model.embed_tokens.weight` present (tied lm_head)
   - Tokenizer / `chat_template.jinja` copied from source

4. Upload to Hugging Face with model card metadata:

```yaml
base_model: microsoft/FastContext-1.0-4B-RL
base_model_relation: quantized
library_name: transformers
tags: [nvfp4, modelopt, qwen3]
```

## Options

- `--exclude-embed-tokens` if quality regresses (re-run PTQ)
- `--calib-samples`, `--batch-size`, `--seq-len` for calibration tuning

## Architecture notes

- Dense Qwen3-4B; no MoE / vision heads
- `tie_word_embeddings=true` — no separate `lm_head.weight` export pitfall when using ModelOpt defaults + tied handling