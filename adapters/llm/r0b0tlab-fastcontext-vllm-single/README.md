# FastContext-1.0-4B-RL-NVFP4

**NVFP4 (W4A4) implementation** for [microsoft/FastContext-1.0-4B-RL](https://huggingface.co/microsoft/FastContext-1.0-4B-RL) — a repository-exploration subagent for coding agents (Read / Glob / Grep → compact `file:line` citations).

**Weights are published on Hugging Face.** This repo provides **quantization**, **vLLM serving**, and **integration** documentation only.

| Resource | Link |
|----------|------|
| **Weights (HF)** | [huggingface.co/r0b0tlab/FastContext-1.0-4B-RL-NVFP4](https://huggingface.co/r0b0tlab/FastContext-1.0-4B-RL-NVFP4) |
| **Base model** | [microsoft/FastContext-1.0-4B-RL](https://huggingface.co/microsoft/FastContext-1.0-4B-RL) |
| **Paper** | [arXiv:2606.14066](https://arxiv.org/abs/2606.14066) |

## What is FastContext?

FastContext is a **4B specialist** trained to explore codebases with read-only tools and return **citations**, so a larger “solver” model does not need the full repository in context. It is built on **Qwen3-4B-Instruct** and distributed by Microsoft under **MIT**.

This project adds a **native NVFP4 checkpoint** (NVIDIA ModelOpt, group_size=16) for **vLLM** on NVFP4-capable GPUs.

## Quick start (serve published weights)

```bash
git clone https://github.com/r0b0tlab/FastContext-1.0-4B-RL-NVFP4
cd FastContext-1.0-4B-RL-NVFP4
chmod +x configs/vllm-serve.sh
export FC_NVFP4_MODEL=r0b0tlab/FastContext-1.0-4B-RL-NVFP4
./configs/vllm-serve.sh
```

Requires **vLLM ≥ 0.23**, `--quantization modelopt`, and an **NVFP4-capable** NVIDIA GPU (vLLM may emulate on older hardware).

### Smoke test (OpenAI-compatible)

```bash
./examples/smoke_tool_call.sh
```

See `configs/vllm-serve.yaml` for the full flag set.

## Reproduce quantization (optional)

From BF16 source `microsoft/FastContext-1.0-4B-RL`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-quantize.txt
# Install torch for your CUDA version first if needed (see NVIDIA/PyTorch docs)

python scripts/quantize_nvfp4.py \
  --model-path microsoft/FastContext-1.0-4B-RL \
  --output-dir ./FastContext-1.0-4B-RL-NVFP4
```

Post-export, confirm `hf_quant_config.json` contains `"quant_method": "modelopt_fp4"` (the script sets this if missing).

Details: `docs/QUANTIZATION.md`

## Integration with FastContext CLI

Point the [FastContext](https://github.com/microsoft/fastcontext) client at your local vLLM endpoint:

```bash
export BASE_URL=http://127.0.0.1:30000/v1
export MODEL=FastContext-1.0-4B-RL-NVFP4
export API_KEY=local
```

Use `--tool-call-parser hermes` on vLLM (Qwen3-style tool XML).

## Performance (GB10 / SM121 validation)

Measured vs BF16 `microsoft/FastContext-1.0-4B-RL` on the same vLLM stack (FlashInfer, FP8 KV):

| Metric | BF16 | NVFP4 (this release) |
|--------|------|-------------------------|
| Decode throughput | 22.8 tok/s | **66.3 tok/s** |
| TTFT | 43 ms | **22 ms** |
| Checkpoint size | 7.6 GB | **2.7 GB** |
| Tool-call smoke | pass | pass |

Full model card: `docs/MODEL_CARD.md` (mirrors Hugging Face README).

## Repository layout

```
configs/          vLLM launch script + reference YAML
scripts/          NVFP4 quantization (ModelOpt)
docs/             Model card + quantization notes
examples/         Minimal serving smoke tests
AGENTS.md         Guidance for coding agents working in this repo
```

## License

- Quantization artifacts & this repo: **MIT** (see `LICENSE`)
- Base model: **MIT** (Microsoft)
- ModelOpt / calibration data: see `docs/MODEL_CARD.md`

## Citation

```bibtex
@misc{zhang2026fastcontext,
  title={FastContext: Training Efficient Repository Explorer for Coding Agents},
  author={Shaoqiu Zhang and Maoquan Wang and Yuling Shi and Yuhang Wang and Xiaodong Gu and Yongqiang Yao and Rao Fu and Shengyu Fu},
  year={2026},
  eprint={2606.14066},
  archivePrefix={arXiv},
  primaryClass={cs.SE}
}
```