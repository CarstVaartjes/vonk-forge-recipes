# AGENTS.md

Guidance for AI coding agents working in **r0b0tlab/FastContext-1.0-4B-RL-NVFP4**.

## Scope of this repository

**In scope**

- NVFP4 weights consumption via **vLLM** (`--quantization modelopt`)
- Reproducing quantization with **NVIDIA ModelOpt** (`scripts/quantize_nvfp4.py`)
- Serving config (`configs/vllm-serve.sh`, `configs/vllm-serve.yaml`)
- Documentation aligned with [Hugging Face model card](https://huggingface.co/r0b0tlab/FastContext-1.0-4B-RL-NVFP4)

**Out of scope (do not add here)**

- Demo videos, HyperFrames, Pillow render pipelines, marketing assets
- Synthetic `sample-repo` fixtures for A/B token comparisons
- Hermes concurrent-agents or unrelated r0b0tlab projects (link externally only)

## Key facts

| Item | Value |
|------|--------|
| HF model ID | `r0b0tlab/FastContext-1.0-4B-RL-NVFP4` |
| Base | `microsoft/FastContext-1.0-4B-RL` |
| Architecture | `Qwen3ForCausalLM`, `tie_word_embeddings=true` |
| Quant | NVFP4 W4A4, `group_size=16`, `quant_method: modelopt_fp4` |
| Tool parser | `hermes` (vLLM `--enable-auto-tool-choice`) |
| Typical port | `30000` |

## Commands agents should know

```bash
# Serve published weights
./configs/vllm-serve.sh

# Quantize from BF16 (GPU + ModelOpt venv required)
python scripts/quantize_nvfp4.py --model-path microsoft/FastContext-1.0-4B-RL --output-dir ./out

# Smoke test after serve
./examples/smoke_tool_call.sh
```

Environment variables for serve script: `FC_NVFP4_MODEL`, `FC_NVFP4_PORT`.

## Editing rules

1. **No machine-specific paths** in committed files (`/home/...`, `~/models/...`). Use env vars or `./relative` paths.
2. **Keep HF and GitHub in sync** — if you change quantization facts, update `docs/MODEL_CARD.md` and remind the user to update the HF README.
3. **`hf_quant_config.json`** must include `"quant_method": "modelopt_fp4"` for vLLM/SGLang auto-detection.
4. Do not commit **weights**, **venv**, or **large generated artifacts**.
5. Prefer **implementation clarity** over benchmark marketing; cite numbers only from `docs/MODEL_CARD.md`.

## Common pitfalls

- **lm_head with tied embeddings:** FastContext uses `tie_word_embeddings=true`. There is often **no** separate `lm_head.weight` in the checkpoint; do not “fix” by copying lm_head incorrectly.
- **KV cache:** Use `fp8` KV on current public stacks; NVFP4 KV may not be available everywhere.
- **Parser:** FastContext tool format expects **hermes** parser on vLLM, not a generic OpenAI tools schema only.

## Related projects

- Microsoft FastContext runtime: https://github.com/microsoft/fastcontext
- Base weights: https://huggingface.co/microsoft/FastContext-1.0-4B-RL

## Verification checklist (before PR)

- [ ] `python3 -m py_compile scripts/quantize_nvfp4.py`
- [ ] README + AGENTS.md + docs consistent with HF model card
- [ ] No `demos/`, `videos/`, or render scripts reintroduced
- [ ] `configs/vllm-serve.sh` runs with `FC_NVFP4_MODEL=r0b0tlab/FastContext-1.0-4B-RL-NVFP4` (document if you cannot run GPU tests)