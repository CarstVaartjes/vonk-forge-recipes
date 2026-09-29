# Models & memory

`llm-eyes` defaults to **Qwen3.5-0.8B** — small enough to run beside your real model, good
enough to say what's in frame. Swap with the `MODEL=` env var (see the fallback table).

## Memory footprint — Mac vs PC

| | Mac (MLX, unified memory) | PC / Linux (llama.cpp + NVIDIA) |
|---|---|---|
| **Model** | `Qwen3.5-0.8B-MLX-8bit` | `Qwen3.5-0.8B-GGUF` (UD-Q4_K_XL) |
| **Weights on disk** | ~0.98 GB | ~0.6 GB + ~0.7 GB mmproj (vision) |
| **Live footprint** | **~2–3 GB unified RAM** | **~2 GB VRAM** (KV + vision encoder + overhead) |
| **Minimum machine** | any **8 GB** Apple Silicon Mac | any GPU with **≥4 GB VRAM** (or CPU-only: ~2–3 GB system RAM) |
| **Speed** | fast (Metal) | fast (`-ngl 99` puts it all on the GPU) |

**Bottom line:** the 0.8B eyes model is featherweight on both. On Mac it sips ~2–3 GB of
unified memory; on a PC GPU it needs only ~2 GB of VRAM, so it fits on essentially any
modern card and leaves your real model the rest. CPU-only works too — just slower frames.

## Fallback / upgrade models

Set `MODEL=` before running a backend (or edit `REPO`/`QUANT` in the script):

| Model | Size | VRAM / RAM | Get it |
|---|---|---|---|
| **Qwen3.5-0.8B** (default) | 0.8B | ~2 GB | this repo's default |
| **SmolVLM-500M** | 0.5B | ~1–2 GB | lightest / fastest frames — `ggml-org/SmolVLM-500M-Instruct-GGUF` |
| **Moondream2** | ~1.9B | ~4 GB | edge-tuned; Ollama one-liner: `ollama pull moondream` |
| **Qwen2.5-VL-3B** | 3B | ~5–6 GB | best caption quality of the tinies; `ollama pull qwen2.5vl:3b` |

> **Ollama note:** Qwen3.5 vision does **not** run in Ollama yet (it can't load the standalone
> mmproj). Use llama.cpp for Qwen3.5, or pick Moondream / Qwen2.5-VL from the table above if
> you specifically want the Ollama one-liner.

## Tuning
- **Bigger quant for quality** (if you have the memory): `QUANT=UD-Q8_K_XL`.
- **Smaller for tight VRAM:** `QUANT=UD-IQ2_XXS` (~338 MB).
- **CPU-only:** `NGL=0`.
- **Caption prompt:** `LLM_EYES_PROMPT="List objects and any text you see."`
