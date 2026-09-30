# Credits — why each community repo is in this compile

This is **not** a new 27B we trained. It is a **single-Spark compile of the best open pieces** the community already shipped. Read the original repos. That is where the real work lives.

Keys (drowzeys) only assembled a measured one-box recipe: nested YaRN + drafter rope overlay, cybersecurity-unlock chat template, MIXED-style sampling, and the eval on one NVIDIA DGX Spark (GB10).

## Body and intelligence

| Piece | Repo | Why we used it (best in class for this seat) |
|---|---|---|
| **Base 27B** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) · [Qwen team](https://github.com/QwenLM) | Best open dense 27B of this generation: coding, agentic tool use, native 262,144 rope, vision, thinking control. Every other piece is built on this. |
| **Abliteration (weights)** | [AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-BF16) · [github.com/AEON-7](https://github.com/AEON-7) · [@SpaceTimeViking](https://x.com/SpaceTimeViking) | Best community uncensored cut of Qwen3.8-27B we measured. Abliterated for **coherence**, not a vanity KL of zero (~0.0991 nats/token). 0 hard “I won’t”. Vision + MTP left stock. That is the ablit. **Not a Keys train.** |
| **SSM repair in AEON’s pipeline** | FernflowerAI methodology (credited on the AEON card) | Best documented conv1d outlier repair before ablit so Gated-DeltaNet does not drift. We did not re-run it; AEON already did. |
| **Abliteration tools AEON used** | [abliterix](https://github.com/wuwangzhang1216/abliterix) (Wangzhang Wu) · [p-e-w/heretic](https://github.com/p-e-w/heretic) · Arditi et al. 2024 [2406.11717](https://arxiv.org/abs/2406.11717) | Best public refusal-direction stack. AEON’s trial 48/50 is on their card — read it. |
| **Uniform NVFP4 body (this Spark)** | [sakamakismile/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4](https://huggingface.co/sakamakismile/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4) (llm-compressor NVFP4 W4A4, MTP/vision/conv1d kept BF16) | Best **uniform** compressed-tensors NVFP4 of the AEON uncensored master that actually fits one GB10 with DFlash2 and a 1M KV pool. 2026-09-09 vs MIXED: tea **147.7 vs 117.4** agg @ c=16, **181.8 vs 152.7** @ c=32. STEM 39/40 both. [COMPARE.md](COMPARE.md). |
| **Official AEON Spark quant (not this seat)** | [AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED](https://huggingface.co/AEON-7/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-NVFP4-MIXED) | Best **official** Spark/5090 deploy knife from AEON (ModelOpt MIXED lattice). We still point you there. We used uniform NVFP4 on `.4` because it was faster on this bake-off, not because MIXED is bad. |

## Speculative decode

| Piece | Repo | Why we used it |
|---|---|---|
| **DFlash 2 drafter** | [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) (canonical) · mirror [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) · [DFlash 2 blog](https://inco.ai/blog/dflash2/) · [z-lab/dflash](https://github.com/z-lab/dflash) · Chen, Liang, Liu [arXiv:2602.06036](https://arxiv.org/abs/2602.06036) | Best **lossless** block-diffusion drafter for Qwen3.8-27B. Official AEON single-Spark recipe is DFlash2 **n=7**. Code decode ~42 tok/s vs prose ~20 on this box — that gap is drafter acceptance, not a second model. Seeded at Z Lab, upgraded at Inco AI. |

## Runtime (the prebuilt image)

| Piece | Repo | Why we used it |
|---|---|---|
| **Prebuilt GB10 vLLM** | `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` digest `sha256:dd2018473ed88bc23b01cfc3179b5b6896a7f0f152ae06d8d274db62d330ef48` · tag `2026-08-24-v0.27.1-omni` · [AEON-7 GitHub](https://github.com/AEON-7) | Best **ready-to-run** vLLM for DGX Spark SM121: TRITON_ATTN, DFlash method, compressed-tensors NVFP4, qwen3 + qwen3_coder parsers, nested YaRN `hf-overrides`. We did not rebuild vLLM. We **retag and ship that image**. |
| **vLLM** | [vllm-project/vllm](https://github.com/vllm-project/vllm) Apache-2.0 | Best production OpenAI-compatible engine this compile can sit on. AEON’s image is `v0.27.1+aeon.sm121a.dspark`. |
| **YaRN** | Peng et al. 2023 [arXiv:2309.00071](https://arxiv.org/abs/2309.00071) | Best standard RoPE stretch. Native 262,144 × factor **4.0** = **1,048,576**. Official AEON 1-Spark 1M recipe. Nested under `text_config.rope_parameters` because Qwen3.8 is a VLM. Drafter `config.json` must be YaRN-extended too — vLLM will not stretch dflash rope past 262k by itself. |
| **NVIDIA DGX Spark / GB10** | NVIDIA | The box this was measured on (128 GB unified). GMU **never above 0.85**. |

## What Keys added (small)

- Nested YaRN JSON + DFlash2 rope overlay so 1M actually loads and retrieves (needle **HIT at 999,714 tokens**).
- Cybersecurity-unlock default system template (cyber / defense / dual-use included). That is how this seat went **cyber 8/8** and **refusal32 32/32** after the AEON uncensored body alone was 7/8 and 30/32 on our first B5 gate.
- MIXED-style sampling (`repetition_penalty=1.05`).
- Measured RESULTS on spark-13b3. The Monday item is a **harness miss the model corrected** (Monday is the right day; the key said Sunday).

If you only remember one thing: **go read AEON-7, Inco/Z Lab, Qwen, and sakamakismile.** This repo is the wiring diagram.
