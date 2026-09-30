# AGENTS.md

## Repository purpose

This repository contains model-specific, evidence-backed reproduction and benchmark material for the audited Hy3-295B W4A4 NVFP4 v3 artifact on two NVIDIA GB10 systems.

## Source of truth

- Machine-readable files under `benchmarks/` and `runtime/` are authoritative.
- `README.md` and `docs/index.html` must be regenerated from those files; never edit metrics manually.
- Keep quality, base concurrency, MTP-1, and base-AR context evidence separate.
- Publish curated summaries only; never include raw logs, telemetry streams, or per-item/request dumps.
- NVFP4 describes model weights; the validated KV cache is FP8.

## Release gates

Before publication, run `python3 scripts/verify_publication.py .`, verify `MANIFEST.sha256`, render the report at phone and desktop widths, and perform an anonymous clean-clone verification. Any changed file requires regenerating the manifest.

## Prohibited content

Never commit weights, caches, credentials, private hosts/IPs, operator home paths, raw private logs, unrelated model branding, or claims unsupported by preserved JSON. Do not introduce Marlin, emulation, dequantized routed experts, or BF16 routed-expert fallback as benchmarked paths.

## Attribution

Preserve upstream credit for Tencent Hy3, NVIDIA ModelOpt/CUDA/GB10, vLLM, FlashInfer, Hugging Face Datasets, lm-eval, llama-benchy, and benchmark datasets.
