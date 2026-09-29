# AGENTS.md — DiffusionGemma NVFP4 on SM121 vLLM

Instructions for AI agents serving, benchmarking, or publishing `nvidia/diffusiongemma-26B-A4B-it-NVFP4` on NVIDIA GB10 / SM121.

## One rule

**Never publish fallback, MARLIN, or emulation results as SM121-native NVFP4.**

Valid claims require evidence in `evidence/` showing:
- compute capability `(12, 1)`
- vLLM SM120-family detection is true
- dLLM Model Runner v2 is enabled
- NVFP4 quant kernels use native CUTLASS / FlashInfer-CUTLASS paths
- zero log hits for MARLIN/emulation fallback, excluding harmless "potential" strings

## Canonical launch

Use `scripts/serve.sh`. Start from NVIDIA's day-0 recipe and add only log-backed overrides.

## Benchmark gate

Run, at minimum:
- `scripts/bench_concurrency.sh` for c1, c2, c4, c5, c8, c16
- `scripts/bench_depth_sweep.sh`
- `scripts/bench_dllm_metrics.py`
- `scripts/capture_telemetry.sh` sidecar during each long run

Every benchmark must save raw JSON/JSONL under `benchmarks/` and telemetry under `evidence/telemetry/`.

## Report gate

Before sending or linking the report, visually inspect `publication/html/index.html` with browser automation or screenshot and verify charts render.
