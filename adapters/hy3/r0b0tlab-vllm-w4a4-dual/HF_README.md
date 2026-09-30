---
license: other
library_name: transformers
pipeline_tag: text-generation
tags:
  - hy3
  - hy_v3
  - moe
  - nvfp4
  - w4a4
  - modelopt
  - base_model:tencent/Hy3
  - base_model:quantized:tencent/Hy3
base_model: tencent/Hy3
---

# Hy3-295B-NVFP4

Audited calibrated/MSE W4A4 NVFP4 v3 export of [Tencent Hy3](https://huggingface.co/tencent/Hy3), validated with native vLLM on two NVIDIA GB10 systems.

## Credits and Attribution

- **Base model:** Tencent Hy3 authors and maintainers.
- **Quantization and hardware:** NVIDIA ModelOpt, CUDA, and GB10.
- **Serving:** vLLM and FlashInfer.
- **Evaluation tooling:** Hugging Face Datasets, lm-eval, llama-benchy, and Docker.
- **Benchmarks:** GSM8K, AI2 ARC-Challenge, PIQA, WinoGrande, TruthfulQA, MMLU, IFEval, and OpenAI HumanEval.

Please cite and follow the licenses/terms of every upstream model, library, and benchmark dataset used.

## Artifact identity and quantization

- Artifact: `Hy3-NVFP4-w4a4-v3`
- Model index SHA-256: `738e3ad2c8d16372e3d681d5793c78676fd256d7a4b43d9ca560f268d26adb26`
- Architecture: `HYV3ForCausalLM`
- Quantization loader: `modelopt_mixed`
- Routed-expert backend: `FLASHINFER_CUTLASS`
- Validated KV cache: FP8
- Tensor parallelism: TP=2 across two GB10 systems
- Runtime image: `ghcr.io/r0b0tlab/vllm-v0250-cu130-sm121:v0.25.0-cu130-sm121-arm64-702f4814-r2`
- Reproducibility evidence: [https://github.com/r0b0tlab/hy3-295b-nvfp4-gb10-benchmark](https://github.com/r0b0tlab/hy3-295b-nvfp4-gb10-benchmark)

The artifact uses calibrated/MSE W4A4 NVFP4 for supported routed-expert projections while preserving explicitly excluded/non-quantized tensors according to the published `hf_quant_config.json`. No performance result in this card uses Marlin, emulation, dequantized routed experts, or BF16 routed-expert fallback.

## Quality — base O0/eager, non-MTP

| Task | N | Metric |
|---|---:|---:|
| gsm8k | 1319 | 95.91% |
| arc_challenge | 1172 | 96.16% |
| piqa | 1838 | 94.61% |
| winogrande | 1267 | 83.90% |
| truthfulqa_mc1 | 817 | 82.25% |
| mmlu_abstract_algebra | 100 | 74.00% |
| mmlu_business_ethics | 100 | 82.00% |
| mmlu_clinical_knowledge | 265 | 92.83% |
| mmlu_college_biology | 144 | 98.61% |
| mmlu_computer_security | 100 | 88.00% |
| mmlu_conceptual_physics | 235 | 94.89% |
| mmlu_high_school_world_history | 237 | 93.25% |
| mmlu_international_law | 121 | 92.56% |
| ifeval | 541 | 91.13% |
| humaneval | 164 | 84.15% |

HumanEval is sandbox-executed pass@1. IFEval reports official prompt- and instruction-level strict/loose metrics in the linked structured evidence.

## Performance profiles

Highest safe staged base-profile concurrency: **c8**.

| Concurrency | Aggregate output tok/s | Per-request tok/s | Mean wall s | Mean output tokens |
|---|---:|---:|---:|---:|
| c1 | 8.73 | 8.73 | 14.66 | 128 |
| c2 | 16.67 | 8.37 | 15.36 | 256 |
| c4 | 32.03 | 8.02 | 15.98 | 512 |
| c8 | 50.94 | 6.37 | 20.10 | 1024 |

### MTP-1 · separate profile

MTP-1 passed at utilization 0.735; measured acceptance 52.97%; c1 mean 11.04 output tok/s under profile `mtp1_o0_eager_sync`. Forced 128/512/1024-token decoding passed 3/3 at each length, representative stability passed 100/100, and llama-benchy 0.4.0 completed.

The c1 label is `mtp1_o0_eager_sync`; it must not be presented as the base profile. MTP-1 drafted 23610 tokens and accepted 12506 (52.97%).

### Practical 48K base-AR context · separate profile

The base-AR O0/eager/synchronous context profile passed a 49,152-token configured window with 49,968 measured KV-token capacity and 3,904 MiB FP8 KV. Exact near-window retrieval passed for needles near the beginning, middle, and end at up to 48,896 prompt tokens. This is a practical tested boundary, not an architectural maximum claim, and it is not an MTP result.

## Reproduction and verification

Clone [https://github.com/r0b0tlab/hy3-295b-nvfp4-gb10-benchmark](https://github.com/r0b0tlab/hy3-295b-nvfp4-gb10-benchmark) and follow its `README.md`, `AGENTS.md`, `runtime/runtime.json`, sanitized scripts, structured benchmark files, and `MANIFEST.sha256`. Human-readable metrics are generated from the machine-readable evidence; they are not edited manually.

## Notes and limitations

- Quality uses only the base O0/eager/max-seqs-1 non-MTP profile.
- Base concurrency, MTP-1, and base-AR context are separately launched profiles and must not be merged into one claim.
- The 49,152-token context result is a practical base-AR boundary; no larger context is claimed.
- NVFP4 describes model weights; the validated KV cache is FP8.
- Results are specific to the documented dual-GB10 hardware, runtime image, prompts, datasets, and decoding contracts.
- The benchmark suite does not establish universal model quality, safety, or optimality.
- MTP performance is claimed only because preserved speculative counters prove positive drafted and accepted tokens.
- Published evidence contains curated summaries, not raw private logs, telemetry streams, or per-item responses.

## License

The base-model license and terms from Tencent apply to the underlying model. Dataset-specific licenses/terms apply to benchmark content. Reproduction scripts and original report assets in the linked GitHub evidence repository are released under that repository's license. Review all upstream terms before use or redistribution.

## Citation

```bibtex
@misc{hy3,
  title={Hy3},
  author={Tencent Hy3 Team},
  year={2026},
  howpublished={Hugging Face model release}
}
@software{nvidia_modelopt,
  title={NVIDIA Model Optimizer},
  author={NVIDIA},
  url={https://github.com/NVIDIA/Model-Optimizer}
}
@software{vllm,
  title={vLLM: A High-Throughput and Memory-Efficient Inference and Serving Engine for LLMs},
  url={https://github.com/vllm-project/vllm}
}
@software{flashinfer,
  title={FlashInfer: Kernel Library for LLM Serving},
  url={https://github.com/flashinfer-ai/flashinfer}
}
```
