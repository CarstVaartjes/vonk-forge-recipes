# Attribution and scope

The MIT license in this repository covers r0b0tlab's original adapter, launch/evaluation integration and packaging work. It does not relicense model weights, the base container, bundled upstream libraries or external benchmark data.

- DeepSeek supplies DeepSeek-V4.1-Flash, its architecture, checkpoint, encoding and bundled speculative head: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash . Use the checkpoint under its own terms; no weights are distributed here.
- SGLang supplies the serving engine and native operators: https://github.com/sgl-project/sglang . The exact engine revision and base-image digest are recorded in the Dockerfile and runtime provenance.
- FlashInfer, PyTorch, Triton, NVIDIA CUDA/CUTLASS and NCCL supply runtime kernels and transport, under their respective upstream licenses.
- The frozen Q200v2 subset and sandbox driver are r0b0tlab tooling: https://github.com/r0b0tlab/r0b0bench/tree/main/subsets/q200v2 . BFCL uses the unmodified official evaluator and selected cases, not a forked leaderboard category. GSM8K, HumanEval, IFEval, CV-Bench and MMVP retain their upstream dataset/evaluator terms.
- LMSYS/SGLang's serving guidance and community Spark investigations informed hardware analysis. No community recipe dump is vendored as original implementation. External throughput reports with different prompts or methods are not direct matched comparisons.

Public score ledgers exclude benchmark prompts/responses and model weights. Older diagnostic source/results remain explicitly historical; they are not evidence for the current profile.
