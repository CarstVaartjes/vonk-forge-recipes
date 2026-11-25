# Credits

This repository is a thin layer of scripts. Almost everything that makes it work was built by others. Its own work is licensed under the Apache
License, Version 2.0 ([`LICENSE`](LICENSE)). The image `scripts/prepare.sh` builds also carries TensorFold's license files from the pinned checkout.

## Model

- **[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** by Thinking Machines: the looped mixture-of-experts model, its
  ModelOpt NVFP4/MXFP8 checkpoint, its chat template and its MTP (DSpark) module. Apache-2.0, with Thinking Machines'
  [Model Acceptable Use Policy](https://thinkingmachines.ai/model-acceptable-use-policy/).
- **[Inkling-2-Mini-DFlash2](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2)** by Inco AI: the DFlash 2 block-diffusion draft model
  (two-tap dynamic convs, candidate selector) this recipe drafts with by default. Apache-2.0. Read unmodified from its pinned revision.
- **[DFlash](https://github.com/z-lab/dflash)** by Jian Chen, Yesheng Liang and Zhijian Liu (z-lab, MIT): block diffusion for speculative
  decoding, the method the draft model and its serving follow; SGLang's DFlash2 implementation is the reference for the drafter's math.
- **Blackfrost**: `Inkling-2-Mini-Derisked`, the behaviorally modified derivative this recipe serves: an edited target shard with refusals
  reduced, the MTP module unchanged. Released subject to the upstream terms and notices (its README). Not part of this repository.

## Engine

- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the TensorFold contributors:
  the Zig engine (`tensorfold-native`), its lane rounds and drafter interface, and the Inkling 2 family code. Apache-2.0 (code before 0.6.0 also MIT).
- **Jürgen Schmied** ([jschmied](https://github.com/jschmied)): the Inkling 2 Mini CUDA engine this recipe builds on, "C4"
  ([#4441](https://github.com/ashhart/TensorFold/pull/4441)), with Mini's shapes on the attention kernels (C1, #4364) and its stored NVFP4 and
  MXFP8 experts in core (C3, #4365); and the load, attention, cross-pass, slot and snapshot fixes to the lead branch
  ([#4640](https://github.com/ashhart/TensorFold/pull/4640)).
- **ajensenwaud** ([ajensenwaud](https://github.com/ajensenwaud)): Inkling 2's reply split into reasoning, text and tool calls (#4124, patch `0004`), its CUDA attention kernels (#4125), its embedding, residual mix, conv,
  router, SwiGLU and pick kernels (#4126), and Inkling 2 Small's DSpark drafter kernels ([#4128](https://github.com/ashhart/TensorFold/pull/4128)):
  the block attention, Markov context, recurrence and vocabulary kernels this recipe's drafter adapts.
- **BlivionIaG** ([BlivionIaG](https://github.com/BlivionIaG)): Inkling 2 Mini on Metal with its bundled DSpark drafter
  ([#4337](https://github.com/ashhart/TensorFold/pull/4337)), the reference for the drafter's math, taps and folded Markov head.
- **sethforprivacy** ([sethforprivacy](https://github.com/sethforprivacy)): the DSpark fixes on Metal (FP32 kappa, the 1/sqrt(128) attention scale,
  [#4447](https://github.com/ashhart/TensorFold/pull/4447)) and Mini's serving fixes.
- **TensorFold's GLM-5.3-Flash CUDA DFlash2 drafter** (TensorFold contributors): the context rings, two-tap convs, chunked block attention,
  top-16 and selector this recipe's Inkling DFlash2 drafter (patch `0018`) ports to Mini's widths.
- **tonyd2wild** ([tonyd2wild](https://github.com/tonyd2wild)): an independent Inkling 2 Mini DFlash2 patch for TensorFold
  ([Inkling-2-Mini-NVFP4-DGX-Spark](https://github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark), Apache-2.0). Patch `0018` takes its NVFP4
  draft-head copy (`fp4Of`) and its lone-stream draft chances calibrated by the lanes' Landing table.
- **gilby** ([gilby](https://github.com/gilby)): Mini's Metal kernels, DSpark prompt cache and snapshot fixes.
- **akol1** ([akol1](https://github.com/akol1)): the plan for Inkling 2 Mini on CUDA and Metal in claimable slices
  ([#4274](https://github.com/ashhart/TensorFold/pull/4274)).
- **[xgrammar](https://github.com/mlc-ai/xgrammar)** by the MLC-AI team (Apache-2.0): `libtfgrammar.so`, built from the patched tree over xgrammar
  v0.2.8 at its pinned commit (its license, NOTICE and dlpack's license ship in the image).
- **[libjpeg-turbo](https://libjpeg-turbo.org)** 3.1.4.1 (IJG, BSD-3-Clause and zlib licenses) and **[libwebp](https://chromium.googlesource.com/webm/libwebp)**
  1.6.0 (BSD-3-Clause) by their authors: the JPEG and WebP decoders for image inputs, built in the image (licenses under `LICENSES-codecs`).
- **SvangenStudios**: the reply-grammar leak fix in patch `0017` (their PR #28, by way of the GLM-5.3-Flash DGX Sparks recipe's patch 0007).
- **[Basalt](https://github.com/jesdga95/basalt)** by [jesdga95](https://github.com/jesdga95): the Blackwell prefill and chunking ideas behind patches
  `0014` and `0015`.
- **[Zig](https://ziglang.org)** by the Zig Software Foundation and contributors.
- **NVIDIA**: the [PyTorch container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) the image is based on, and Model Optimizer
  (the checkpoint's format).

## Recipe

- **MiaAI-Lab** (Mia's AI Lab): this repository's scripts, after its
  [Qwen3.8 Flash Next recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold), whose `start.sh`, `prepare.sh` and
  `tools/` it adapts.
