# GLM upstream review

Checked at `2026-09-05T23:15:00+02:00` (Europe/Amsterdam), against exact
recipes main57 `57f61176510938542ad50d1d7d86ed8d052c7369` (PR70 merge) and the
frozen review inputs in `/private/tmp/vonk-all-recipe-upstream-review-20260905`.
The assigned GLM coverage contains these seven IDs: `glm-5-2-aqlm-vllm-triple`,
`glm-5-2-quanttrio-vllm-four`, `glm-5-3-flash-exl3-dflash2-vllm-dual`,
`glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual`,
`glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual`,
`glm-5-3-flash-nvfp4-vllm-dual`, and `glm-5-3-flash-nvfp4-vllm-four`.
The live source audit records the corresponding source statuses. This report keeps
source revisions, Model revisions, runtime images, and physical acceptance as
separate evidence boundaries. No package, catalog index, generator, workflow,
or upstream-watch file is changed.

## Results

### `vonk-forge/glm-5-2-aqlm-vllm-triple`

- **Checked immutable refs:** source `MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks@5c85163ccb8d98d395880e71e2dbd03976a3f4ad`; model `jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid@c5d93567f1ff2de4dbba6018b58a653654c1309a` (751 files); runtime image `ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks@sha256:f8f350d46b33858eda4f0c0a5c39a7f0b27005111d393098baaaacae18c10fdb`.
- **Evidence/decision:** upstream source remains current and documents the v4.5 NVFP4/FP8, MoonViT vision, TP3/DCP1, MTP3, and AQLM profile already represented by the recipe. Decision: **already current**, intentionally distinct TP3/vision/AQLM artifact.
- **Applied commit:** none.
- **Blocker:** physical TP3 acceptance remains separate; no upstream drift blocker.
- **Sources:** [source](https://github.com/MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks/tree/5c85163ccb8d98d395880e71e2dbd03976a3f4ad), [model](https://huggingface.co/jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid/tree/c5d93567f1ff2de4dbba6018b58a653654c1309a).

### `vonk-forge/glm-5-2-quanttrio-vllm-four`

- **Checked immutable refs:** source `drowzeys/keys-GLM5.2-Quantrio-INT4-INT8-Mixed-Abliterated-C1-30toks-4x-DGX-Sparks@296699b44e024de938cb8b3251d49d4b1306a507`; model `QuantTrio/GLM-5.2-Int4-Int8Mix@1d3bcfe5ec549ecd000fd80b37f191183842e983` (142 files); runtime image `ghcr.io/drowzeys/vllm-node-tf5-glm52-b12x@sha256:e006935eb4f8266705f213c369de1eac8de7d20417254c5f234601a2fd56d481`.
- **Evidence/decision:** the pinned source is archived and marks this C1-30/MTP-only recipe superseded, redirecting four-Spark users to a different DFlash successor with different weights, drafter, and launch profile. Decision: **intentional historical variant**, retained without relabeling.
- **Applied commit:** none.
- **Blocker:** successor intake requires new immutable Model/runtime records and four-Spark qualification.
- **Sources:** [archived source](https://github.com/drowzeys/keys-GLM5.2-Quantrio-INT4-INT8-Mixed-Abliterated-C1-30toks-4x-DGX-Sparks), [successor](https://github.com/drowzeys/keys-latest-GLM-5.2-Quantrio-INT4-INT8-Mixed-Abliterated-DFlash-4x-DGX-Sparks).

### `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual`

- **Checked immutable refs:** pinned source `MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks@493cb88fc69f8ba73ac87404f429d763e2739d89`; current source head `3021f24c88a0904c768c46ff22a508407e31360a`; target `Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw@024db9f7e9871e8efdf21538ba55af7442be3cd5`; DFlash2 `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a`; runtime image `ghcr.io/miaai-lab/glm-5.3-flash-2x-dgx-sparks@sha256:9bb1557a4234fce63d59599e44d10747eabd742beb337eebf9e7070be8a0fd58`.
- **Evidence/decision:** the target Model was already advanced to 024db9f7. Later upstream launcher defaults use E2/7168 tokens, while this recipe intentionally pins the tested 493cb88 image and 2048 prefill profile. Decision: **intentional pinned variant**.
- **Applied commit:** none.
- **Blocker:** newer launcher/image is not vendored or physically canaried.
- **Sources:** [history](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/commits/main), [target](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw/tree/024db9f7e9871e8efdf21538ba55af7442be3cd5), [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a).

### `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual`

- **Checked immutable refs:** consumed source `tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark@3eef46632c45ffb6c397de0716c23b3d2d594798`; current source `@050081dc41ce6edd4d3f15fa19dc3410ba4210e3`; target `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-43-mtp-l45@80b6d18d77e3020f2384597081d405f19893f101`; DFlash2 `incoai/GLM-5.3-Flash-DFlash2@bf582e4eacc1810f76656d1811693ff6c6737d2a`; runtime image `ghcr.io/tonyd2wild/vllm-glm53-flash@sha256:4def0ef644cb2e9814136dcffd5e385e21bc594f48f3b292234051904abe85a6`.
- **Evidence/decision:** upstream 050081dc changes the TP2 launcher KV allocation from 3 GiB to 6 GiB while retaining 262K context, DFlash2 K7, FP8 E4M3 KV, eager execution, and 8192 batched tokens. The target and companion Model revisions remain unchanged. Decision: **updated** to recipe release 1.0.5.
- **Applied commit:** `21ef2fb`.
- **Blocker:** physical two-Spark acceptance remains pending; the consumed runtime source and image remain pinned.
- **Sources:** [current source](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/tree/050081dc41ce6edd4d3f15fa19dc3410ba4210e3), [target](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-43-mtp-l45/tree/80b6d18d77e3020f2384597081d405f19893f101), [DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/bf582e4eacc1810f76656d1811693ff6c6737d2a).

### `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual`

- **Checked immutable refs:** source `drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated@c36b5958412158a69629e7fbed321312e6d0761d`; specialized Model `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-45-anchorstock@d7f8afa81e62c156446fda43304d704563622590`; runtime image `ghcr.io/drowzeys/keys-vllm-glm53-flash-nvfp4-ablit@sha256:f722ec19d8260833e948d5bf46949d9ac574841860060caa24213cf550d1a41b`.
- **Evidence/decision:** source head b4e75478 redirects the abliterated L15-43 identity used by the separate DFlash2 recipe, while this assigned recipe remains the specialized L15-45/KV/1M artifact with its 1M launcher and 6,334,808,064-byte KV allocation. Decision: **intentional specialized variant; blocked for refresh**.
- **Applied commit:** none.
- **Blocker:** the gated redirected snapshot needs authenticated inventory before changing this specialized Model; do not merge it with the DFlash2 L15-43 Model.
- **Sources:** [source history](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated/commits/main), [redirect target](https://huggingface.co/drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-43-mtp-l45).

### `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual`

- **Checked immutable refs:** source `MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark@aed98a13ca75140d2691cc5c651ea5817d9a3e44`; model `LibertAIDAI/GLM-5.3-Flash-NVFP4@caca4e6a4ebbd66f159d3d2fc256683fd6e27177`; runtime image `docker.io/vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`.
- **Evidence/decision:** source and Model are current; Ray TP2, Marlin, MTP4, FP8 E4M3 KV, multimodal serving, and the conservative GMU profile remain the distinct standard dual-Spark artifact. Decision: **already current**.
- **Applied commit:** none.
- **Blocker:** physical dual-Spark acceptance remains separate.
- **Sources:** [history](https://github.com/MiaAI-Lab/GLM-5.3-Flash-NVFP4-Dual-DGX-Spark/commits/main), [model](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/tree/caca4e6a4ebbd66f159d3d2fc256683fd6e27177).

### `vonk-forge/glm-5-3-flash-nvfp4-vllm-four`

- **Checked immutable refs:** source `tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark@98fc5d8fd48e7d1e2e95499f93cf90ceef14faf4`; current source `@8fd2fcd27c04c7fa93e770000b818657f338875d`; model `LibertAIDAI/GLM-5.3-Flash-NVFP4@caca4e6a4ebbd66f159d3d2fc256683fd6e27177`; runtime image `docker.io/vllm/vllm-openai@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce`.
- **Evidence/decision:** current upstream TP4 defaults are materially different: RedHatAI compressed-tensors is now recommended, with 24 GiB KV, 64 sequences, 16,384 batched tokens, and `FULL_AND_PIECEWISE` graphs. The recipe intentionally preserves the older, separately qualified LibertAIDAI/MTP4/eager 4-Spark profile at 9,663,676,416-byte KV. Decision: **intentional pinned variant**.
- **Applied commit:** none.
- **Blocker:** adopting current defaults requires a new runtime/image review and four-Spark canary; no compatible metadata-only update is justified.
- **Sources:** [source history](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/commits/main), [current source](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark/tree/8fd2fcd27c04c7fa93e770000b818657f338875d), [model](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/tree/caca4e6a4ebbd66f159d3d2fc256683fd6e27177).
