# Creator variant audit

This is the 2026-09-01 rescan of the public `tonyd2wild` and `drowzeys` repositories. A row is a recipe candidate only when its model identity, exact weight revision, quantization, alignment, topology, runtime image, and source-bundle closure can be pinned. Repository names and tags are discovery hints, not alignment declarations.

| Creator | Candidate repository | Model / version | Quantization | Alignment | Topology | Intake state |
| --- | --- | --- | --- | --- | --- | --- |
| tonyd2wild | [MiniMax-M3-2x-DGX-Spark-36-tok-s](https://github.com/tonyd2wild/MiniMax-M3-2x-DGX-Spark-36-tok-s) | MiniMax M3 428B | W4A16 GPTQ + 4-bit KV + EAGLE3 | standard | 2 Sparks, TP2 | Candidate; pin HF revisions and close the custom vLLM build |
| tonyd2wild | [MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark](https://github.com/tonyd2wild/MiMo-V2.5-TP2-1M-NVFP4-KV-2xDGX-Spark) | MiMo V2.5 Omni | NVFP4 weights + NVFP4 KV | standard | 2 Sparks, TP2 | Candidate; public image exists, but the custom vLLM build needs a reproducible source bundle |
| tonyd2wild | [Hy3-295B-NVFP4-MTP-2x-DGX-Spark](https://github.com/tonyd2wild/Hy3-295B-NVFP4-MTP-2x-DGX-Spark) | Hunyuan 3 295B | NVFP4 W4A16 | standard | 2 Sparks, TP2 + MTP | Candidate; pin the serving image and vendored overlays |
| tonyd2wild | [GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Int4-Int8Mix-TP4-4x-DGX-Spark) | GLM 5.3 | Int4/Int8 mix | standard | 4 Sparks, TP4 + MTP4/DFlash2 | Candidate; Triton overlays and license closure remain to be packaged |
| tonyd2wild | [Deepseek-V4-Flash-TP4-4x-DGX-Spark](https://github.com/tonyd2wild/Deepseek-V4-Flash-TP4-4x-DGX-Spark) | DeepSeek V4 Flash | NVFP4 family | standard | 4 Sparks, TP4 | Candidate; exact public weight and runtime revisions need pinning |
| drowzeys | [vLLm-0.24-optimized-NVIDIA-Nemotron-Lab-Puzzle-75B-A9B-A4Q-MTP3-NVFP4-KV-2.7M-Pool-Single-DGX-Spark](https://github.com/drowzeys/vLLm-0.24-optimized-NVIDIA-Nemotron-Lab-Puzzle-75B-A9B-A4Q-MTP3-NVFP4-KV-2.7M-Pool-Single-DGX-Spark) | Nemotron Labs 3 Puzzle 75B A9B | A4Q + NVFP4 KV | standard | 1 Spark, MTP3 | Candidate; exact checkpoint revision is not declared in the README |
| drowzeys | [Keys-NVIDIA-Two-Tower-Diffusion--dual-dgx-spark](https://github.com/drowzeys/Keys-NVIDIA-Two-Tower-Diffusion--dual-dgx-spark) | Nemotron Two-Tower 30B | Diffusion towers | standard | 1 or 2 Sparks | Candidate job recipe; serving contract differs from OpenAI recipes |
| drowzeys | [keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark](https://github.com/drowzeys/keys-vLLm.0.27-Qwen3.8-27B-ADay777Ablit-NVFP4-A4Q-NVFP4-KV-4M-KV-token-pool-MTP3-Single-DGX-Spark) | Qwen 3.8 27B ADay777 | NVFP4/A4Q + NVFP4 KV | abliterated | 1 Spark, MTP3 | High-value variant; exact checkpoint revision and source bundle still need pinning |
| drowzeys | [DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-1M-57toks](https://github.com/drowzeys/DeepSeek-V4-Flash-DSpark-Abliterated-Uncensored-1M-57toks) | DeepSeek V4 Flash | NVFP4 family | abliterated | 2 Sparks, 1M | High-value variant; v1.0 and v1.1 are separate recipe choices |
| drowzeys | [keys-MaxThink-Abliterated-DeepSeekV4-Flash-Vision-EXP](https://github.com/drowzeys/keys-MaxThink-Abliterated-DeepSeekV4-Flash-Vision-EXP) | DeepSeek V4 Flash Vision EXP | Abliterated checkpoint | abliterated | 2 Sparks | Candidate text-only runtime; README says image calls are not supported |
| drowzeys | [keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated) | GLM 5.3 Flash | NVFP4 + NVFP4 KV | abliterated | 2 Sparks, 1M | Already represented by `glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` |

The existing GLM recipe now declares `metadata.alignment: "abliterated"`. Other legacy recipes remain `unspecified` until their authorship and checkpoint alignment are explicitly reviewed; the catalog must not infer `standard` or `abliterated` from titles or tags.

## Live HEAD rescan (2026-09-01)

The creator repositories were checked again against their current default-branch
heads. A moved GitHub commit is not itself a recipe update: the local source
bundle, runtime distribution, model-version inventory, and exact artifact
revisions must move together.

| Creator / source | Current head | What changed | Catalog action |
| --- | --- | --- | --- |
| [antirez/ds4](https://github.com/antirez/ds4) | `b0982a1b` | Adds GLM 5.3 Flash text and vision graphs, current CUDA/Spark instructions, and a 1.1 GB vision encoder (`ae23e14c…`). | Two new candidates are now well-defined: GLM 5.3 Flash Q2 single Spark, and the same profile with the vision encoder. They remain blocked until the `ds4-spark` runtime is rebuilt from this head; the checked-in runtime bundle is still `84cc8823`. |
| [MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) | `d97c808e` | Vision-Exp abliterated swap and opt-in worker-over-NFS support; the existing recipe is pinned to `7963d432`. | No new executable row yet. The official and gated abliterated model inventories must be refreshed with an authenticated Hugging Face token before changing the model-version pin. |
| [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks) | `c190db1a` | EXL3 fat-expert prefill, indexer workspace sizing, E2 diagnostics, and spinwait changes; recipe is pinned to `493cb88f`. | Runtime/source-bundle refresh required; no distinct checkpoint variant was published by this delta. |
| [MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks) | `169fbad2` | Default branch history was rewritten, so GitHub cannot compare it to the recipe pin `0f950012`. | Re-audit the complete tree and exact Hugging Face revisions before updating or adding a row. |
| [drowzeys/keys-vLLm…Abliterated](https://github.com/drowzeys/keys-vLLm.0.27.1-GLM-5.3-Flash-NVFP4-NVFP4KV-1M-Context-Abliterated) | `b4e75478` | Launcher now uses the renamed gated checkpoint `drowzeys/keys-GLM-5.3-Flash-NVFP4-ablit-l15-43-mtp-l45` at HF revision `80b6d18d`; current recipe still points at the withdrawn `l15-45-anchorstock` inventory. | Existing row is stale but cannot be safely rewritten anonymously: 13 non-LFS metadata files require authentication for an exact 139-file inventory. |
| [tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-1M-KV-4x-DGX-Spark) | `1ffba70d` | 1M is the launcher default and a reproducible DFlash2 overlay is now shipped; recipe is pinned to `98fc5d8f`. | DFlash2 is a candidate variant, but needs its companion checkpoint, source bundle, and runtime digest added as one closed contract. |
| [sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark](https://github.com/sfxnz/GLM-5.3-Flash-NVFP4-vLLM-2x-DGX-Spark) | `a27e6541` | Distinct two-Spark vLLM/DFlash2 recipe: LibertAIDAI calibrated NVFP4 `caca4e6`, a 327,680-token safe window, DFlash2-7, CUDA graphs, and vision/tool smoke coverage is now wired. | High-value duplicate creator variant. It is not yet in the library because its `glm53-sm121-v11` image chain and DFlash2 patches are not the checked-in Vonk source bundle; package those plus the pinned draft snapshot before intake. |
| [sfxnz/DeepSeek-V4-Flash-Vision-Exp-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4-Flash-Vision-Exp-vLLM-2x-DGX-Spark) | `025544a3` | New two-Spark native-vision profile: DeepSeek Vision-Exp snapshot `86f746b3`, B12X fused DSpark-6, 1M context, fp8 KV 12 GiB pin, and a custom multimodal vLLM plugin; latest commits wire `image_url` and tool smoke tests. | High-value duplicate creator variant of the existing Mia DS4 recipe. Add after packaging the pinned B12X base image, `dsv4_vision` plugin, router caveat, and exact source/runtime digests as a Vonk bundle. |
| [sfxnz/Ornith-1.5-35B-A3B-NVFP4-DGX-Spark](https://github.com/sfxnz/Ornith-1.5-35B-A3B-NVFP4-DGX-Spark) | `45c2e78d` | New single-Spark official-vLLM profile for `ornith-ai/Ornith-1.5-35B-A3B-NVFP4` snapshot `9660379a`, ModelOpt NVFP4/FP8, MTP-3, 262K context, and pinned vLLM 0.27.1 image digest. | Added as `ornith-1-5-35b-a3b-nvfp4-vllm-single` using the generic pinned vLLM runtime; physical Spark readiness remains pending. |
| [sfxnz/Nemotron-3.5-Lightning-vLLM-DGX-Spark](https://github.com/sfxnz/Nemotron-3.5-Lightning-vLLM-DGX-Spark) | `fca6a4e8` | Measured single-Spark official-vLLM profile for the same NVIDIA Nemotron 3.5 Lightning NVFP4 + DSpark model already represented in the catalog; creator defaults are util `0.8`, 262K, four sequences, and DSpark-3. | No new model recipe is required. Treat it as a creator-specific performance/profile duplicate only if we want to expose its lower-utilization settings; the README does not pin immutable HF revisions. |
| [sfxnz/spark-recipes](https://github.com/sfxnz/spark-recipes) | `217205c0` | Repository contains only a short README; no recipe files or executable serving contracts. | No catalog addition. |
| [sfxnz/L.A.I.L](https://github.com/sfxnz/L.A.I.L) | `cae039b5` | Serve/eval console with HF auto-configuration, vLLM/llama.cpp backends, and Spark topology discovery; it is an orchestration tool, not a model recipe. | No catalog addition; potentially useful as upstream evidence for future recommendation/benchmark metadata. |
| `mialabs`, `sfxnz` GitHub users | `0` public model repositories for `mialabs`; the linked `sfxnz` repositories are covered above | The exact account typo was corrected during this rescan. | No other additional source was found under those exact accounts. `MiaAI-Lab` remains the relevant organization namespace and was scanned above. |

The current rescan therefore adds two concrete antirez/DS4 candidate rows,
three additional sfxnz model profiles (GLM, DeepSeek Vision-Exp, and Ornith),
and identifies one stale gated drowzeys row that needs an authenticated
inventory refresh. Nemotron is already represented by existing model rows;
`spark-recipes` and `L.A.I.L` are tooling/evidence repositories rather than
catalog recipes. None of these upstream findings is silently promoted to an
executable recipe without a closed installable contract.

## Mia ablation-toggle audit

The current Mia DeepSeek launchers expose explicit standard/abliterated
switches, so these should be represented as separate catalog choices when the
corresponding payload is closed:

| Mia launcher | Standard path | Abliterated path | Variant shape | Catalog status |
| --- | --- | --- | --- | --- |
| [DeepSeek-v4-Flash-One-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-One-DGX-Spark) | `ABLATE=0` | `ABLATE=1` | Same `0xSero/deepseek-v4-flash-0731-spark` checkpoint; runtime mounts the bundled `direction_r1.pt` and model overlay only for the ablated lane | Pair not yet materialized: the checked-in Vonk SparkInfer adapter predates the toggle and does not contain the direction/overlay payload |
| [DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) | `ABLITERATED=0` | `ABLITERATED=1` | Same dual-Spark Vision-Exp runtime profile, but the abliterated lane selects the separate gated `drowzeys/keys-DeepSeekV4Flash-Vision-EXP-ablit` checkpoint | Pair not yet materialized: the abliterated Hugging Face inventory is gated (exact manifest/hash closure requires authentication) |

These are genuine variants, not merely tags: the first changes the runtime
ablation overlay while preserving the model bytes; the second changes the
checkpoint selected by the launcher. The existing Vonk recipes remain
`alignment: unspecified` until those two exact contracts are packaged and
validated. No equivalent explicit ablation switch was found in the Mia Qwen,
GLM, Ling, Laguna, or image repositories during this scan.
