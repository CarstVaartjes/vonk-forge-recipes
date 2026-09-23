# Recipe qualification plan — 2026-09-23

This plan covers every recipe in the 85-recipe catalog after the 2026-09-23
source audit. The audited source snapshot includes the MOSS-VL-Realtime refresh
and metadata correction, the target-only SparkInfer XGrammar fix, the bounded
dual-Qwen preparation path, and selected DeepSeek/GLM dual-runtime backports.
It deliberately keeps source review, repository validation, publication,
Controller deployment, and physical Spark acceptance as separate evidence gates.

## Upstream audit outcome

The live audit checked 178 watched model and recipe sources at
`2026-09-23T19:22:37Z`: 119 matched their watched channel and 59 channels had
advanced. An advanced channel was not treated as an automatic update. Exact
selected files, executable closure, model identity, topology, and specialized-fork
behavior were compared first.

The audit closed five actionable recipe paths. MOSS-VL-Realtime moves from
`25e81cb9` to `d1f71a58` without changing its BF16 weight shards, adopts the
cached-vision/runtime fixes, and reports the correct separate weight and
remote-code identities. The target-only SparkInfer recipe installs a hash-bound
offline XGrammar 0.2.3 wheel and adds forced-tool-call coverage. The dual-Qwen
vLLM recipe bounds its shared preparation lock and helpers. The two Mia DeepSeek
recipes gain selected serialization, prefill-accounting, XGrammar, and shared-
memory recovery fixes; the Mia GLM recipe gains selected Mamba alignment/state
reclamation and `tool_choice:none` fixes. Exact-image patch gates were rerun for
the Mia recipes. Other advanced channels are intentionally retained where only
README/unselected files changed, the upstream default changed model identity, a
moving engine main is not a release channel, history was rewritten, or the newer
implementation lacks a closed and requalified Vonk source/image path.

## Per-recipe procedure

For each row below, finish all gates before starting the next row:

1. Run `scripts/qualify-recipe --level structural` against the exact platform and recipe commits.
2. Confirm the published package, Model snapshots, recipe content digest, and runtime-image/source-build identity match that structural result.
3. On the Controller, create one profile for only this recipe, run `profile prepare-cache`, inspect the exact model/image blockers, and do not apply until the cache is ready.
4. Apply to the named node set, follow durable progress, and wait for route or job readiness. Do not use SSH as the rollout path.
5. Execute the qualification-index smoke case(s), including streaming and non-streaming for OpenAI services where declared, and retain the evidence ledger with the exact recipe/platform/fleet identities.
6. Exercise one recovery boundary: restart the workload for single-node recipes; for multi-node recipes, withdraw one rank and prove route withdrawal plus worker-first recovery.
7. Stop the workload (`cleanup: stop`) but retain verified models, images, and partial-transfer checkpoints for the next row. Record pass, blocked, or fail before continuing.

The existing authority is stale: it names 84 recipes at `7173cb48`, while the
current catalog has 85. Regenerate the authority and campaign from the final
published recipe commit before any apply. License acknowledgements are operator
information/acceptance steps, not territorial admission denials. Capacity and
topology are genuine execution constraints.

## One-by-one sequence

| # | Recipe | Nodes | Check | Current disposition | Source review |
|---:|---|---:|---|---|---|
| 1 | `vonk-forge/step1x-3d-geometry-pytorch-single` | 1 | job / 3600s | single-Spark | current |
| 2 | `vonk-forge/step1x-3d-label-geometry-pytorch-single` | 1 | job / 3600s | single-Spark | current |
| 3 | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` | 1 | job / 3600s | single-Spark | current |
| 4 | `vonk-forge/skintokens-pytorch-single` | 1 | job / 3600s | single-Spark | current |
| 5 | `vonk-forge/trellis-2-4b-pytorch-single` | 1 | job / 3600s | single-Spark | current |
| 6 | `vonk-forge/triposg-pytorch-single` | 1 | job / 3600s | single-Spark | current |
| 7 | `vonk-forge/flux-2-klein-4b-comfyui-single` | 1 | job / 3600s | single-Spark | current |
| 8 | `vonk-forge/pixal3d-pytorch-single` | 1 | job / 3600s | single-Spark | retained: upstream added a distinct multiview checkpoint |
| 9 | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` | 1 | service | single-Spark | current |
| 10 | `vonk-forge/lfm2-5-vl-3b-vllm-single` | 1 | service | single-Spark | current |
| 11 | `vonk-forge/lfm2-5-vl-3b-vllm028-single` | 1 | service | single-Spark | current |
| 12 | `vonk-forge/qwen3-5-9b-vllm-single` | 1 | service | single-Spark | current |
| 13 | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` | 1 | service | single-Spark | retained: recipe already uses current model pin |
| 14 | `vonk-forge/qwen3-8-27b-fp8-vllm-single` | 1 | service | single-Spark | current |
| 15 | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` | 1 | service | single-Spark | current |
| 16 | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` | 1 | job / 3600s | single-Spark | retained: README-only change |
| 17 | `vonk-forge/moss-vl-realtime-11b-pytorch-single` | 1 | job / 1800s | single-Spark | updated to d1f71a58; transcript/NOTICE identities corrected |
| 18 | `vonk-forge/ltx-2-19b-dev-bf16-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 19 | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` | 1 | job / 3600s | single-Spark | current |
| 20 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` | 1 | service | single-Spark | retained: README-only model change |
| 21 | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` | 1 | service | single-Spark | retained: README-only model change |
| 22 | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` | 1 | service | single-Spark | current |
| 23 | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` | 1 | service | single-Spark | retained: upstream change is DFlash-only |
| 24 | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` | 1 | job / 3600s | single-Spark | retained: README-only change |
| 25 | `vonk-forge/qwen3-6-27b-vllm-single` | 1 | service | single-Spark | current |
| 26 | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 27 | `vonk-forge/step1x-3d-texture-pytorch-single` | 1 | job / 3600s | single-Spark | current |
| 28 | `vonk-forge/gemma-4-26b-a4b-vllm-single` | 1 | service | single-Spark | current |
| 29 | `vonk-forge/gemma-4-26b-a4b-vllm028-single` | 1 | service | single-Spark | current |
| 30 | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` | 1 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 31 | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` | 1 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 32 | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` | 1 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 33 | `vonk-forge/ui-mate-27b-vllm-single` | 1 | service | single-Spark | retained: only benchmark/demo material changed |
| 34 | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` | 1 | service | single-Spark | current |
| 35 | `vonk-forge/qwen3-8-27b-vllm-single` | 1 | service | single-Spark | current |
| 36 | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 37 | `vonk-forge/mova-360p-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 38 | `vonk-forge/deepseek-v4-flash-0731-ds4-dspark-latency-single` | 1 | service | single-Spark | retained: upstream now spans other model/runtime paths |
| 39 | `vonk-forge/deepseek-v4-flash-0731-ds4-single` | 1 | service | single-Spark | retained: upstream now spans other model/runtime paths |
| 40 | `vonk-forge/ltx-2-19b-distilled-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 41 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` | 1 | service | single-Spark | retained: recipe already uses current model pin |
| 42 | `vonk-forge/mova-720p-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 43 | `vonk-forge/nemotron-3-5-lightning-dspark-lowmem-canary-single` | 1 | service | single-Spark | retained: README-only model change |
| 44 | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` | 1 | service | single-Spark | current |
| 45 | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 46 | `vonk-forge/qwen-image-2512-comfyui-single` | 1 | job / 3600s | single-Spark | retained: README-only change |
| 47 | `vonk-forge/qwen-image-2512-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 48 | `vonk-forge/qwen-image-2512-lightning-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 49 | `vonk-forge/qwen-image-edit-2511-comfyui-single` | 1 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 50 | `vonk-forge/qwen-image-edit-2511-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 51 | `vonk-forge/qwen-image-edit-2511-lightning-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 52 | `vonk-forge/qwen-image-layered-diffusers-single` | 1 | job / 3600s | single-Spark | current |
| 53 | `vonk-forge/wan-2-2-i2v-14b-comfyui-single` | 1 | job / 3600s | single-Spark | retained: README-only change |
| 54 | `vonk-forge/wan-2-2-t2v-14b-comfyui-single` | 1 | job / 3600s | single-Spark | retained: README-only change |
| 55 | `vonk-forge/wan-dancer-14b-disk-offload-pytorch-single` | 1 | job / 3600s | single-Spark | retained: Wan-Dancer adapter path unchanged |
| 56 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single` | 1 | service | single-Spark | updated: SHA-pinned XGrammar 0.2.3 bundled offline; forced tool-call smoke added |
| 57 | `vonk-forge/ling-3-0-flash-dspark-sglang-single` | 1 | service | single-Spark | current |
| 58 | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` | 1 | service | single-Spark | current |
| 59 | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` | 1 | job / 3600s | operator acceptance | current |
| 60 | `vonk-forge/hunyuan-video-foley-xxl-pytorch-single` | 1 | job / 3600s | operator acceptance | current |
| 61 | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` | 1 | job / 3600s | operator acceptance | current |
| 62 | `vonk-forge/hunyuan3d-omni-pytorch-single` | 1 | job / 3600s | operator acceptance | current |
| 63 | `vonk-forge/hunyuan-video-15-distilled-diffusers-single` | 1 | job / 3600s | operator acceptance | current |
| 64 | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` | 1 | job / 3600s | operator acceptance | current |
| 65 | `vonk-forge/hunyuan-video-15-t2v-diffusers-single` | 1 | job / 3600s | operator acceptance | current |
| 66 | `vonk-forge/minimax-h3-diffusers-single` | 1 | job / 3600s | operator acceptance | current |
| 67 | `vonk-forge/minimax-h3-fl2va-diffusers-single` | 1 | job / 3600s | operator acceptance | current |
| 68 | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` | 1 | job / 3600s | capacity review | current |
| 69 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` | 1 | service | capacity review | retained: README-only model change |
| 70 | `vonk-forge/wan-dancer-14b-pytorch-single` | 1 | job / 3600s | capacity review | current |
| 71 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` | 1 | service | capacity review | retained: fixed public image not republished |
| 72 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-single` | 1 | service | capacity review | retained: recipe already uses current model pin |
| 73 | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 2 | service | dual-Spark | updated: selected issue 27/55/117/210 fixes verified in pinned image |
| 74 | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 2 | service | dual-Spark | updated: selected issue 27/55/210 fixes verified in pinned image |
| 75 | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 2 | service | dual-Spark | updated: Mamba alignment/state reclamation and tool-choice fixes verified in pinned image |
| 76 | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 2 | service | dual-Spark | retained: moving SGLang main is not a release channel |
| 77 | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 2 | service | dual-Spark | retained: upstream history was rewritten |
| 78 | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 2 | service | dual-Spark | updated: preparation lock/helper waits bounded with recovery coverage |
| 79 | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 2 | service | dual-Spark | current |
| 80 | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 2 | service | dual-Spark | retained: upstream default/profile changed |
| 81 | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 2 | service | dual-Spark | retained: upstream renamed the target checkpoint |
| 82 | `vonk-forge/glm-5-3-flash-nvfp4-vllm-four` | 4 | no fixture | topology unavailable | retained: upstream default changed model identity |
| 83 | `vonk-forge/glm-5-2-quanttrio-vllm-four` | 4 | no fixture | topology unavailable | current |
| 84 | `vonk-forge/inkling-975b-a41b-nvfp4-sglang-eight` | 8 | no fixture | topology unavailable | retained: moving SGLang main is not a release channel |
| 85 | `vonk-forge/glm-5-2-aqlm-vllm-triple` | 3 | no fixture | topology unavailable | current |

## Evidence and stop rules

A recipe passes only with the exact structural record, published package identity,
Controller preparation/apply operation IDs, per-node transfer/start receipts,
declared serving/job assertions, and cleanup result. A repository test or healthy
container is not physical acceptance. A blocked recipe remains in sequence with its
named blocker; it is never omitted to make the campaign green.

Stop the campaign on an integrity mismatch, malformed contract, denied authority,
unexpected model/image substitution, or a failure that could affect another recipe.
A local capacity shortfall blocks only that row. Preserve completed downloads,
verified images, and exact failure evidence so the row can resume after repair.
