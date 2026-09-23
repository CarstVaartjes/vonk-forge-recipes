# Recipe qualification plan — 2026-09-24

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

1. Run `scripts/qualify-recipe --level structural` against the exact platform and recipe commits. Bind its result to the authority row's recipe content digest and package SHA-256.
2. Read current Fleet inventory and capacity through the Controller. Choose one compatible Spark for a one-node row or one compatible two-Spark group for a dual row. Use a whole-fleet profile with only this recipe assigned; all unassigned Sparks remain idle.
3. Submit `vonkctl recipe download RECIPE` for the exact recipe and verify the required model and image assets are ready in the Controller/NAS cache. Do not rely on Spark-local copies. Then review `vonkctl --profile N profile load --dry-run`; resolve named cache blockers with `recipe download` and preview again before loading.
4. Load only the reviewed plan with its exact plan digest and a fresh request key. Follow the durable profile application by application ID or request key until every selected Spark has reported its result and the route or artifact job is ready. Use Controller-authorized operations; do not use SSH as the rollout path.
5. Run every `smoke_cases` entry and `qualification_inputs` fixture listed by that authority row, including streaming and non-streaming service paths where the declared cases cover both. Check all output assertions and retain the exact recipe, package, model, image, plan, node, application, and smoke identities.
6. Complete the full restart ladder before marking a row accepted. For a one-Spark recipe, capture a passing canary, perform an offline host restart, verify the selected node returns with a different boot ID from serialized Fleet telemetry, then rerun serving or the artifact job. For a dual-Spark recipe, prove failure-rank loss, route withdrawal, rank recovery, recovered serving, and then an offline restart of both selected hosts; verify both boot IDs changed before the final serving result.
7. Stop the workload (`cleanup: stop`) but retain verified model files, recipe images, and compatible partial-transfer checkpoints. Record pass, blocked, or fail before starting the next sequence number. A capacity-review row remains in sequence and may proceed only when its fresh live preview proves fit.

The refreshed authority is `qualification/authorities/nl-sequential-2c118a99.json`; it binds the 85-recipe v1.0.8 catalog and lists every 1- or 2-Spark recipe in sequence. Its campaign manifest has no fixed Spark IDs or concurrent lanes. Rows with `operator_acceptance_required` need an explicit per-recipe operator acknowledgement before physical execution; none is implied by structural validation or by this plan. Territorial license notices remain user information and do not create geographic admission denials. Capacity review is a fresh live-preview gate, not a permanent exclusion. The four 3+-Spark recipes are listed after the 81 in-scope rows only to make the catalog boundary auditable.

## One-by-one sequence

Rows 1–81 are the one-at-a-time campaign and are bound by the ordered authority. Rows 82–85 close the catalog audit only; they require more than two Sparks and are not in the campaign.

| # | Recipe | Nodes | Check | Campaign gate | Source review |
|---:|---|---:|---|---|---|
| 1 | `vonk-forge/step1x-3d-geometry-pytorch-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 2 | `vonk-forge/step1x-3d-label-geometry-pytorch-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 3 | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` | 1 | job / 3600s | single-Spark | current |
| 4 | `vonk-forge/skintokens-pytorch-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 5 | `vonk-forge/trellis-2-4b-pytorch-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 6 | `vonk-forge/triposg-pytorch-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 7 | `vonk-forge/flux-2-klein-4b-comfyui-single` | 1 | job / 3600s | single-Spark | current |
| 8 | `vonk-forge/pixal3d-pytorch-single` | 1 | job / 3600s | single-Spark; operator acceptance required | retained: upstream added a distinct multiview checkpoint |
| 9 | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` | 1 | service | single-Spark; operator acceptance required | current |
| 10 | `vonk-forge/lfm2-5-vl-3b-vllm-single` | 1 | service | single-Spark; operator acceptance required | current |
| 11 | `vonk-forge/lfm2-5-vl-3b-vllm028-single` | 1 | service | single-Spark; operator acceptance required | current |
| 12 | `vonk-forge/qwen3-5-9b-vllm-single` | 1 | service | single-Spark | current |
| 13 | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` | 1 | service | single-Spark | retained: recipe already uses current model pin |
| 14 | `vonk-forge/qwen3-8-27b-fp8-vllm-single` | 1 | service | single-Spark | current |
| 15 | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` | 1 | service | single-Spark; operator acceptance required | current |
| 16 | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` | 1 | job / 3600s | single-Spark | retained: README-only change |
| 17 | `vonk-forge/moss-vl-realtime-11b-pytorch-single` | 1 | job / 1800s | single-Spark | updated to d1f71a58; transcript/NOTICE identities corrected |
| 18 | `vonk-forge/ltx-2-19b-dev-bf16-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 19 | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 20 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` | 1 | service | single-Spark; operator acceptance required | retained: README-only model change |
| 21 | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` | 1 | service | single-Spark; operator acceptance required | retained: README-only model change |
| 22 | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` | 1 | service | single-Spark; operator acceptance required | current |
| 23 | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` | 1 | service | single-Spark; operator acceptance required | retained: upstream change is DFlash-only |
| 24 | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | retained: README-only change |
| 25 | `vonk-forge/qwen3-6-27b-vllm-single` | 1 | service | single-Spark | current |
| 26 | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 27 | `vonk-forge/step1x-3d-texture-pytorch-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 28 | `vonk-forge/gemma-4-26b-a4b-vllm-single` | 1 | service | single-Spark | current |
| 29 | `vonk-forge/gemma-4-26b-a4b-vllm028-single` | 1 | service | single-Spark | current |
| 30 | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` | 1 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 31 | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` | 1 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 32 | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` | 1 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 33 | `vonk-forge/ui-mate-27b-vllm-single` | 1 | service | single-Spark | retained: only benchmark/demo material changed |
| 34 | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` | 1 | service | single-Spark | current |
| 35 | `vonk-forge/qwen3-8-27b-vllm-single` | 1 | service | single-Spark | current |
| 36 | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 37 | `vonk-forge/mova-360p-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 38 | `vonk-forge/deepseek-v4-flash-0731-ds4-dspark-latency-single` | 1 | service | single-Spark | retained: upstream now spans other model/runtime paths |
| 39 | `vonk-forge/deepseek-v4-flash-0731-ds4-single` | 1 | service | single-Spark | retained: upstream now spans other model/runtime paths |
| 40 | `vonk-forge/ltx-2-19b-distilled-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 41 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` | 1 | service | single-Spark; operator acceptance required | retained: recipe already uses current model pin |
| 42 | `vonk-forge/mova-720p-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 43 | `vonk-forge/nemotron-3-5-lightning-dspark-lowmem-canary-single` | 1 | service | single-Spark; operator acceptance required | retained: README-only model change |
| 44 | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` | 1 | service | single-Spark; operator acceptance required | current |
| 45 | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
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
| 57 | `vonk-forge/ling-3-0-flash-dspark-sglang-single` | 1 | service | single-Spark; operator acceptance required | current |
| 58 | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` | 1 | service | single-Spark | current |
| 59 | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 60 | `vonk-forge/hunyuan-video-foley-xxl-pytorch-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 61 | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 62 | `vonk-forge/hunyuan3d-omni-pytorch-single` | 1 | job / 3600s | single-Spark | current |
| 63 | `vonk-forge/hunyuan-video-15-distilled-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 64 | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 65 | `vonk-forge/hunyuan-video-15-t2v-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 66 | `vonk-forge/minimax-h3-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 67 | `vonk-forge/minimax-h3-fl2va-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 68 | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` | 1 | job / 3600s | single-Spark; operator acceptance required; capacity review | current |
| 69 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` | 1 | service | single-Spark; operator acceptance required; capacity review | retained: README-only model change |
| 70 | `vonk-forge/wan-dancer-14b-pytorch-single` | 1 | job / 3600s | single-Spark; capacity review | current |
| 71 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` | 1 | service | single-Spark; capacity review | retained: fixed public image not republished |
| 72 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-single` | 1 | service | single-Spark; operator acceptance required; capacity review | retained: recipe already uses current model pin |
| 73 | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 2 | service | dual-Spark | updated: selected issue 27/55/117/210 fixes verified in pinned image |
| 74 | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 2 | service | dual-Spark | updated: selected issue 27/55/210 fixes verified in pinned image |
| 75 | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 2 | service | dual-Spark; operator acceptance required | updated: Mamba alignment/state reclamation and tool-choice fixes verified in pinned image |
| 76 | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 2 | service | dual-Spark | retained: moving SGLang main is not a release channel |
| 77 | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 2 | service | dual-Spark; operator acceptance required | retained: upstream history was rewritten |
| 78 | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 2 | service | dual-Spark; operator acceptance required | updated: preparation lock/helper waits bounded with recovery coverage |
| 79 | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 2 | service | dual-Spark; operator acceptance required | current |
| 80 | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 2 | service | dual-Spark; operator acceptance required | retained: upstream default/profile changed |
| 81 | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 2 | service | dual-Spark; operator acceptance required | retained: upstream renamed the target checkpoint |
| 82 | `vonk-forge/glm-5-3-flash-nvfp4-vllm-four` | 4 | no fixture | out of scope (>2 Sparks) | retained: upstream default changed model identity |
| 83 | `vonk-forge/glm-5-2-quanttrio-vllm-four` | 4 | no fixture | out of scope (>2 Sparks) | current |
| 84 | `vonk-forge/inkling-975b-a41b-nvfp4-sglang-eight` | 8 | no fixture | out of scope (>2 Sparks) | retained: moving SGLang main is not a release channel |
| 85 | `vonk-forge/glm-5-2-aqlm-vllm-triple` | 3 | no fixture | out of scope (>2 Sparks) | current |

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
