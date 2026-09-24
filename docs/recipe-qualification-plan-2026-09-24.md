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

For each row below, finish all gates before starting the next row. The current main CLI supports recipe download and read-only profile preview, but its profile-load API does not bind a load to a previously reviewed plan digest. The campaign is therefore preview-only until the digest compare-and-swap prerequisite in step 5 is implemented and deployed.

1. Run `scripts/qualify-recipe --level structural` against the exact platform and recipe commits. Bind its result to the authority row's recipe content digest and package SHA-256.
2. Read current Fleet inventory and capacity through the Controller. Choose one compatible Spark for a one-node row or one compatible two-Spark group for a dual row. Use an explicitly dedicated whole-fleet profile reserved for this campaign; the runner preview writes only the current row's recipe assignment and leaves all other Sparks idle. Before using the runner, create/use its dedicated profile and set labels `qualification-authority=nl-sequential-2c118a99` and `qualification-ledger=<SHA-256 of the canonical resolved ledger path>`; keep the ledger outside the recipe inputs and use the same resolved path throughout the campaign.
3. With the current main CLI, submit `vonkctl recipe download RECIPE` for the exact recipe and verify the required model and image assets are ready in the Controller/NAS cache. Do not rely on Spark-local copies. Review `vonkctl --profile N profile load --dry-run`; resolve named cache blockers with `recipe download` and preview again. A capacity-review gate is live: proceed only after a fresh whole-fleet preview proves fit and the required capacity acknowledgement is recorded.
4. The planned `vonk-fleet-qualify-campaign` runner is on an unmerged platform branch and is not yet the main CLI. Its pre-apply preview invocation is `vonk-fleet-qualify-campaign --manifest qualification/campaigns/nl-sequential-2c118a99.json --library-root RECIPE_ROOT --ledger LEDGER_OUTSIDE_INPUTS.jsonl --profile-number N --spark CONTROLLER_NODE_ID [--spark SECOND_CONTROLLER_NODE_ID] [--failure-spark CONTROLLER_NODE_ID] --recipe RECIPE_KEY`. Unlike the current main CLI's `profile load --dry-run`, this runner preview saves the row assignment into the explicitly dedicated profile; it changes desired profile state but does not load the profile or start a workload. It binds the campaign digest to the manifest, selected recipe, profile and explicit target-node selection; dual rows also identify the exact failure target. Use exact node IDs from fresh Controller inventory, never IDs copied from an earlier row. The runner's `--observe` mode records operator-run checkpoints; it does not itself perform a host power cycle or rank failure.
5. **Apply blocker — do not load a campaign profile yet.** On current main, `/api/profile/{number}/load` accepts only a `request_key` and recomputes the preview internally; it cannot compare the current plan against the digest reviewed in steps 3–4 or bind the exact node/failure-target selection. The existing CLI cannot enforce that compare-and-swap. Keep all rows blocked at preview until a plan-digest-bound load API is implemented and deployed and the compatible runner is merged. The API must accept the reviewed plan digest and exact target binding and refuse the load if either changed. Only then may the runner's `--apply --campaign-digest DIGEST` path be used with a fresh request key, `--accept-operator-gate RECIPE_KEY` for each operator-gated row, and `--accept-capacity-review RECIPE_KEY` for a capacity-review row whose fresh preview proves fit. These acknowledgements are future apply inputs only; they do not override a blocked or stale preview. Follow durable application progress until every selected Spark reports and the route or artifact job is ready. No recipe is physically accepted while this blocker remains.
6. Once apply is enabled, run every `smoke_cases` entry and `qualification_inputs` fixture from that authority row, including streaming and non-streaming service paths where its cases declare both. Check all assertions and retain exact recipe, package, model, image, plan, target, application, campaign and smoke identities. The runner owns the fixture smoke execution; do not replace it with a generic health check.
7. Use `vonk-fleet-qualify-campaign --manifest qualification/campaigns/nl-sequential-2c118a99.json --library-root RECIPE_ROOT --ledger LEDGER_OUTSIDE_INPUTS.jsonl --profile-number N --recipe RECIPE_KEY --observe` to record the full recovery ladder; observe mode takes no Spark, failure-Spark or review-acknowledgement flags and is available only for an already-recorded canary/checkpoint. For a one-Spark row, first capture the passing canary, then record the host offline, perform the offline restart, and observe the Spark returning with a different boot ID from serialized Fleet telemetry; stop the workload after the restart evidence is complete. For a dual-Spark row, observe failure-rank loss and route withdrawal, then rank recovery and recovered serving/smoke; stop the workload, then record and perform sequential offline restarts of both selected Sparks and verify both changed boot IDs. Preserve the exact sequence and recovery evidence; a workload restart alone is not acceptance.
8. Retain verified model files, recipe images and compatible partial-transfer checkpoints. Record pass, blocked or fail before starting the next sequence number. A capacity-review row remains in sequence and may proceed only when its fresh live preview proves fit and its explicit `--accept-capacity-review RECIPE_KEY` acknowledgement is present.

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
