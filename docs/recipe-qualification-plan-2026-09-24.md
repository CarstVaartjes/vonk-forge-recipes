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

## Family-aware qualification strategy

This is the current execution strategy, updated after operator review on
2026-09-24. The current source implementation uses authority v4 with explicit
paired single-Spark batches, exclusive dual-Spark batches and typed recovery
coverage. Run it only after the compatible platform release and recipe catalog
are accepted and deployed. The checked-in authority remains bound to the exact
accepted recipe release shown in its catalog block; authored source edits do not
become executable identities until their catalog artifacts are accepted.

### Validate broadly before using the Sparks

1. Validate all 85 recipes against exact platform and library commits: canonical
   contracts, selected upstream files, licenses/access gates, companion models,
   mount paths, adapter imports and complete dependency closure. Record blockers
   without hiding a recipe from the inventory. Use off-device checks wherever
   they provide valid evidence; native ARM/CUDA builds and GPU kernels still
   require their appropriate execution environment.
2. Derive a coverage matrix from the canonical recipes. Group by exact runtime
   image/build identity, engine version or fork, CUDA/native dependencies,
   patches and adapter identity, then model family, quantization and topology.
   Merely sharing the name vLLM, SGLang or PyTorch is not evidence equivalence.
3. Build and check each exact stack once where its verified artifacts can be
   reused. Audit its whole dependency/import path before retrying a build, not
   just the last missing import. Fix shared defects once and revalidate every
   affected recipe; preserve completed cache work and durable partial progress.
4. Select a ready, low-cost representative for each distinct stack/family risk,
   then test its related variants together to reuse images and model assets.
   Each recipe still needs its own physical inference and declared assertions.

| Coverage group | Representative checks; variants requiring separate evidence |
|---|---|
| vLLM | Model loader, tokenizer/template, quantization and attention kernels; declared text, tool, structured-output and vision paths. Different image versions and native stacks remain separate. |
| SGLang | Loader and serving paths plus applicable cache, state-space, speculative-decoding and tensor-parallel behavior. A vLLM pass does not qualify SGLang. |
| Specialized forks and engines | Mia, DeepSeek-specific stacks, SparkInfer, EXL3 and DFlash retain their exact fork, patch, kernel and auxiliary-model checks; generic engine evidence cannot replace them. |
| Image/video workflows | Keep Diffusers and ComfyUI stacks distinct. Verify selected checkpoints, companion encoders/decoders, custom nodes, workflow and output validity for every recipe. |
| 3D pipelines | Step1X and other pipelines need their own import/native-extension closure and valid geometry outputs. A shared PyTorch base does not qualify another pipeline. |
| Audio and multimodal pipelines | Exercise the declared modality inputs, encoders/decoders and output assertions, not merely an HTTP response or a text-only prompt. |
| Dual-Spark variants | Independently prove distributed initialization, inter-node communication, inference and rank-loss recovery. Single-Spark evidence is only a prerequisite, not a distributed pass. |

### Two single-Spark lanes, one fleet owner

The two available Sparks may run **two independent single-Spark recipes at the
same time**. Schedule them as one paired batch under **one whole-fleet profile**,
with one recipe assigned to each Spark. Do not launch competing per-lane
profiles: every profile always owns the entire fleet, including idle nodes.

- One coordinator owns profile changes, the reviewed plan and application
  request. It binds both assignments and all replacement effects together.
  The runner must retain distinct recipe, node, smoke and result identities so
  one successful lane cannot qualify or conceal failure in the other.
- Admit a pair only after fresh Controller previews prove both workloads fit
  their respective nodes and shared storage/transfer/build resources permit it.
  A Spark busy with native compilation is not assumed free for inference.
- Run the two lane smoke suites concurrently. Initially use a batch barrier:
  preserve both results and reconcile cleanup before applying the next pair.
  Do not refill one lane by silently replacing or interrupting the other.
- Dual-Spark recipes reserve both Sparks exclusively. Disruptive recovery
  checks also run exclusively, after the other lane is stopped and its resource
  release is observed. Host restart/rank-failure actions still require an
  approved mechanism; observation mode does not perform them.
- A recipe-local failure remains local in the evidence. Integrity, authority,
  changed-plan or fleet-ownership failures stop the batch. Resume reconciles
  the original operation and completed lane receipts instead of duplicating
  loads or rerunning already-proven work unnecessarily.

### Evidence reuse and recovery coverage

Earlier working GLM-5.3 runs remain historical physical evidence; this campaign
must not describe them as never tested. Match their actual model, recipe,
image, topology, platform and test receipts before reusing any particular
claim. A changed identity invalidates the affected evidence, not automatically
every unrelated build or cached asset. Record missing provenance as an evidence
gap instead of inventing a fresh pass.

Every recipe needs an attributable inference result, including all its declared
smoke cases and fixtures. Shared build checks can be reused only for identical
artifacts. Recovery coverage definitions bind the exact runtime stack, topology
and member identities; this release has no shared recovery group, so each recipe
keeps dedicated recovery receipts. Any future shared receipt must satisfy the
typed representative and member-use contract, including its invalidation
conditions.

### Implementation and execution order

- [x] Record family-aware coverage and two single-Spark lanes in this plan.
- [x] Derive the complete 85-row stack/family matrix, representative selection,
  paired batches and recovery-coverage mapping from exact catalog identities.
  Keep all 81 two-Spark-fleet candidates visible; the four larger-topology
  recipes remain explicitly outside available physical capacity. The derived
  matrix is `qualification/coverage/family-aware-coverage-2026-09-24.md`,
  generated by `tools/build-family-aware-coverage` and bound to v1.0.16. The
  typed recovery definitions are explicit but no group is shared in this
  release; representative selection is a build-cost choice, not a readiness,
  cache or physical claim.
- [ ] Complete the paired campaign runner integration for lane evidence,
  exclusive recovery, resumable batches, competing fleet ownership, stale
  plans, wrong-recipe/node evidence, duplicate apply after disconnect and
  premature lane replacement. The canonical contract is implemented; combined
  producer/consumer tests and platform checks remain pending. Do not introduce a
  parallel client-side planner or legacy runner fallback.
- [ ] Verify runner consumption of the explicit recovery-coverage references
  before any shared recovery evidence is accepted. The current authority keeps
  full per-recipe checks because this release has no exact shared recovery group.
- [ ] Complete required review, CI, accepted publication and Controller/runner
  deployment, then regenerate and review the exact campaign authority/order.
  Respect separate schema-merge decisions if implementation changes schema.
- [ ] Reconcile Step1X's completed native-build evidence and earlier GLM
  receipts first. Run ready representative batches, then related variants;
  schedule dual-Spark and recovery work exclusively. Provider/access blockers
  stay visible and do not become physical passes.
- [ ] Produce a final per-recipe report separating source freshness,
  structural validity, cache readiness, inference and recovery evidence.

Use bounded GPT-6 Luna Max agents for independent audits or implementation,
each in its own task-owned worktree. Avoid agents for routine polling, duplicate
audits and repeated full suites on unchanged inputs. Integrate related fixes,
run focused regression checks during iteration and the required combined gates
before release. Refresh campaign authority after relevant artifacts settle,
not after every intermediate failed build.

## Current per-recipe procedure

### File-closure blockers found during cache preparation

The cache audit found authored aggregate paths (`snapshot` and
`filtered-snapshot`) that are not upstream files. Exact per-file manifests and
companion selections must replace them before those revisions can run. The
source repair closes 14 recipes: the three Step1X variants, MOVA 360p/720p,
Hunyuan3D-Omni, Foley XL/XXL, the three Hunyuan Video 1.5 variants,
SkinTokens, MiniMax H3, and TripoSG. Exact companion Models and adapter mount
selections are included. Hunyuan3D and Foley manifests omit checkpoints their
selected adapters do not use, so cache preparation does not download them.
All 14 repaired recipes passed structural compilation against platform
`0123eeb46a36f828d57b293c0b6704f5aa13d5f2`; this is not cache or physical
acceptance. The following unresolved rows remain blocked, without a physical
pass:

- **LTX 2.5:** its 28 selected files require authentic per-file SHA-256 values.
  The official pinned `audio_vae/config.json` at
  `426936f8b22dc28e4def61e515478b0b7e4a53cc` returned HTTP 401. Authorized
  Hugging Face access is required; an aggregate checksum is not a substitute.
- **TRELLIS 2 and Pixal3D:** their Meta DINOv3 dependency at
  `ea8dc2863c51be0a264bab82070e3e8836b02d51` requires provider approval and
  authenticated access. A public mirror must not silently replace that source.
- **Pixal3D additionally:** its official NAF checkpoint is a GitHub release
  asset, while the current model-cache provider only resolves Hugging Face
  model files. Its source needs a supported cache contract before execution.

### Current batch execution gates

The batch runner and authority source implementation are ready for the paired
workflow, but source merge is not platform acceptance or deployment. Before
execution, verify the compatible accepted platform release is running and that
the recipe catalog publication matches the reviewed authority. Recheck current
deployment and Fleet state; earlier observations are not current evidence.

1. Run `scripts/qualify-recipe --level structural` for each exact recipe and
   bind the result to its authority row's content digest and package SHA-256.
   Preserve all 85 catalog rows; the 81 one- and two-Spark recipes are in scope
   and the four wider topologies are explicitly excluded for capacity.
2. Prepare missing NAS assets with `vonkctl recipe download RECIPE`; Spark-local
   copies are not cache authority. Read current Fleet inventory and capacity.
   Admit a batch only when fresh whole-Fleet previews show both independent
   single-Spark lanes fit, or both Sparks fit an exclusive dual-Spark row.
3. Reserve one dedicated whole-Fleet profile with
   `installation_policy=keep-cached`, label
   `qualification-authority=nl-family-aware-20260924`, and
   `qualification-ledger=<first 63 hexadecimal characters of the SHA-256 of
   the canonical resolved ledger path>`. Keep the ledger outside recipe inputs
   and use the same resolved path throughout the campaign. Every profile owns
   the entire enrolled Fleet, including idle Sparks.
4. Preview a paired batch by passing both exact Controller node IDs in lane
   order; for an exclusive dual batch, pass its two node IDs and the selected
   rank-loss target. For example:

   ```text
   vonk-fleet-qualify-campaign --manifest qualification/campaigns/nl-family-aware-20260924.json --library-root RECIPE_ROOT --ledger LEDGER_OUTSIDE_INPUTS.jsonl --profile-number N --batch batch-001 --spark CONTROLLER_NODE_ID_LANE_1 --spark CONTROLLER_NODE_ID_LANE_2
   ```

   The preview saves the complete batch assignments into the dedicated profile
   but does not load the profile or start workloads. Use fresh node IDs and, if
   a foreign workload would be stopped, acknowledge only its exact current run
   with `--replace-run-id RUN_ID`.
5. **Deployment gate — do not load the batch profile until verified.** Apply
   only the reviewed preview using `--apply --campaign-digest DIGEST`, plus
   `--accept-operator-gate RECIPE_KEY` and/or
   `--accept-capacity-review RECIPE_KEY` for every gated recipe in that batch.
   These acknowledgements do not override a blocked or stale preview. Follow
   durable progress until both selected Sparks report ready. The runner records
   each lane's smoke result independently; run every declared smoke case and
   fixture, including streaming and non-streaming service paths where present.
6. Resume interrupted work with `--observe --batch batch-001`; observe mode uses
   the same manifest, library, ledger and profile and takes no new Spark or
   review flags. After both canaries, run recovery one lane at a time using
   `--batch batch-001 --recover-lane LANE`, review that whole-Fleet transition,
   then apply its fresh `--campaign-digest DIGEST`. For a paired single-Spark
   lane, record the offline host and changed boot ID, then observe the recovered
   smoke. For a dual-Spark lane, select `--failure-spark` in the initial batch
   preview; observe rank loss and route withdrawal, restore the failed rank,
   verify recovered serving and smoke, then complete each required host restart.
   These physical actions remain operator-run; `--observe` only records and
   reconciles their evidence.
7. Retain verified model files, recipe images and compatible partial-transfer
   checkpoints. Reconcile cleanup and the completion of both lane receipts
   before advancing to the next batch. A failure local to one lane stays local
   in the evidence; integrity, authority, changed-plan and whole-Fleet ownership
   failures stop the batch.

The current authority is
`qualification/authorities/nl-family-aware-20260924.json`. It binds the
85-recipe v1.0.16 catalog from tag `v1.0.16` (catalog commit
`7b23f1a1569e16f8f2728ddd6846cf4f7c58f0f7`, source commit
`a0ffd8735270c35b0c99b5b04cfe8956ce8c351d`), assigns 72 single-Spark recipes
to 36 paired batches and schedules nine dual-Spark recipes exclusively. It
contains 90 recipe-specific recovery definitions; no recovery group is shared
in this release. The four recipes requiring more than two Sparks remain
explicit exclusions. Operator acceptance and capacity review remain row-level
gates, and territorial license notices are informational. Refresh authority
and coverage only after the exact new catalog and packages have been accepted;
the generators read the accepted catalog pin and record current indexed-package-
versus-accepted stack divergences until then.

The v1.0.16 catalog's `source_commit` metadata points to `a0ffd873…`, whose LTX
2.19B build context contains protocol wheel 2.2.0 (SHA-256
`7555df9ec0f576ac0530d1e6abdd2f3845614cf397a7bc2c8dcd01bc86967565`). The
digest-verified package pinned by that same catalog commit contains protocol
wheel 3.0.0 (SHA-256
`519484690b626f03e27efabad787ad1040a7d527404f2dd4faa3e2d84c40d116`). This is
a stale source-pointer metadata discrepancy, not evidence that the released LTX
package lacks the newer wheel. Runtime-stack identities now hash the selected
regular files from each exact catalog-pinned package. The generated matrix
compares those identities with the current indexed package artifacts and lists
any per-recipe stack differences; it does not infer build-source differences
from the catalog's `source_commit` field. Step1X geometry 1.2.15 completed its
native build on 2026-09-24 at 14:44 UTC (operation
`6a5acb3b-d397-4fb2-8500-21779b854ebf`); this is not an inference or recovery
pass. Reconcile its exact image receipt before applying it.

## Complete inventory and current batch assignments

Keep this inventory complete. Family grouping and paired batches require a
regenerated reviewed authority, not hand-edited assignments or skipped ledger
checkpoints. The table below shows the current authority batch assignments.

<!-- generated:begin qualification-inventory -->
The authority assigns 72 one-Spark recipes to 36 batches and 9 two-Spark recipes to exclusive batches. Wider topologies close the catalog audit only.

| # | Recipe | Nodes | Batch | Lane | Check | Campaign gate | Source review |
|---:|---|---:|---|---:|---|---|---|
| 1 | `vonk-forge/step1x-3d-geometry-pytorch-single` | 1 | `batch-001` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 2 | `vonk-forge/step1x-3d-label-geometry-pytorch-single` | 1 | `batch-001` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 3 | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` | 1 | `batch-002` | 1 | job / 3600s | single-Spark | current |
| 4 | `vonk-forge/skintokens-pytorch-single` | 1 | `batch-002` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 5 | `vonk-forge/trellis-2-4b-pytorch-single` | 1 | `batch-003` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 6 | `vonk-forge/triposg-pytorch-single` | 1 | `batch-003` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 7 | `vonk-forge/flux-2-klein-4b-comfyui-single` | 1 | `batch-004` | 1 | job / 3600s | single-Spark | current |
| 8 | `vonk-forge/pixal3d-pytorch-single` | 1 | `batch-004` | 2 | job / 3600s | single-Spark; operator acceptance required | retained: upstream added a distinct multiview checkpoint |
| 9 | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` | 1 | `batch-005` | 1 | service | single-Spark; operator acceptance required | current |
| 10 | `vonk-forge/lfm2-5-vl-3b-vllm-single` | 1 | `batch-005` | 2 | service | single-Spark; operator acceptance required | current |
| 11 | `vonk-forge/lfm2-5-vl-3b-vllm028-single` | 1 | `batch-006` | 1 | service | single-Spark; operator acceptance required | current |
| 12 | `vonk-forge/qwen3-5-9b-vllm-single` | 1 | `batch-006` | 2 | service | single-Spark | current |
| 13 | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` | 1 | `batch-007` | 1 | service | single-Spark | retained: recipe already uses current model pin |
| 14 | `vonk-forge/qwen3-8-27b-fp8-vllm-single` | 1 | `batch-007` | 2 | service | single-Spark | current |
| 15 | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` | 1 | `batch-008` | 1 | service | single-Spark; operator acceptance required | current |
| 16 | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` | 1 | `batch-008` | 2 | job / 3600s | single-Spark | retained: README-only change |
| 17 | `vonk-forge/moss-vl-realtime-11b-pytorch-single` | 1 | `batch-009` | 1 | job / 1800s | single-Spark | updated to d1f71a58; transcript/NOTICE identities corrected |
| 18 | `vonk-forge/ltx-2-19b-dev-bf16-diffusers-single` | 1 | `batch-009` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 19 | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` | 1 | `batch-010` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 20 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` | 1 | `batch-010` | 2 | service | single-Spark; operator acceptance required | retained: README-only model change |
| 21 | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` | 1 | `batch-011` | 1 | service | single-Spark; operator acceptance required | retained: README-only model change |
| 22 | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` | 1 | `batch-011` | 2 | service | single-Spark; operator acceptance required | current |
| 23 | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` | 1 | `batch-012` | 1 | service | single-Spark; operator acceptance required | retained: upstream change is DFlash-only |
| 24 | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` | 1 | `batch-012` | 2 | job / 3600s | single-Spark; operator acceptance required | retained: README-only change |
| 25 | `vonk-forge/qwen3-6-27b-vllm-single` | 1 | `batch-013` | 1 | service | single-Spark | current |
| 26 | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` | 1 | `batch-013` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 27 | `vonk-forge/step1x-3d-texture-pytorch-single` | 1 | `batch-014` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 28 | `vonk-forge/gemma-4-26b-a4b-vllm-single` | 1 | `batch-014` | 2 | service | single-Spark | current |
| 29 | `vonk-forge/gemma-4-26b-a4b-vllm028-single` | 1 | `batch-015` | 1 | service | single-Spark | current |
| 30 | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` | 1 | `batch-015` | 2 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 31 | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` | 1 | `batch-016` | 1 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 32 | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` | 1 | `batch-016` | 2 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 33 | `vonk-forge/ui-mate-27b-vllm-single` | 1 | `batch-017` | 1 | service | single-Spark | retained: only benchmark/demo material changed |
| 34 | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` | 1 | `batch-017` | 2 | service | single-Spark | current |
| 35 | `vonk-forge/qwen3-8-27b-vllm-single` | 1 | `batch-018` | 1 | service | single-Spark | current |
| 36 | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` | 1 | `batch-018` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 37 | `vonk-forge/mova-360p-diffusers-single` | 1 | `batch-019` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 38 | `vonk-forge/deepseek-v4-flash-0731-ds4-dspark-latency-single` | 1 | `batch-019` | 2 | service | single-Spark | retained: upstream now spans other model/runtime paths |
| 39 | `vonk-forge/deepseek-v4-flash-0731-ds4-single` | 1 | `batch-020` | 1 | service | single-Spark | retained: upstream now spans other model/runtime paths |
| 40 | `vonk-forge/ltx-2-19b-distilled-diffusers-single` | 1 | `batch-020` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 41 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` | 1 | `batch-021` | 1 | service | single-Spark; operator acceptance required | retained: recipe already uses current model pin |
| 42 | `vonk-forge/mova-720p-diffusers-single` | 1 | `batch-021` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 43 | `vonk-forge/nemotron-3-5-lightning-dspark-lowmem-canary-single` | 1 | `batch-022` | 1 | service | single-Spark; operator acceptance required | retained: README-only model change |
| 44 | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` | 1 | `batch-022` | 2 | service | single-Spark; operator acceptance required | current |
| 45 | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` | 1 | `batch-023` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 46 | `vonk-forge/qwen-image-2512-comfyui-single` | 1 | `batch-023` | 2 | job / 3600s | single-Spark | retained: README-only change |
| 47 | `vonk-forge/qwen-image-2512-diffusers-single` | 1 | `batch-024` | 1 | job / 3600s | single-Spark | current |
| 48 | `vonk-forge/qwen-image-2512-lightning-diffusers-single` | 1 | `batch-024` | 2 | job / 3600s | single-Spark | current |
| 49 | `vonk-forge/qwen-image-edit-2511-comfyui-single` | 1 | `batch-025` | 1 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 50 | `vonk-forge/qwen-image-edit-2511-diffusers-single` | 1 | `batch-025` | 2 | job / 3600s | single-Spark | current |
| 51 | `vonk-forge/qwen-image-edit-2511-lightning-diffusers-single` | 1 | `batch-026` | 1 | job / 3600s | single-Spark | current |
| 52 | `vonk-forge/qwen-image-layered-diffusers-single` | 1 | `batch-026` | 2 | job / 3600s | single-Spark | current |
| 53 | `vonk-forge/wan-2-2-i2v-14b-comfyui-single` | 1 | `batch-027` | 1 | job / 3600s | single-Spark | retained: README-only change |
| 54 | `vonk-forge/wan-2-2-t2v-14b-comfyui-single` | 1 | `batch-027` | 2 | job / 3600s | single-Spark | retained: README-only change |
| 55 | `vonk-forge/wan-dancer-14b-disk-offload-pytorch-single` | 1 | `batch-028` | 1 | job / 3600s | single-Spark | retained: Wan-Dancer adapter path unchanged |
| 56 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-target-only-canary-single` | 1 | `batch-028` | 2 | service | single-Spark | updated: SHA-pinned XGrammar 0.2.3 bundled offline; forced tool-call smoke added |
| 57 | `vonk-forge/ling-3-0-flash-dspark-sglang-single` | 1 | `batch-029` | 1 | service | single-Spark; operator acceptance required | current |
| 58 | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` | 1 | `batch-029` | 2 | service | single-Spark | current |
| 59 | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` | 1 | `batch-030` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 60 | `vonk-forge/hunyuan-video-foley-xxl-pytorch-single` | 1 | `batch-030` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 61 | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` | 1 | `batch-031` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 62 | `vonk-forge/hunyuan3d-omni-pytorch-single` | 1 | `batch-031` | 2 | job / 3600s | single-Spark | current |
| 63 | `vonk-forge/hunyuan-video-15-distilled-diffusers-single` | 1 | `batch-032` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 64 | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` | 1 | `batch-032` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 65 | `vonk-forge/hunyuan-video-15-t2v-diffusers-single` | 1 | `batch-033` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 66 | `vonk-forge/minimax-h3-diffusers-single` | 1 | `batch-033` | 2 | job / 3600s | single-Spark; operator acceptance required | current |
| 67 | `vonk-forge/minimax-h3-fl2va-diffusers-single` | 1 | `batch-034` | 1 | job / 3600s | single-Spark; operator acceptance required | current |
| 68 | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` | 1 | `batch-034` | 2 | job / 3600s | single-Spark; operator acceptance required; capacity review | current |
| 69 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` | 1 | `batch-035` | 1 | service | single-Spark; operator acceptance required; capacity review | retained: README-only model change |
| 70 | `vonk-forge/wan-dancer-14b-pytorch-single` | 1 | `batch-035` | 2 | job / 3600s | single-Spark; capacity review | current |
| 71 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` | 1 | `batch-036` | 1 | service | single-Spark; capacity review | retained: fixed public image not republished |
| 72 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-single` | 1 | `batch-036` | 2 | service | single-Spark; operator acceptance required; capacity review | retained: recipe already uses current model pin |
| 73 | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 2 | `batch-037` | 1 | service | dual-Spark | updated: selected issue 27/55/117/210 fixes verified in pinned image |
| 74 | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 2 | `batch-038` | 1 | service | dual-Spark | updated: selected issue 27/55/210 fixes verified in pinned image |
| 75 | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 2 | `batch-039` | 1 | service | dual-Spark; operator acceptance required | updated: Mamba alignment/state reclamation and tool-choice fixes verified in pinned image |
| 76 | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 2 | `batch-040` | 1 | service | dual-Spark | retained: moving SGLang main is not a release channel |
| 77 | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 2 | `batch-041` | 1 | service | dual-Spark; operator acceptance required | retained: upstream history was rewritten |
| 78 | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 2 | `batch-042` | 1 | service | dual-Spark; operator acceptance required | updated: preparation lock/helper waits bounded with recovery coverage |
| 79 | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 2 | `batch-043` | 1 | service | dual-Spark; operator acceptance required | current |
| 80 | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 2 | `batch-044` | 1 | service | dual-Spark; operator acceptance required | retained: upstream default/profile changed |
| 81 | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 2 | `batch-045` | 1 | service | dual-Spark; operator acceptance required | retained: upstream renamed the target checkpoint |
| 82 | `vonk-forge/glm-5-3-flash-nvfp4-vllm-four` | 4 | — | — | no fixture | out of scope (>2 Sparks) | retained: upstream default changed model identity |
| 83 | `vonk-forge/glm-5-2-quanttrio-vllm-four` | 4 | — | — | no fixture | out of scope (>2 Sparks) | current |
| 84 | `vonk-forge/inkling-975b-a41b-nvfp4-sglang-eight` | 8 | — | — | no fixture | out of scope (>2 Sparks) | retained: moving SGLang main is not a release channel |
| 85 | `vonk-forge/glm-5-2-aqlm-vllm-triple` | 3 | — | — | no fixture | out of scope (>2 Sparks) | current |
<!-- generated:end qualification-inventory -->

## Evidence and stop rules

A recipe passes only with the exact structural record, published package identity,
Controller preparation/apply operation IDs, per-node transfer/start receipts,
declared serving/job assertions, and cleanup result. A repository test or healthy
container is not physical acceptance. A blocked recipe remains in its assigned
batch with its named blocker; it is never omitted to make the campaign green.

Stop the campaign on an integrity mismatch, malformed contract, denied authority,
unexpected model/image substitution, or a failure that could affect another recipe.
A local capacity shortfall blocks only that row. Preserve completed downloads,
verified images, and exact failure evidence so the row can resume after repair.
