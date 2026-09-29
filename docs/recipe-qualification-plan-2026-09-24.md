# Recipe qualification plan — 2026-09-24

This plan covers every recipe in the 85-recipe catalog after the 2026-09-23
source audit. The audited source snapshot includes the MOSS-VL-Realtime refresh
and metadata correction, the target-only SparkInfer XGrammar fix, the bounded
dual-Qwen preparation path, and selected DeepSeek/GLM dual-runtime backports.
It deliberately keeps source review, repository validation, publication,
Controller deployment, and physical Spark acceptance as separate evidence gates.

## Operating principles (owner direction, 2026-09-28)

These override older wording elsewhere in this plan:

- **Requests lead; the fleet converges.** The platform must be fault tolerant,
  self-healing, robust and eventually consistent. A failure is retried with
  backoff, recovered automatically or reported as a visible degraded state with
  the next automatic step; it must not end in a terminal state or an operator
  wait when the system could converge on its own.
- **No blockers.** A local problem (one recipe, one lane, one stale record)
  stays local and must not stop unrelated work. Only real security boundaries
  (authentication, mTLS identity, signature/digest verification of downloaded
  artifacts) may refuse work.
- **Audit trails are not important.** A recipe is qualified when it loads,
  serves or produces valid output for its declared checks, and recovers from
  the declared fault. Operation IDs, receipts and ledgers are useful for
  debugging, not acceptance criteria; missing bookkeeping never blocks a batch.
- **Standing authorization.** The owner authorized proceeding without waiting
  for per-step approval, including fault injection (host restart, rank loss)
  on the two Sparks, on 2026-09-28.

## Execution checkpoint — 2026-09-28

Platform [PR #924](https://github.com/CarstVaartjes/vonk-forge/pull/924) merged
as `8763b517d` (resilience and usability): 30-day agent certificates (renewal at
~day 20, so a Controller outage of up to ~10 days needs no re-enrollment),
worker route-lease renewal independent of image preparation with helper
timeouts and a worker watchdog, agent systemd watchdog and in-process startup
retry, bounded single-Spark crash recovery with automatic resume, a drift check
for running profiles, a 15-minute final-verify bound that hands the run to
recovery, verified backups, LAN-only lab install, a web enrollment form and
`vonkctl run`. Development images and installer publication succeeded. It is
not yet running on the NAS: its Controller refuses to start on a database whose
schema differs from the models, so a Controller schema self-healing change is in
progress before that deployment (no database wipe or re-enrollment).

Profile-1 application `36d22487-a750-468c-944b-92d9a3e81419` **failed** at
2026-09-27 20:32 UTC in `final-verify` ("Run/Switch child returned waiting";
route publication never happened). Root cause: on both Sparks the deployed agent
reports `exact recipe observation failed … (metadata)` every minute because one
historical run directory (`e85c4710…`, already known to fail current parsing)
makes the fail-closed observation report empty for all runs. The Controller
therefore never sees the GLM ranks running and its recovery stops and restarts
them every ~4 minutes. Fix in progress: per-run observation isolation plus
automatic cleanup of stale run directories in the agent. Stale Controller-build
runtime-image receipts lacking an adapter are also logged every scan; they are
skipped and will be cleaned automatically.

Revised order (see *Implementation and execution order*): deploy the agent and
Controller fixes, then reload profile 1 and in parallel start the single-Spark
job batches, beginning with Step1X-3D geometry (batch-001). Job batches do not
depend on the GLM dual-Spark gate.

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
  the original operation and completed lane results instead of duplicating
  loads or rerunning already-proven work unnecessarily.

### Standing full-fleet profile target (not shipped)

This target describes persistent full-fleet desired state, not the campaign
runner's paired-batch barrier above. That barrier remains a qualification policy:
preserve both lane results and reconcile cleanup before starting the next pair.

Under the standing-profile target, a newly added Spark joins the current roster
implicitly idle, while an independently removed Spark is ignored. If a
multi-Spark model loses a roster member, withdraw that model as incomplete and
stop its reachable remaining ranks; report a model-specific error and continue
unrelated work. Do not try to stop the missing node or rewrite the saved profile.
Keep the removed node's historical claim out of live-fleet capacity accounting
and reconcile a same-ID rejoin before placing work on that node.

Every particular preview and load must revalidate the current roster. Continuing
an ongoing standing selection requires one durable singleton selected-profile
record pointing at the immutable accepted application snapshot; a saved profile
or earlier preview is not that record.
This is target behavior, not shipped behavior. Implementation and physical
evidence remain pending.

### Evidence reuse and recovery coverage

Earlier working GLM-5.3 runs remain physical evidence for the recipe, model
and image they ran; this campaign must not describe them as never tested.

Every recipe needs an inference result for all its declared smoke cases and
fixtures. Shared build checks can be reused only for identical artifacts.
Recovery coverage definitions bind the exact runtime stack, topology and member
identities: one representative recovery run covers every member of a shared
group. This catalog has no shared recovery group, so each recipe is recovered
on its own.

### Implementation and execution order

- [x] Record family-aware coverage and two single-Spark lanes in this plan.
- [x] Derive the complete 85-row stack/family matrix, representative selection,
  paired batches and recovery-coverage mapping from exact catalog identities.
  Keep all 81 two-Spark-fleet candidates visible; the four larger-topology
  recipes remain explicitly outside available physical capacity. The derived
  matrix is `family-aware-coverage-2026-09-24.md`, generated by
  `tools/build-family-aware-coverage` and published with each signed release
  (bound to the catalog the authority was generated from). The
  typed recovery definitions are explicit but no group is shared in this
  release; representative selection is a build-cost choice, not a readiness,
  cache or physical claim.
- [ ] Complete the current profile-1 application and verify its declared serving
  and inference checks. At 2026-09-27 19:03:43 UTC application
  `36d22487-a750-468c-944b-92d9a3e81419` was still in `target-copy`, with
  360.9 GB transferred of approximately 398.2 GB planned. Its reviewed plan
  digest starts `113a6635`, and its request key is
  `07ad74ac-2cfe-43ce-bac1-d9fbe0412cec`; reconnect to this operation instead
  of duplicating it. No inference result is recorded.
- [ ] Deploy the per-run agent observation fix (fleet agent upgrade) and the
  Controller schema self-healing release, then the #924 generation, preserving
  the database and enrollments. Reload profile 1 and verify serving and
  inference.
- [ ] In parallel with the GLM gate, run single-Spark job batches starting with
  batch-001 (Step1X-3D geometry and label-geometry: cache ready as of
  2026-09-25; re-check after the database reset) to prove the job path
  end to end on hardware: profile load, artifact-job create/upload/submit,
  agent JobRun, GLB download and validation. Then batch-003/002/004 as their
  caches and provider gates clear.
- [ ] Fault injection is authorized (see *Operating principles*); run it with
  the authorized mechanism. A physical single-Spark reboot fault remains
  unproved. Exercise recovery after a recoverable fault clears. Do not claim
  physical recovery from serving, inference, or metadata alone.
- [ ] Independently of profile 1, add the official NAF checkpoint as a
  canonical Pixal3D Model selection, use the next recipe revision, regenerate
  and validate package/catalog/qualification outputs, publish the catalog, and
  refresh the reviewed authority. Keep NAF cache preparation, upstream asset
  verification, runtime use, and license disposition as separate gates.
- [ ] After the provider gates, resume the full physical campaign across all
  81 one- and two-Spark recipes, using paired single-Spark lanes and exclusive
  dual-Spark work. Keep the four wider topologies out of scope. Reuse completed
  cache work only for its exact bound artifact identity.
- [ ] Produce the final per-recipe report after physical execution, separating
  source freshness, structural validation, timestamped cache evidence,
  inference and recovery state. The 2026-09-25 evidence-inventory snapshot was
  removed because it went stale; build the report from the live sources
  instead: the recipes and models on `main`, the qualification authority and
  campaign under `qualification/`, the signed release assets, and the
  campaign's result log.

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
- **TRELLIS 2 and Pixal3D DINOv3:** the pinned Meta source at
  `ea8dc2863c51be0a264bab82070e3e8836b02d51` requires provider approval and
  authenticated access. Pixal3D's current aggregate Hugging Face snapshot is a
  separate unresolved source path; do not substitute a public mirror.
- **Pixal3D NAF checkpoint:** the [official NAF release](https://github.com/valeoai/NAF/releases/tag/model)
  confirms release `264676230`, asset `320107386`, file `naf_release.pth`,
  2,664,431 bytes, SHA-256
  `c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f`. Recipe
  main `f16d092a8644f3604fc8c7f9fca42088783f3774` now has the typed GitHub
  release-source contract, but still has no canonical NAF Model document or
  Pixal3D selection. PR #914's provider-support source and signed publication
  are accepted at generation
  `e2ee090625a9101425e89aa02b6fecc20b51dcb5b36cd4473edd452ab5dfcfb1`; its
  deployment is confirmed as recorded above. Provider resolution,
  package/catalog closure, cache preparation, and runtime use remain
  unverified.
  The accepted authority still binds Pixal3D 2.0.9, content digest
  `286221bcbfaf2419497cc6eff661b1493281bf8388d789671b171bcc5419d26b`, and
  package digest `bad9f8a39e2618bda305a10d82badc32c1ae5f27f6c57f2fb9044aa608851c8e`.
  A NAF selection requires the next recipe revision, a rebuilt package, and
  regenerated catalog and qualification artifacts followed by a reviewed
  authority refresh. The repository LICENSE is Apache 2.0 text with a blank
  copyright placeholder, and no checkpoint-specific terms were found. License
  disposition remains uncertain; apply the operator decision required by the
  canonical Model/authority contract, without inferring an extra acceptance
  requirement from the placeholder or claiming license clearance.

### Current batch execution gates

The paired runner and authority baseline are accepted and deployed. PR #916's
exact-Stop, selected whole-fleet, degraded-recovery, singleton reboot, startup-
retry, and final-verify/logging source is merged and required CI is complete.
Its source is included in the currently deployed signed #914 generation
`e2ee090625a9101425e89aa02b6fecc20b51dcb5b36cd4473edd452ab5dfcfb1` with
manifest source `0ccd225620c1ed2c1e1660f355274010b9188fbe`. The physical reboot
and fault-to-recovery path remain unproved; the pending explicit fault-injection
authorization remains required.

The old profile-1 application shown at 2026-09-27 16:46 UTC in the historical
snapshot was lost when the full logical Controller database was recreated; do
not reconcile or report that old request as complete. Current profile 1 revision
2 has the exact two-Spark GLM assignment. At 19:00 UTC its preview was allowed
with zero blockers and the exact image was cache-ready. The current application
`36d22487-a750-468c-944b-92d9a3e81419` was still in `target-copy` at 19:03:43
UTC with 360.9 GB of approximately 398.2 GB planned complete. Re-read the same
operation, then prove the workload loads, publishes its endpoint, and passes
declared serving/inference checks. Only after that live profile/inference gate
should fault injection and physical recovery proceed. After profile-1 physical
recovery, begin the NAF Model/recipe/catalog and authority refresh. Recheck
signed release, catalog/authority match, deployment, and Fleet state before
each batch; earlier observations are not current admission evidence.

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
   but does not load the profile or start workloads. Use IDs from the current
   active Controller Fleet inventory; “fresh node IDs” means verified current
   Controller IDs, not newly assigned identities. If a foreign workload would
   be stopped, acknowledge only its exact current run with
   `--replace-run-id RUN_ID`.
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
   verify recovered serving and smoke, then explicitly stop the dual workload
   before sequentially restarting the two idle hosts and recording both changed
   boot IDs. Single-Spark recovery retains its exact active workload through
   the restart; the dual ladder separately proves active rank-loss recovery.
   The deployed-DB decision and reset are recorded above. Before a physical
   host reboot, re-verify the exact signed Controller/agent source and artifact
   identities currently deployed, complete the profile inference gate, and
   obtain explicit fault-injection authorization. Required source/CI completion
   and deployment do not satisfy this physical gate. Agent certificates last 30 days
   and renew at about day 20; re-enrollment is only needed after roughly ten
   days offline. If it is needed, preserve the existing Controller node ID and
   use a fresh enrollment grant; never bypass trust or mint a replacement
   identity. These physical actions remain operator-run;
   `--observe` only records and reconciles their evidence.
7. Retain verified model files, recipe images and compatible partial-transfer
   checkpoints. Preview cleanup with `--cleanup-lane LANE`, then apply its
   reviewed digest with `--apply --cleanup-lane LANE --campaign-digest DIGEST`.
   After all required recovery checkpoints, that explicit cleanup apply records
   the terminal lane result and releases the batch; observation alone does not
   finalize it. Reconcile cleanup and the completion of both lane results
   before advancing to the next batch. A failure local to one lane stays local
   in the evidence; integrity, authority, changed-plan and whole-Fleet ownership
   failures stop the batch.

The current authority is
`qualification/authorities/nl-family-aware-20260924.json`. It binds the
85-recipe catalog generated from its recorded source commit for the contract
release `v2.0.0` (`tools/build-catalog-index`, then
`tools/build-qualification-authority`). It assigns 72 single-Spark recipes to
36 paired batches and schedules nine dual-Spark recipes exclusively. It
contains 90 recipe-specific recovery definitions; no recovery group is shared.
The four recipes requiring more than two Sparks remain explicit exclusions.
Capacity review is the only row-level gate; license terms and territorial
notices are informational. Coverage hashes the current packages and reports
stack divergences from the authority until the authority is regenerated.

## Complete inventory and current batch assignments

Keep this inventory complete. Family grouping and paired batches require a
regenerated reviewed authority, not hand-edited assignments or skipped ledger
checkpoints. The table below shows the current authority batch assignments.

<!-- generated:begin qualification-inventory -->
The authority assigns 75 one-Spark recipes to 38 batches and 12 two-Spark recipes to exclusive batches. Wider topologies close the catalog audit only.

| # | Recipe | Nodes | Batch | Lane | Check | Campaign gate | Source review |
|---:|---|---:|---|---:|---|---|---|
| 1 | `vonk-forge/step1x-3d-geometry-pytorch-single` | 1 | `batch-001` | 1 | job / 3600s | single-Spark | current |
| 2 | `vonk-forge/step1x-3d-label-geometry-pytorch-single` | 1 | `batch-001` | 2 | job / 3600s | single-Spark | current |
| 3 | `vonk-forge/flux-2-klein-4b-nvfp4-comfyui-single` | 1 | `batch-002` | 1 | job / 3600s | single-Spark | current |
| 4 | `vonk-forge/skintokens-pytorch-single` | 1 | `batch-002` | 2 | job / 3600s | single-Spark | current |
| 5 | `vonk-forge/trellis-2-4b-pytorch-single` | 1 | `batch-003` | 1 | job / 3600s | single-Spark | current |
| 6 | `vonk-forge/triposg-pytorch-single` | 1 | `batch-003` | 2 | job / 3600s | single-Spark | current |
| 7 | `vonk-forge/flux-2-klein-4b-comfyui-single` | 1 | `batch-004` | 1 | job / 3600s | single-Spark | current |
| 8 | `vonk-forge/pixal3d-pytorch-single` | 1 | `batch-004` | 2 | job / 3600s | single-Spark | retained: upstream added a distinct multiview checkpoint |
| 9 | `vonk-forge/ornith-1-5-35b-a3b-nvfp4-vllm-single` | 1 | `batch-005` | 1 | service | single-Spark | current |
| 10 | `vonk-forge/lfm2-5-vl-3b-vllm-single` | 1 | `batch-005` | 2 | service | single-Spark | current |
| 11 | `vonk-forge/lfm2-5-vl-3b-vllm028-single` | 1 | `batch-006` | 1 | service | single-Spark | current |
| 12 | `vonk-forge/qwen3-5-9b-vllm-single` | 1 | `batch-006` | 2 | service | single-Spark | current |
| 13 | `vonk-forge/qwen3-6-35b-a3b-nvfp4-vllm-single` | 1 | `batch-007` | 1 | service | single-Spark | retained: recipe already uses current model pin |
| 14 | `vonk-forge/qwen3-8-27b-fp8-vllm-single` | 1 | `batch-007` | 2 | service | single-Spark | current |
| 15 | `vonk-forge/laguna-xs-2-1-nvfp4-vllm-single` | 1 | `batch-008` | 1 | service | single-Spark | current |
| 16 | `vonk-forge/wan-2-2-ti2v-5b-comfyui-single` | 1 | `batch-008` | 2 | job / 3600s | single-Spark | retained: README-only change |
| 17 | `vonk-forge/moss-vl-realtime-11b-pytorch-single` | 1 | `batch-009` | 1 | job / 1800s | single-Spark | updated to d1f71a58; transcript/NOTICE identities corrected |
| 18 | `vonk-forge/ltx-2-19b-dev-bf16-diffusers-single` | 1 | `batch-009` | 2 | job / 3600s | single-Spark | current |
| 19 | `vonk-forge/ltx-2-19b-dev-fp4-pytorch-single` | 1 | `batch-010` | 1 | job / 3600s | single-Spark | current |
| 20 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-single` | 1 | `batch-010` | 2 | service | single-Spark | retained: README-only model change |
| 21 | `vonk-forge/nemotron-3-nano-30b-a3b-vllm-single` | 1 | `batch-011` | 1 | service | single-Spark | retained: README-only model change |
| 22 | `vonk-forge/nemotron-3-nano-omni-30b-a3b-vllm-single` | 1 | `batch-011` | 2 | service | single-Spark | current |
| 23 | `vonk-forge/qwen3-8-27b-nvfp4-dspark-sglang-single` | 1 | `batch-012` | 1 | service | single-Spark | retained: upstream change is DFlash-only |
| 24 | `vonk-forge/ltx-2-3-22b-distilled-1-1-diffusers-single` | 1 | `batch-012` | 2 | job / 3600s | single-Spark | retained: README-only change |
| 25 | `vonk-forge/qwen3-6-27b-vllm-single` | 1 | `batch-013` | 1 | service | single-Spark | current |
| 26 | `vonk-forge/ltx-2-19b-distilled-fp8-diffusers-single` | 1 | `batch-013` | 2 | job / 3600s | single-Spark | current |
| 27 | `vonk-forge/step1x-3d-texture-pytorch-single` | 1 | `batch-014` | 1 | job / 3600s | single-Spark | current |
| 28 | `vonk-forge/gemma-4-26b-a4b-vllm-single` | 1 | `batch-014` | 2 | service | single-Spark | current |
| 29 | `vonk-forge/gemma-4-26b-a4b-vllm028-single` | 1 | `batch-015` | 1 | service | single-Spark | current |
| 30 | `vonk-forge/qwen-image-2512-fp8-lightning-comfyui-single` | 1 | `batch-015` | 2 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 31 | `vonk-forge/qwen-image-edit-2511-fp8mixed-comfyui-single` | 1 | `batch-016` | 1 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 32 | `vonk-forge/qwen-image-edit-2511-int8-convrot-comfyui-single` | 1 | `batch-016` | 2 | job / 3600s | single-Spark | retained: selected model bytes unchanged |
| 33 | `vonk-forge/ui-mate-27b-vllm-single` | 1 | `batch-017` | 1 | service | single-Spark | retained: only benchmark/demo material changed |
| 34 | `vonk-forge/muse-glimmer-30b-bf16-vllm-single` | 1 | `batch-017` | 2 | service | single-Spark | current |
| 35 | `vonk-forge/qwen3-8-27b-vllm-single` | 1 | `batch-018` | 1 | service | single-Spark | current |
| 36 | `vonk-forge/ltx-2-5-22b-distilled-fp8-cast-diffusers-single` | 1 | `batch-018` | 2 | job / 3600s | single-Spark | current |
| 37 | `vonk-forge/mova-360p-diffusers-single` | 1 | `batch-019` | 1 | job / 3600s | single-Spark | current |
| 38 | `vonk-forge/deepseek-v4-flash-0731-ds4-dspark-latency-single` | 1 | `batch-019` | 2 | service | single-Spark | retained: upstream now spans other model/runtime paths |
| 39 | `vonk-forge/deepseek-v4-flash-0731-ds4-single` | 1 | `batch-020` | 1 | service | single-Spark | retained: upstream now spans other model/runtime paths |
| 40 | `vonk-forge/ltx-2-19b-distilled-diffusers-single` | 1 | `batch-020` | 2 | job / 3600s | single-Spark | current |
| 41 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-low-memory-canary-single` | 1 | `batch-021` | 1 | service | single-Spark | retained: recipe already uses current model pin |
| 42 | `vonk-forge/mova-720p-diffusers-single` | 1 | `batch-021` | 2 | job / 3600s | single-Spark | current |
| 43 | `vonk-forge/nemotron-3-5-lightning-dspark-lowmem-canary-single` | 1 | `batch-022` | 1 | service | single-Spark | retained: README-only model change |
| 44 | `vonk-forge/nemotron-3-super-120b-a12b-vllm-single` | 1 | `batch-022` | 2 | service | single-Spark | current |
| 45 | `vonk-forge/nvidia-qwen-image-flash-diffusers-single` | 1 | `batch-023` | 1 | job / 3600s | single-Spark | current |
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
| 57 | `vonk-forge/ling-3-0-flash-dspark-sglang-single` | 1 | `batch-029` | 1 | service | single-Spark | current |
| 58 | `vonk-forge/deepseek-v4-flash-0731-mia-sparkinfer-single` | 1 | `batch-029` | 2 | service | single-Spark | current |
| 59 | `vonk-forge/hunyuan-video-foley-xl-pytorch-single` | 1 | `batch-030` | 1 | job / 3600s | single-Spark | current |
| 60 | `vonk-forge/hunyuan-video-foley-xxl-pytorch-single` | 1 | `batch-030` | 2 | job / 3600s | single-Spark | current |
| 61 | `vonk-forge/hunyuanocr-1-5-vllm-dflash-single` | 1 | `batch-031` | 1 | job / 3600s | single-Spark | current |
| 62 | `vonk-forge/hunyuan3d-omni-pytorch-single` | 1 | `batch-031` | 2 | job / 3600s | single-Spark | current |
| 63 | `vonk-forge/hunyuan-video-15-distilled-diffusers-single` | 1 | `batch-032` | 1 | job / 3600s | single-Spark | current |
| 64 | `vonk-forge/hunyuan-video-15-i2v-step-distilled-diffusers-single` | 1 | `batch-032` | 2 | job / 3600s | single-Spark | current |
| 65 | `vonk-forge/hunyuan-video-15-t2v-diffusers-single` | 1 | `batch-033` | 1 | job / 3600s | single-Spark | current |
| 66 | `vonk-forge/minimax-h3-diffusers-single` | 1 | `batch-033` | 2 | job / 3600s | single-Spark | current |
| 67 | `vonk-forge/minimax-h3-fl2va-diffusers-single` | 1 | `batch-034` | 1 | job / 3600s | single-Spark | current |
| 68 | `vonk-forge/ltx-2-5-22b-distilled-bf16-diffusers-single` | 1 | `batch-034` | 2 | job / 3600s | single-Spark; capacity review | current |
| 69 | `vonk-forge/nemotron-3-5-lightning-30b-a3b-vllm-dspark-latency-single` | 1 | `batch-035` | 1 | service | single-Spark; capacity review | retained: README-only model change |
| 70 | `vonk-forge/wan-dancer-14b-pytorch-single` | 1 | `batch-035` | 2 | job / 3600s | single-Spark; capacity review | current |
| 71 | `vonk-forge/deepseek-v4-flash-0731-sparkinfer-single` | 1 | `batch-036` | 1 | service | single-Spark; capacity review | retained: fixed public image not republished |
| 72 | `vonk-forge/laguna-s-2-1-nvfp4-vllm-single` | 1 | `batch-036` | 2 | service | single-Spark; capacity review | retained: recipe already uses current model pin |
| 73 | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-single` | 1 | `batch-037` | 1 | service | single-Spark | new: Mia single-Spark TP1 kit 7d0712dc |
| 74 | `vonk-forge/glm-5-3-flash-exl3-dflash2-r0b0tlab-vllm-single` | 1 | `batch-037` | 2 | service | single-Spark | new: r0b0tlab 2dfbdbe4 vLLM single-Spark EXL3/DFlash2 profile (draft CC BY-NC-ND) |
| 75 | `vonk-forge/qwen3-8-flash-next-tensorfold-single` | 1 | `batch-038` | 1 | service | single-Spark | new: Mia TensorFold single-Spark kit 856bb6be on TensorFold v0.3.7 |
| 76 | `vonk-forge/deepseek-v4-flash-0731-mia-dual` | 2 | `batch-039` | 1 | service | dual-Spark | updated: selected issue 27/55/117/210 fixes verified in pinned image |
| 77 | `vonk-forge/deepseek-v4-flash-vision-exp-mia-dual` | 2 | `batch-040` | 1 | service | dual-Spark | updated: selected issue 27/55/210 fixes verified in pinned image |
| 78 | `vonk-forge/glm-5-3-flash-exl3-dflash2-vllm-dual` | 2 | `batch-041` | 1 | service | dual-Spark | updated: Mamba alignment/state reclamation and tool-choice fixes verified in pinned image |
| 79 | `vonk-forge/inkling-small-nvfp4-sglang-dual` | 2 | `batch-042` | 1 | service | dual-Spark | retained: moving SGLang main is not a release channel |
| 80 | `vonk-forge/qwen3-8-flash-next-nvfp4-sglang-dual` | 2 | `batch-043` | 1 | service | dual-Spark | retained: upstream history was rewritten |
| 81 | `vonk-forge/qwen3-8-flash-next-nvfp4-vllm-dual` | 2 | `batch-044` | 1 | service | dual-Spark | updated: preparation lock/helper waits bounded with recovery coverage |
| 82 | `vonk-forge/glm-5-3-flash-nvfp4-vllm-dual` | 2 | `batch-045` | 1 | service | dual-Spark | current |
| 83 | `vonk-forge/glm-5-3-flash-nvfp4-ablit-l15-43-dflash2-vllm-dual` | 2 | `batch-046` | 1 | service | dual-Spark | retained: upstream default/profile changed |
| 84 | `vonk-forge/glm-5-3-flash-nvfp4-kv-1m-abliterated-vllm-dual` | 2 | `batch-047` | 1 | service | dual-Spark | retained: upstream renamed the target checkpoint |
| 85 | `vonk-forge/glm-5-3-flash-nvidia-nvfp4-dflash2-vllm-dual` | 2 | `batch-048` | 1 | service | dual-Spark | new: r0b0tlab 22269731 on nvidia/GLM-5.3-Flash-NVFP4 |
| 86 | `vonk-forge/deepseek-v4-1-flash-exl3-mia-dual` | 2 | `batch-049` | 1 | service | dual-Spark | new: Mia 6f7d1590 EXL3 2.9 bpw |
| 87 | `vonk-forge/mimo-v2-6-flash-rl-vllm-dual` | 2 | `batch-050` | 1 | service | dual-Spark | new: tonyd2wild 13621bb3 vLLM TP2 DFlash |
| 88 | `vonk-forge/glm-5-3-flash-nvfp4-vllm-four` | 4 | — | — | no fixture | out of scope (>2 Sparks) | retained: upstream default changed model identity |
| 89 | `vonk-forge/glm-5-2-quanttrio-vllm-four` | 4 | — | — | no fixture | out of scope (>2 Sparks) | current |
| 90 | `vonk-forge/inkling-975b-a41b-nvfp4-sglang-eight` | 8 | — | — | no fixture | out of scope (>2 Sparks) | retained: moving SGLang main is not a release channel |
| 91 | `vonk-forge/glm-5-2-aqlm-vllm-triple` | 3 | — | — | no fixture | out of scope (>2 Sparks) | current |
<!-- generated:end qualification-inventory -->

## Evidence and stop rules

A recipe passes when it loads on its declared topology, passes its declared
serving/job assertions (valid output, not merely a healthy container) and
recovers from its declared fault. Operation IDs and receipts are recorded when
available for debugging; missing bookkeeping does not fail a recipe. A blocked recipe remains in its assigned
batch with its named blocker; it is never omitted to make the campaign green.

Stop the campaign only on a real integrity or security failure (digest or
signature mismatch, unexpected model/image substitution). Everything else is
retried, recovered or recorded as that recipe's blocker while the campaign
continues.
A local capacity shortfall blocks only that row. Preserve completed downloads,
verified images, and exact failure evidence so the row can resume after repair.
