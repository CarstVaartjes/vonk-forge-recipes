# Recipe qualification plan — 2026-09-24

This plan covers every recipe in the 85-recipe catalog after the 2026-09-23
source audit. The audited source snapshot includes the MOSS-VL-Realtime refresh
and metadata correction, the target-only SparkInfer XGrammar fix, the bounded
dual-Qwen preparation path, and selected DeepSeek/GLM dual-runtime backports.
It deliberately keeps source review, repository validation, publication,
Controller deployment, and physical Spark acceptance as separate evidence gates.

## Execution checkpoint — 2026-09-27

Platform [PR #913](https://github.com/CarstVaartjes/vonk-forge/pull/913) is
merged and deployed. Merge `03b2d6cddbf2a13c922ed15c5e65a0322766afed`, final head
`e326429f7f884167f43cd04ac192be9ed9177da1`, and required CI run `36275927334`
succeeded. The affected real-PostgreSQL profile suite passed 272 tests, with
zero failures and zero skips; the final typed pair/same-key tests passed 2/2.
Development-image run `36276218398` and installer-acceptance run
`36276601591` accepted
generation `a9fc4cf592dcf67f2ea0b54b1a8b1f4353835f8c135a70ca77609009f7a97629`
for that source. Merged recovery behavior covers crash resume, reviewed-digest
preservation, late-outcome fencing and newer-intent supersession, bounded
contention retries, pre-effect authority rechecks, fair due scheduling, and
visible startup constraint drift.

Deployment verification found 11 healthy services; only API and worker
containers were recreated. The `.env`, `secrets/`, and
named-volume identities match before and after. Captured Controller image
digest: `sha256:25b83b51cb4b81f3340ba993a820c1a98f936342ecac8115e2863dbb82a27012`.

Fresh structural, catalog, authority, and coverage checks passed for all 85
recipes on platform `03b2d6cddbf2a13c922ed15c5e65a0322766afed` and recipe
checkout `303eaf35e0e7949eb858d9e50b35aa3a4d7f2dc0`: 81 recipes require one or
two Sparks; four wider topologies remain outside physical scope.

The fresh profile 1 review for Mia GLM 5.3 allowed plan digest
`361cf7d673838a2323ed74eb5cfbfa7d68743344a03fb2c01ab89567987988f4`;
application `3e865d60-acd6-405f-b2e5-7891772668d9`, request
`17f0e0fb-de96-4a8b-88c4-817dff7ea6e3`, ordinal 79, was queued at
`2026-09-26T22:45:12.931660Z`. A detailed read at `2026-09-27T04:01:09Z`
observed the application running in `final-verify` with endpoint
`not-published-yet`; its persisted start deadline
`2026-09-27T00:21:28.173639Z` had expired. Root reported a later read-only
observation at `2026-09-27T08:22:00Z`; its operation details are not recorded
here. At 2026-09-27 11:32 UTC, read-only evidence showed profile 1 overdue;
recheck before reconciliation. The source fixes below are separate and do not
reconcile or extend it. Re-read the durable operation and reconcile exact
effects; do not extend its deadline. Load, serving, inference, and physical
recovery remain unproven in the evidence recorded here.

Earlier read-only checks on both Sparks found current run
`1f444de3-99aa-41b1-a417-f61409fdc7ca` metadata and installed spec/runtime valid;
historical run `e85c4710-e437-4d12-8191-499596aa2a4c` failed current parsing at
`/` and `/runtime/placement`. These dated checks prove metadata compatibility
only, not inference or recovery.

As of 2026-09-27, the main recovery integration tree is active, with narrow
profile-owned JobRun Stop work in progress. Degraded multi-group and
selected-profile workers are continuing in separate trees. The documentation
target is not shipped. This checkpoint claims neither green integrated tests
nor physical acceptance. Luna Max owns implementation; root owns design
decisions and review.

Earlier component-branch evidence is not integration acceptance. The observer
branch's planner-integration regression was red before per-run isolation and
passed after it; its separate current-receipt test was only checked after the
fix. The serialized Linux `vonk-agent` package suite passed, while parallel
package runs hit `ReconciliationBusy` and still need investigation. The
child-progress donor passed four distributed-start cases (98 deselected), local
claim-refusal, and two real-PostgreSQL claim-refusal/progress-projection cases;
pinned Ruff and format passed, types retain one existing reviewed exception,
and the coordination scan found zero reviewed sites. The separate
helper/protocol donor passed 61 protocol, 79 helper, 11 helper-binary, 12 Python
response, and one focused Controller test; generated parity, Ruff, Rustfmt,
and its oversized-invocation red/green regression passed. These results have
not been rerun against the integration branch. The preserved helper/agent
donor's full agent compile still has three hook-work-in-progress errors:
`begin_post_stop_hooks` is missing, `PostStopHookProgress.execution` is missing,
and the `Execute` permit is incomplete. The integration is not release-ready.

Several stop-authority and size proposals remain unapproved and unapplied:
Controller cleanup authority over generation/deadline/node eligibility;
adoption of an exact old recovery Stop by a newer explicit Stop; durable helper
generation closure after Stop, including `Stop(false)`; and replacement of the
16 MiB aggregate parent-payload cap. The size proposal still needs caller
integration and tests, and its automatic review raised a memory-exhaustion
concern. The proposed rollback of per-hook receipts was rejected because it
would weaken crash/restart safeguards; keep those safeguards intact. Preserve
strict `phases: null` handling and use the shared recovery predicate during
integration. Bind the exact reviewed stop effects and plan digest, plus
`run_generation` on Start and Stop; preserve one-shot hook markers and uncertain
post-stop outcomes. Goal continuation does not approve any held proposal.

These are component and integration-worktree observations, not a shipped
end-to-end self-healing path. Recovery must preserve actionable blockers for
revoked authority, integrity failures, changed plans, unavailable resources,
and provider/license gaps.

The dated cache and health observations below remain historical evidence. Reuse
model and recipe-image artifacts by their exact bound identities; a changed
platform source SHA alone does not invalidate those artifacts.

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
  generated by `tools/build-family-aware-coverage` and bound to v1.0.17. The
  typed recovery definitions are explicit but no group is shared in this
  release; representative selection is a build-cost choice, not a readiness,
  cache or physical claim.
- [x] Complete the paired campaign runner source integration for lane evidence,
  exclusive recovery, resumable batches, fleet ownership, stale plans,
  wrong-recipe/node evidence, duplicate apply after disconnect and premature
  lane replacement. Platform PR #893 merged as `4c8cf45540bca1f32c6c9b1b963c68783aa644bb`; required CI run `36089768144` passed. This records source and CI completion, not successful physical recovery.
- [x] Merge the approved recovery identity correction and real-PostgreSQL
  contention fix. Platform PR #894 merged as
  `1b1e10f9847ed185ab1e2f4f006e4c797ca9b5a1`; final source head was
  `e7188a946219684fa2c24b4a8f4a66dce212876d`. Required CI run `36106444427`
  passed. Six focused availability cases included one real-PostgreSQL
  contention/recovery case; the other five covered explicit retry and model-child
  behavior. There were also 26 provenance tests, one receipt-builder test and
  106 earlier identity tests. Type checking reported one existing exception;
  there was no database schema change.
- [x] Accept the signed PR #894 publication. Installer acceptance workflow
  `36107593799` accepted generation
  `bad7e9e7d903eac0db3acb41ad21d444f61283a2f696c66cb4f01a4d0c6f4abf` for
  source `1b1e10f9847ed185ab1e2f4f006e4c797ca9b5a1`; development-image workflow
  `36106900488` also succeeded.
- [x] Complete the PR #894 NAS deployment and verify the Controller/CLI state.
  Deployment verification reports 11 healthy services, nine retained
  containers, two replaced containers (API and worker), preserved volume mounts
  and no added paths. The live bundle hashes match staging without a rewrite;
  the API reports the signed source commit. The installed CLI updated from
  `4c8cf45540bca1f32c6c9b1b963c68783aa644bb` to
  `1b1e10f9847ed185ab1e2f4f006e4c797ca9b5a1`. The staged `.env` and 62 secret
  files are unchanged. No agent rollout was needed because the accepted package
  digest matches the previous package.
- [x] Verify the installed CLI helper consumes fresh Controller provenance
  under the canonical `agent_build_sha256` identity. Both Sparks passed at
  `07:39:16 UTC`; their observed agent build digests match the signed target.
  This verifies the identity boundary only. No physical inference or recovery
  receipt is implied, and the Controller's own publication boundary remains
  unknown.
- [x] Complete the initial signed publication and Controller/CLI deployment
  for platform PR #893, and review the authority bound to recipe release
  v1.0.17. Publication run `36090614107` accepted generation
  `f057a9e59058d7beb7098d721fd21c07fbddd9ca39b1327babd9557e4756095e` for source
  `4c8cf45540bca1f32c6c9b1b963c68783aa644bb`; deployment completed on 2026-09-25
  at 03:45 UTC. This is the prior baseline and does not complete physical
  qualification.
- [x] Complete platform PR #912 publication and deployment. It merged as
  `1b685415695d79dadf4d418a48844bc81bc1e6a5`; accepted generation
  `f8081e89e13c9e52c9e3ea39095c3d55355e4b0098824494a286f9c8f18a47f7` was the
  deployed baseline before PR #913 superseded it. This deployment did not prove
  serving or physical acceptance.
- [x] Merge platform PR #913 and complete its combined source/CI validation;
  merge, final-head, and test evidence are recorded in the execution checkpoint
  above.
- [x] Complete PR #913 signed publication and NAS deployment. Accepted
  generation, health, replacement, preserved-state, and Controller-image
  evidence are recorded in the execution checkpoint above. This completes
  deployment only; load, inference, and physical recovery remain open.
- [x] Commit and integrate observer-isolation and truthful child-progress/
  claim-refusal components on `codex/recovery-integration`. This is unpublished
  source progress only. The main integration tree is active; profile-owned
  JobRun Stop work remains in progress, while degraded multi-group and
  selected-profile work continues in separate trees.
- [ ] Continue the active profile-owned JobRun Stop integration and bring the
  degraded multi-group and selected-profile work through root review. Keep
  Controller cleanup authority, old-Stop dependency adoption, helper generation
  closure after Stop, and replacement of the 16 MiB aggregate parent-payload
  cap as unapproved proposals unless explicitly authorized; do not apply the
  rejected per-hook rollback. Preserve strict `phases: null` handling and the
  shared recovery predicate. Require the Controller to validate the durable exact
  Stop plan and sign its digest, and the helper to reconstruct and enforce the
  complete stop/hook argv. Bind `run_generation` on Start and Stop so a late
  Stop cannot affect a replacement; reject stale authority before effects;
  retain one-shot hook markers and reconcile uncertain outcomes without
  replaying completed hooks.
- [ ] Verify compilation and resolve any remaining hook gaps recorded in the
  earlier helper/agent donor (`begin_post_stop_hooks`,
  `PostStopHookProgress.execution`, and the incomplete `Execute` permit), then
  complete generated parity and integrated producer-to-Controller-to-agent/
  helper-to-receipt regressions. Use real
  PostgreSQL and processes to cover valid current receipts beside malformed
  history, unknown results for incomplete/all-failed inventory, newer-intent supersession,
  delayed old Stops, process death/restart, partial storage, busy slots,
  heartbeat contention, truthful child progress, and refusal of wrong
  node/plan/image/hook/generation or stale authority before effects. Resolve
  the parent-payload bound with its memory-safety and caller/tests review;
  investigate parallel `ReconciliationBusy`. Then run pinned lint, format,
  types, generators, coordination checks, and required CI on the combined tree.
- [ ] Complete the source and deployment prerequisites for physical
  single-Spark host-reboot recovery: host-reboot recovery, agent startup
  resilience, and final-verify expiry with actionable logging remain in
  progress. Pass process and systemd tests for the exact combined build. Do not
  schedule a physical reboot until those tests pass and the matching signed
  Controller/agent release is deployed.
- [ ] After review and CI, once those prerequisites and tests pass, merge the
  integrated recovery release, accept its signed publication, and deploy
  through the Controller-authorized path. Verify
  the signed agent on both Sparks records valid current receipts while the
  malformed historical run remains unknown; metadata and container health do
  not prove serving.
- [ ] Reconcile profile 1 request
  `17f0e0fb-de96-4a8b-88c4-817dff7ea6e3` after the persisted deadline expired.
  Re-read its durable operation and exact effects; do not extend the deadline
  or duplicate an active request. If the operation is terminal, reconcile exact
  effects before a fresh explicitly authorized request. Prove final verification
  completes, the workload loads, its endpoint is published, and Mia GLM 5.3
  passes declared serving/inference checks; that proves inference, not
  recovery.
- [ ] Obtain the pending explicit fault-injection authorization before using
  the authorized mechanism. A physical single-Spark reboot fault remains
  blocked on the signed Controller/agent deployment and passing process and
  systemd tests above. Exercise recovery after a recoverable fault clears and
  capture an attributable exact-identity fault-to-recovery receipt. Do not
  claim physical recovery from serving/inference or metadata alone.
- [ ] After profile 1 physical recovery, complete PR #914's accepted
  publication and deployment. Then add the official NAF checkpoint as a
  canonical Pixal3D Model selection, use the next recipe revision, regenerate
  and validate package/catalog/qualification outputs, publish the catalog, and
  refresh the reviewed authority. Keep NAF cache preparation, upstream asset
  verification, runtime use, and license disposition as separate gates.
- [ ] After the provider gates, resume the full physical campaign across all
  81 one- and two-Spark recipes, using paired single-Spark lanes and exclusive
  dual-Spark work. Keep the four wider topologies out of scope. Reconcile
  historical receipts against current recipe, model, image, topology, and
  platform identities; preserve provider, license, authority, integrity, and
  actual-resource blockers; reuse completed cache work only for its exact
  bound artifact identity.
- [ ] Produce the final per-recipe report after physical execution. The current
  snapshot at `qualification/reports/recipe-evidence-inventory-2026-09-25.md`
  and its JSON companion separate source freshness, structural validation,
  timestamped cache evidence, inference and recovery state; refresh it as the
  campaign progresses.

Use bounded GPT-6 Luna Max agents for independent audits or implementation,
each in its own task-owned worktree. Avoid agents for routine polling, duplicate
audits and repeated full suites on unchanged inputs. Integrate related fixes,
run focused regression checks during iteration and the required combined gates
before release. Refresh campaign authority after relevant artifacts settle,
not after every intermediate failed build.

## Historical execution status snapshot — 2026-09-25

The live cache, Fleet, node, fit, and health observations in this snapshot are
dated evidence from 2026-09-25. Refresh them before making current admission or
readiness decisions.

The accepted recipe release remains v1.0.17
(`efbbba29bd4c706c73d295d24047787be3f36d78`); recipe `main` includes the prior
report update through `506a4af6f9c8c52acf0213aba7f2ab63bbad254f`. The structural
evidence remains bound to recipe checkout
`784bda637a45c9fab6b18fc0cd2faa2669ab9800` and platform PR #893 merge
`4c8cf45540bca1f32c6c9b1b963c68783aa644bb`. Required CI run `36089768144`
passed with no database schema change; structural validation passed all 85 exact
recipe identities at `2026-09-25T03:29:03.058743Z`. Structural success makes no
physical qualification claim.

Platform PR #894 merged as `1b1e10f9847ed185ab1e2f4f006e4c797ca9b5a1` at
07:17:31 UTC; its final source head is `e7188a946219684fa2c24b4a8f4a66dce212876d`.
Required CI run `36106444427` passed. Six focused availability cases included
one real-PostgreSQL contention/recovery case; the other five covered explicit
retry and model-child behavior. There were also 26 provenance tests, one
receipt-builder test and 106 earlier identity tests. Type checking reported one
existing exception, and no database schema changed. Development-image workflow
`36106900488` and installer acceptance workflow `36107593799` succeeded.

The installer accepted signed generation
`bad7e9e7d903eac0db3acb41ad21d444f61283a2f696c66cb4f01a4d0c6f4abf` for source
`1b1e10f9847ed185ab1e2f4f006e4c797ca9b5a1` (release
`0.1.1~dev.611+gf2be83c2f165`). The accepted API digest is
`sha256:c56a3bac9cc625bcff74a3cf9482efcb3b93c128d2f09eede85bbeaa4aead4b7`; all
four image digests are retained in the JSON report. The agent package digest
`0dd85a2fc642fb5143a8c7fc8f04430108286e27c7e579d0035b427c7a168c71` matches the
previous package, so no agent rollout was needed. Staging preserved `.env` and
all 62 secret files with no Compose diff. Deployment verification found 11
healthy services, nine retained containers, two replaced containers (API and
worker), preserved volume mounts and no added paths. API build provenance matches
the accepted source and all staged bundle hashes match the live bundle; the CLI
updated from platform source `4c8cf45540bca1f32c6c9b1b963c68783aa644bb` to
`1b1e10f9847ed185ab1e2f4f006e4c797ca9b5a1`.

The installed site-packages helper passed fresh Controller-provenance validation
at `2026-09-25T07:39:16.008465Z` on Spark 3542 and
`2026-09-25T07:39:16.021975Z` on Spark 2297. Both observations bind Controller
image `sha256:c56a3bac9cc625bcff74a3cf9482efcb3b93c128d2f09eede85bbeaa4aead4b7`
and agent build `f1198ce592e940e17c7bbe017f4c9755df25f74351163851279c4eb95fe713bf`;
optional package receipts are absent on both nodes. The Controller's own
publication boundary remains unknown. This proves the corrected identity-bound
producer/consumer path, not physical inference or recovery; no physical recovery
receipt exists.

The prior signed schema-2 publication run `36090614107` accepted generation
`f057a9e59058d7beb7098d721fd21c07fbddd9ca39b1327babd9557e4756095e` for source
`4c8cf45540bca1f32c6c9b1b963c68783aa644bb`; the NAS deployment and Controller
CLI update completed at 03:45 UTC. The verified API image was
`sha256:519cf084dba79fd79c35f89053bd000f10e4f05d290d769e8ea50dfff5e16344`, with
all 11 NAS services healthy. These facts describe the PR #893 baseline, not the
merged PR #894 follow-up.

The latest complete exact-identity cache assessment finished at
`2026-09-25T07:03:08.790476+00:00`. Its 85 summary observations span
`06:59:39.098475`–`07:03:06.507613 UTC`; all 85 identities matched with zero
mismatches and zero command errors. The scan reports 3 cache-ready and 82
blocked rows. The ready recipes are Step1X geometry, Step1X label geometry and
GLM-5-3 Flash EXL3 DFlash2. Cache state is only a timestamped observation; it
does not establish fit, inference, recovery or availability after the scan.
Earlier Sep 24 counts are retained as historical evidence in the JSON report.

Step1X label geometry's no-force preparation request
`8800c9b1-39f8-4a37-9c65-998cf780919f` has parent operation
`ed53ea64-ae62-490c-9b44-1a787d3d09ab`, recorded as failed and non-retryable
with PostgreSQL `55P03` (`model_cache_operations` row-lock error). Later child
progress showed the runtime-image child (13,611,009,024 bytes) and model-cache
child (8,781,511,993 bytes across 14 items) succeeded, totaling
22,392,521,017 bytes at `06:45:47.300430 UTC`. The full scan found this recipe
identity cache-ready at `06:59:39.098659 UTC` (detail assessment
`06:59:40.790032Z`). A separate fresh no-force request
`86f1b561-3132-4342-b0e0-498fe9459fe7` then succeeded as operation
`93795a0e-3f4e-44b8-b92a-1c152d0732f8` at `07:40:23.428405 UTC`, returning the
same model-cache child and the same build/image identity for 22,392,521,017
completed bytes. Its exact detail at `07:41:14.794466Z` remained cache-ready and
fit only Spark 2297. Keep the original failed parent receipt intact; the later
successful preparation and fit do not establish inference or recovery. Step1X
geometry's earlier successful no-force preparation remains bound to its own
recipe identity.

The physical campaign remains incomplete. Fresh node snapshots at 07:38–07:39
UTC show both Sparks online and ready, with zero loaded workloads, five installed
records each, and disk reservations of 1,320,491,003,815 bytes on each node. On
Spark 3542, five installed plans still fail canonical parsing because required
`memory_floor_bytes` and `memory_kind` placement fields are absent; no capacity
was discounted. No supported repair or per-installation uninstall assessment
was available. LTX 2.5 per-file
hashes, gated Meta DINOv3 access for TRELLIS 2/Pixal3D, and Pixal3D's unsupported
GitHub-release NAF cache path remain provider/source blockers from the plan.

The PR #893 deployment snapshot showed Spark 3542 already running the requested
agent build and the Controller refusing another upgrade with HTTP 409. Authenticated
binary/build digests matched the then-signed target; the optional agent package
receipt remained an evidence gap, not a proven campaign blocker. No current-
identity inference or physical recovery receipt has been recorded. The historical
GLM 1.6.6 results remain valid only for their exact prior recipe, image and run
identity. The 85-row report remains a progress snapshot, not the final
post-campaign report.

After the operator's subsequent Docker update, the Fleet read at
`2026-09-25T07:51:08.014673Z` showed both Sparks online with no loaded workloads.
The NAS check found all 11 services healthy, accepted Vonk image identities
unchanged, and prior volume mounts preserved. Docker reported 29.6.2. This later
health snapshot does not change the physical qualification or admission gaps.

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
  Pixal3D selection. PR #914's provider-support path still requires accepted
  publication and deployment; provider resolution, package/catalog closure,
  cache preparation, and runtime use are unverified.
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

The paired runner and authority baseline are accepted and deployed. Observer
and child-progress components are committed in the unpublished recovery
integration. The main integration tree is active with profile-owned JobRun Stop
work in progress; degraded multi-group and selected-profile work continues in
separate trees. Integrated regression, release, and physical acceptance remain
gates; this checkpoint records no green integrated test or physical-acceptance
result. Host-reboot recovery, agent startup resilience, and final-verify
expiry/logging remain source and deployment prerequisites. Do not schedule a
physical reboot until the signed agent and matching Controller are deployed
and process/systemd tests pass. At 2026-09-27 11:32 UTC, read-only evidence
showed profile 1 overdue; recheck before reconciliation, separately from those
source fixes. Complete and release the recovery path, then prove profile 1
physical recovery before beginning the provider sequence: PR
#914 accepted publication/deployment, followed by the NAF Model/recipe/catalog
and authority refresh. Recheck signed release, catalog/authority match,
deployment, and Fleet state before each batch; earlier observations are not
current admission evidence.

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
   Do not schedule any physical host reboot until the matching signed agent
   and Controller are deployed and the required process/systemd tests pass. If
   certificate re-enrollment is needed after more than 24 hours offline,
   preserve the existing Controller node ID and campaign ledger and require an
   explicit authorized grant; never bypass trust automatically or mint a
   replacement identity. These physical actions remain operator-run;
   `--observe` only records and reconciles their evidence.
7. Retain verified model files, recipe images and compatible partial-transfer
   checkpoints. Preview cleanup with `--cleanup-lane LANE`, then apply its
   reviewed digest with `--apply --cleanup-lane LANE --campaign-digest DIGEST`.
   After all required recovery checkpoints, that explicit cleanup apply records
   the terminal lane result and releases the batch; observation alone does not
   finalize it. Reconcile cleanup and the completion of both lane receipts
   before advancing to the next batch. A failure local to one lane stays local
   in the evidence; integrity, authority, changed-plan and whole-Fleet ownership
   failures stop the batch.

The current authority is
`qualification/authorities/nl-family-aware-20260924.json`. It binds the
85-recipe v1.0.17 catalog from tag `v1.0.17` (catalog commit
`efbbba29bd4c706c73d295d24047787be3f36d78`, source commit
`7b4ef279d4e531e51408ad08c11efd80483912e5`), assigns 72 single-Spark recipes
to 36 paired batches and schedules nine dual-Spark recipes exclusively. It
contains 90 recipe-specific recovery definitions; no recovery group is shared
in this release. The four recipes requiring more than two Sparks remain
explicit exclusions. Operator acceptance and capacity review remain row-level
gates, and territorial license notices are informational. Refresh authority
and coverage only after the exact new catalog and packages have been accepted;
the generators read the accepted catalog pin and record current indexed-package-
versus-accepted stack divergences until then. Recipe PR #124 merged the typed
contracts and generators; release workflow `36078172793` published v1.0.17 on
2026-09-25 at 00:37 UTC. This closes recipe publication, not platform runner
deployment or physical qualification.

The previous v1.0.16 catalog had a stale `source_commit` pointer to
`a0ffd873…`, while the digest-verified LTX 2.19B package contained protocol
wheel 3.0.0. Recipe release v1.0.17 corrected the catalog source pointer to
`7b4ef279d4e531e51408ad08c11efd80483912e5`. Runtime-stack identities continue
to bind the selected regular-file bytes in the exact catalog-pinned package;
the `source_commit` field alone does not establish those bytes.

Historical cache evidence for Step1X geometry 1.2.15 is exact. Its detail at
`2026-09-25T03:48:58.288046Z` reports the recipe cache ready and fit allowed
only on Spark 2297. A normal `force=false` preparation completed at
`03:49:48.966382Z`, returned image digest
`sha256:c8e6ac563cfa95c1f3a26bec630d8928aebc0b0fe994c2c7dea3a08794b1acbf`, and
reused 11 model artifacts totaling 7,249,194,310 bytes. This is exact cache and
image evidence, not inference or recovery evidence.

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
