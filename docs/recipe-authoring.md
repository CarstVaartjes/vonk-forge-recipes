# Create and update recipes

Use this workflow for a new recipe, an upstream refresh, or a targeted fix.
The outcome is a runnable recipe with exact inputs, useful version information,
and a self-contained downloadable package. A repository recipe is expected to
work; authors do not maintain a separate approval or readiness lifecycle.
The repository process and supported-contract policy live in AGENTS.md.

## Hardware qualification status

Qualification is derived from hardware evidence, never an authored recipe label.
A proven recipe stays proven until a hardware test on the same identity says
otherwise. Updating a recipe, runtime image or Model creates a re-check, never
an advance declaration that the recipe is broken. Reverting to a passed exact
identity restores Working without running hardware again. Quality notes do not
turn a successful hardware check into a failure. A missing historical Spark
count does not change a matching recipe digest, which already binds topology;
local proof survives while public certification still requires recorded platform
provenance. An unrecorded historical image is unknown, not a changed image pin:
matching recipe, resolved Models and recorded topology retain local proof unless
image-bound evidence establishes a different runtime. Restoring that proof must
retain its measured identity, never fill missing pins from the current library.

The contract 2.4.0 models and `derive_status` are the authority. Evidence lives
separately in `qualification/evidence/<publisher>/<recipe-slug>.json`; recording
it never changes the recipe's content digest. Each record binds recipe
`content_sha256`, resolved runtime image digest/build key, Model digests, test
date, Controller source/version, agent version, GB10 hardware class and Spark
count/topology. Unknown historical provenance stays unknown. Public evidence
has closed summaries/codes and digest references, never raw messages, host
names, addresses, run/request/operation ids, paths or credentials.

The local sweep view uses these statuses:

- **Working:** passed on the current exact identity; quality notes may remain.
- **Re-check: worked before:** an identity changed after a pass; show recipe,
  image and/or Model changes and the last successful test.
- **Fix to verify:** a recipe failure preceded an identity change.
- **Failed: recipe:** failed on the current identity; show cause and public fix
  reference, or “needs fix”.
- **Blocked: platform:** infrastructure/Controller fault; never recipe blame.
- **Untested:** no hardware verdict.
- **Can't test here:** requires more Sparks than are enrolled.
- **Testing now:** occupies a lane; the prior proof remains visible.

The signed catalog's per-recipe `qualification` is recomputed at publication
against the recipe, image and Models in that release. It publishes Working,
Re-check, Fix to verify, Failed: recipe, or **Not yet verified**, with a reason
(never tested, needs more Sparks, platform issue under fix, or incomplete
historical provenance). Fleet/lane states remain local. Working requires exact
resolved identity and recorded platform provenance. The proof includes the
Controller release, so consumers can show whether it was proven on the user's
same, older or newer platform; it is not a claim about an untested platform.
Normal library refresh delivers the newest evidence. Controllers must retain
published proof and use the shared contract rather than storing a status label.

## 1. Establish the change

Follow [`AGENTS.md`](../AGENTS.md) for branch isolation and repository checks.
Inspect the closest existing recipe and Model before choosing a contract shape.

State which kind of work you are doing:

- **Create:** add a supported model/variant or a distinct way to run it.
- **Refresh:** compare existing immutable inputs with current upstream sources
  and adopt applicable changes.
- **Repair:** correct a specific execution, packaging, or metadata problem.

Do not report a structural edit or repair as a complete upstream refresh. For
a catalog-wide request, account for every recipe, including unchanged recipes.

## 2. Check the actual upstreams

Inventory the sources used by the recipe: primary and companion models,
engine or specialized fork, wrapper/adapter code, build dependencies, patches,
and every external Dockerfile base image. A multi-stage Dockerfile can have
several independent base images. For creator intake, inspect new runtime/model
profiles inside older tracked repositories as well as newly created ones;
`creators.json` records discovery sources. A README inventory does not establish
coverage of every branch or runtime file. This is an author review obligation,
not an automated completeness guarantee.

Use primary sources: release notes, model cards, repository commits and diffs,
Dockerfiles, image manifests, and upstream tests. Resolve tags and branches to
immutable revisions or image digests. Record the time of the check and the
exact source URLs. An inaccessible source is unverified, not unchanged.

Compare each previous pin with the proposed pin in the same source repository.
Inspect relevant changed files as well as commit subjects. Distinguish actual
weights, tokenizer, configuration, runtime, and build changes from documentation
or metadata changes. Check renamed/removed options and whether the selected
ARM64 image, CUDA stack, patches, and Spark topology still fit together.

Follow the recipe's intended upstream. A specialized fork may contain required
support absent from the newest generic engine. Keep such a pin with a concrete
reason, or port its required behavior before replacing it. Do not use mutable
`latest` references in the resulting execution contract.

`tools/check-upstream-drift` can help inventory watched sources, but verify its
coverage before relying on it. A clean report for watched sources does not
prove that every embedded source, container, or dependency is current.

## 3. Author the Model and Recipe

The [Pydantic definitions](../contracts/src/vonk_forge_contracts) are the source
of truth. Use the [examples](../contracts/src/vonk_forge_contracts/examples) for
structure only; replace synthetic identities and data with verified inputs.

The roots are strict: authoring rejects unknown fields. Do not use lenient
published-document readers to bypass producer validation (`tests/test_contracts.py`).

| Document | Required structure (read the definitions for typed variants) |
| --- | --- |
| Model | `identity`, `metadata`, `requires_token`, `dependencies`, `modalities`, `source`, `format`, `license`, `files`, `capabilities`. |
| Recipe | `identity`, `metadata`, `models`, `execution`, `runtime`, `topology`, `interfaces`, `validation`, `provenance`, `settings`, `release`; `options` is optional. |

`settings.kind` selects the typed generation, embedding, job or decision shape;
never invent an LLM context length for an artifact job. `interfaces` contains one
OpenAI-compatible HTTP or filesystem job interface. Native decision models use
`POST /v1/systemone`, `openai.systemone` and `decision.probabilities`; use the
`systemone.py` typed requests/responses, preserving native result shapes and
extensions (`tests/test_systemone_contract.py`).

Contract compatibility: the library's release version is the contract's
semantic version (`CONTRACT_VERSION`, release `v2.3.0`). New or changed models
and recipes never change it: publication updates that release in place and
records when it was last updated, and vonk-forge follows the newest release
within its contract major version, so recipe changes never need a vonk-forge
release. A change to the contract itself bumps `CONTRACT_VERSION`:

- **Additive (minor):** add an optional field without changing existing wire
  requirements. The Controller reads published documents with `read_model` and
  `read_recipe`, which ignore fields they do not know, and identifies each
  document by the `document_sha256` of its published JSON, so an older
  Controller keeps working and picks the field up when vonk-forge upgrades.
- **Breaking (major, `3.0.0`):** remove or rename a field, make one required,
  or change a field's meaning. Making a required field optional also needs
  consumer compatibility evidence if producers will stop emitting it: ignoring
  unknown fields does not let an old reader accept a missing required field.
  Coordinate a vonk-forge release; Controllers on
  the previous major keep following the previous major's release.

### Model: the exact files and their capabilities

Reuse an existing exact Model when appropriate. Create a distinct version or
variant record when the underlying model identity changes. Family names and
new versions are data; they do not require new Python classes.

`identity.family` is the upstream lineage, not the checkpoint you are adding:
every quantization, fine-tune and republished checkpoint of Qwen carries
`{"publisher": "qwen", "slug": "qwen", "title": "Qwen"}`, whoever published
it; the republisher is the Model's own `identity.publisher`, and the weight
format belongs in `version`, `variant` and `format`. Reuse the family an
existing Model of that lineage already declares
(`rg -l '"slug": "glm"' models/`); invent a new one only for a new lineage,
with the upstream creator's slug as its publisher. `tools/check-model-families` fails a family
that names a republisher or a format, or a slug that two documents describe
differently, because the public catalog groups by this field.

Resolve an immutable source identity and enumerate the files needed to load
the model, including configuration, tokenizer, companion data, and license.
For GitHub release assets, bind the numeric release ID and each file's numeric
asset ID; do not use a mutable tag or browser download URL as the source
identity. This provider supports public, anonymous assets only. For Git LFS
objects in a public GitHub repository, use the `github-lfs` source: pin the
full commit SHA as `revision`, make every file of the Model one LFS object
(`sha256` is the LFS oid, `size_bytes` the LFS size, `path` the repository path
at that commit; no empty files, no split parts, no token), and vendor the
repository's plain git files in the recipe package, where
`check-vendored-upstream` proves them byte for byte. Keep the file
path, SHA-256, and byte size in the Model file manifest. Record actual file
paths, IDs, hashes, sizes, and purposes. Preserve legitimate
empty supporting files with their real empty-content digest. Do not invent
hashes, sizes, capabilities, context limits, or memory measurements.

For a Hugging Face model, `tools/catalog-hf-model` drafts that record. It pins
`--revision` (or the current head), lists every file in the snapshot with its
LFS SHA-256 and size (downloading and hashing small non-LFS files, and refusing
non-LFS files over 16 MiB), and writes a validated
`models/<--version-slug>.json`. It prints the pinned revision, the Model's
`document_sha256` to use in recipe model references, and the total bytes. You
still supply identity, license, `--requires-token` for a gated repository and
`--capability` names through its flags. Review the file list and roles, drop
files the model does not need, and list only capabilities the model supports
before you use the Model in a recipe.

For a `github-lfs` Model, `tools/catalog-github-lfs-model` takes the same
identity, licence and format flags plus `--repository`, a full 40-hex
`--revision`, and one `--role` for each `--path`. It reads each path
anonymously at that commit as a Git LFS pointer, and the pointer's oid and size
become `sha256` and `size_bytes`. A path that is not a pointer (a plain git
file) is refused: vendor it in the recipe package. It never downloads LFS
content, because the Controller verifies the bytes at ingress. `--probe-batch`
adds one batch request per object to confirm it exists; it transfers no bytes.

An upstream shard may be published only as `<name>.part00`, `<name>.part01`,
... (joined by `cat`). The tool recognizes
such a set and writes one file entry for `<name>` with `parts` (contract
2.2.0): each part's LFS SHA-256 and size, while the entry's own `sha256` and
`size_bytes` describe the joined file. Hugging Face cannot know the joined
file's digest, so give it with `--assembled-sha256 <name>=<sha256>` (repeat per
file) or let the tool read it from the repository's `sha256-manifest.txt` when
that lists the file. Without one the tool refuses; it never guesses. A set with
a gap, or whose joined name is also published whole, is refused. The platform
downloads and verifies the parts, assembles the file and installs it as one
ordinary file, so recipes select `<name>` like any other file.

Provider-gated checkpoints may be included. Set `requires_token: true` and
preserve the provider's license and access requirements. The Controller keeps
the token in its existing secret store; never embed it in a Model, Recipe,
build context or publication. A configured token does not prove that the account
has accepted terms or received model-specific approval. Verify access separately
and report pending approval accurately; publication is not download readiness.

`capabilities` is the list of capability names the model supports. Leave out
what is unknown. A source model's vision capability does not prove that every
engine recipe can serve images.

Preserve upstream license terms, territorial notices, and their source links
as information for the user; a republisher's permissive card metadata does not
replace the original weight licence. License compliance decisions belong to the user;
do not add location setup, territorial admission gates, or tests requiring
Vonk Forge to deny downloads or runs based on geography. A provider may still
require authenticated access to download gated files: preserve those actual
credential requirements and report provider access errors accurately.

### Recipe: a complete way to run those files

Reference exact validated Model documents by the `document_sha256` of their
published JSON, and their file IDs; do not duplicate their manifests. Mount
the selected files for the required topology roles; model mounts are always
read-only. Every recipe builds its image from a digest-pinned ARM64 base image
and a recipe-owned build context, Dockerfile and patches. The build reaches
only the hosts in `execution.build.network.hosts`; an empty list builds
offline. Include all required local build inputs, patches, entrypoints,
wrappers, and test fixtures. Model weights and container image bytes stay
outside the recipe package.

The image must be buildable by the Controller; publication builds the planned
image on a hosted runner and pins it in the signed index. The PR
`validate-recipe-library` checks the Controller's static source policy over the
build context (no Dockerfile heredocs or `ADD`, digest-pinned
bases, a numeric non-root final user, every URL host in
`execution.build.network.hosts`, no Compose device, host-namespace or bind
settings; put multi-line scripts in vendored files and `COPY` them). For the
launcher rule, the platform's runtime adapter verifies the engine's canonical
launcher in the built image (`/opt/vonk/bin/vllm`, `sglang-serve`,
`exllamav3-serve`, and so on), so the Dockerfile installs your wrapper at that
exact path and `runtime.entrypoint` names it. A launcher for the `ray`
parallelism backend starts Ray, so a build step must check the Ray CLI or
package it uses (`tests/test_ray_vllm_launchers.py`). When the parent image
has no Ray, install it from a hash lock as described in
[Dependency locks and build-time smokes](#dependency-locks-and-build-time-smokes). The image workflow lists, in each
run summary, why a recipe has no prebuilt image. Every image builds on a hosted
runner within the build job's 360-minute timeout; there is no exclusion list.

Leave out unused optional fields. For optional fields with a `None` default,
explicit `null` and omission have the same meaning; required nullable fields
must still be written, even when their value is `null`. Do not remove false,
zero, empty values or engine-owned JSON nulls as if they were absent. A
document's identity is the `document_sha256` of its JSON (key order and
whitespace aside), so any value change changes it. Keep documents in the
repository's sorted, two-space-indented form. Do not add hand-maintained digests
for files authored in the same change; tooling owns package/source digests.
Recorded upstream artifact checksums remain valid inputs
(`tools/build-catalog-index`, `tests/test_recipe_packages.py`).

Preserve launch behavior: executable, ordered arguments, environment, topology,
ports, model aliases, resource envelope, and stop timeout. Model targets stay
under `/models`; every distributed role needs selected model files
(`tests/test_recipe_model_mounts.py`). Multi-node topology derives one connected
fabric with a 200 Gbit/s floor, endpoint withdrawal on rank loss, worker-first
recovery and endpoint-owner-first stopping; do not author contradictory copies
of those derived properties (`RecipeTopology`, `tests/test_contracts.py`).
Declare per-role
unified memory as `peak_bytes` (the most the workload uses at any time,
startup or steady state) plus `reserve_bytes`, and disk as image, artifact,
working (staging plus caches) and safety-margin bytes. Declare `node_count`,
roles and their start order. Bind tunable
arguments to the corresponding declared settings instead of maintaining two
copies of a default. Respect the contract's automatic/unspecified settings;
benchmark request counts are not engine concurrency limits. If a wrapper
hardcodes a setting, align its implementation with the declared setting.

### Sandbox constraints

The workload sandbox deliberately blocks io_uring. Recipes, option choices and
image launchers must use a loader with a verified non-io_uring path: standard
safetensors, InstantTensor automatic/AIO or BUFFERED/MMAP selection (never
URING-only), or b12x disk PLE/Engram with `B12X_DISK_BACKEND=gds` and
`CUFILE_FORCE_COMPAT_MODE=true` (requires cuFile >= 1.14 development files).
Older b12x images without that backend need a tested POSIX reader replacement;
never request relaxed seccomp. Enforce image-owned transport policy in the
launcher before exec, including when a kit declares an unsafe backend. b12x
defaults PLE BF16 tables and disk Engram to io_uring even when no backend is specified.
The separate `--load-format b12x` checkpoint reader also selects io_uring bounce
I/O on GB10; `B12X_DISK_BACKEND` does not control it. Use `--load-format safetensors`
at the image launcher handoff for checkpoint loading while retaining the B12X
attention, MoE and linear kernels. Enforce that transport when an upstream Recipe
requests `b12x`; keep the checkpoint I/O policy separate from PLE/Engram I/O.
See [b12x disk transports](https://github.com/local-inference-lab/b12x/blob/2cc7f66aaef76a9c7cbb660a1efb865cf3f26c88/docs/disk-embedding-backends.md),
[NVIDIA compatibility I/O](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/#tracking-the-io-after-enabling-compatibility-mode),
and [InstantTensor backend selection](https://github.com/scitix/InstantTensor#backend-selection).

### Memory: preserve proof, reserve by need

The shared `vonk_forge_contracts.memory_admission.evaluate_memory_admission` owns
admission, explanatory headroom and the 6 GB platform floor. Every recipe and
option variant, on every role, must pass its empty-target-pool check: peak plus platform floor fits the
pool. `reserve_bytes` is informational and is never charged. Offline validation
(`require_admissible_declaration`, `qualification/vllm_memory_fit.py`) charges
one idle host baseline, `ADMISSION_IDLE_USE_BYTES` (130,663,231,488 −
126,600,000,000 idle-available bytes), calibrated on the proven set: the live
Controller admitted proven 120,370,379,081-byte peaks on both Sparks of a dual.
It proves declaration admissibility, not live fit.
The Controller supplies observed host use, exact commitments, unmaterialized
claims and unknown resident upper bounds to this same function. It does not
assume the historical idle baseline below. Proven recipes are calibration inputs,
never exemptions or candidates for automatic retuning.

**A.** Preserve the runtime settings of every recipe that passed on real hardware.
Change them only after a hardware failure, or deliberately with hardware retesting.

**B.** New/failing recipes declare, per role, their honest measured peak
(owner, 2026-10-27). It covers selected weights (including drafts), any
absolute KV pool and runtime overhead; that loading floor is a lower bound, and
where hardware evidence shows a role using more, its peak is that measured use
(entrypoint and worker may differ). It must be admissible
on an idle Spark. An honest peak that is not admissible means the recipe can
never fit: report it; never shave the number. The guard is calibrated on the proven set, which
`tests/test_absolute_kv_capacity.py` replays through the touched path for every
proven vLLM/SGLang document. A KV pool holds at least one full-context request,
not declared context × engine concurrency: engines schedule requests within it.
A single-Spark recipe file a change touches must bound KV: an absolute pool
(vLLM `kv-cache-memory-bytes`; SGLang `max-total-tokens`), or a declared
`gpu-memory-utilization`/`mem-fraction-static` whose share of physical memory
plus the 8 GB overhead (plus the resident expansion of geometry pinned to the
launch) is admissible on an idle Spark. Proven singles run up to 0.86, exactly the largest admitted peak;
vLLM's implicit 0.90 is not admissible. A touched pool-less vLLM budget (any
Spark count) must also hold the selected weights per rank plus the 8 GB
overhead: Laguna S 2.1 r0b0tlab at 0.6 left 4.25 GB beside 74.15 GB of target
and draft weights, and vLLM found no memory for KV cache blocks. Where the launch
pins cache geometry, the budget also holds one full-context request, which vLLM
requires before it starts: Laguna S 2.1 Mia on the 826aacdf checkpoint at 0.85
left 1.12 GB beside 101.94 GB of weights for a 13.0 GB request. gpt-oss-120b at 0.85 beside its 10 GB
Marlin repack exhausted host memory. On one Spark a declared SGLang fraction is
bounded whether or not `max-total-tokens` caps the pools: its share must be
admissible, and the fraction × the idle available bytes must hold the selected
weights, their resident expansion and any capped token/state pools (GLM a0 at
0.75 was OOM-killed; restoring its 262,144-token cap needs 116.8 GB against
0.85's 106.8 GB). The declared peak covers the same static need plus the 8 GB
overhead. SGLang derives an omitted fraction at launch, so none is required. A pool-less single-Spark vLLM declaration's peak must cover its
utilization budget plus the overhead, since the Controller
reserves the declared peak. `tests/test_absolute_kv_capacity.py` replays the recorded
failures and must reject each. `qualification/memory_need.py`
supplies the guard and an informative full-workload calculator. For a
multi-Spark vLLM absolute pool, also set an explicit `gpu-memory-utilization`
startup budget whose fraction of physical memory plus runtime overhead fits the
smallest role peak: vLLM checks it against free memory before loading weights.

An absolute pool must also hold one request at the declared context (or the
pinned model default when the engine context is omitted/automatic), including block rounding, packed scales
and per-rank cache layout. Tensor parallelism shards GQA heads but does not
halve replicated MLA latents. `require_absolute_kv_capacity` checks every option
combination; the multi-Spark startup invariant precedes the historical-content
waiver. SGLang absolute token pools must hold one context too. TRT-LLM fractional
pools are outside this absolute-pool check. Cache geometry binds configuration
hashes and records padded/hybrid allocation bounds with their sources.
Every geometry names its pinned config's `architecture`, classified as hybrid
or attention in `qualification/memory_need.py`. A vLLM-pinned hybrid geometry
records its Mamba/GDN layers as typed `recurrent` dimensions from that config
(recurrent and attention layer counts, conv channels and kernel, temporal state
elements and dtype, kernel block, and whether the loaded vLLM class supports
prefix-caching mode "all", with its Mamba chunk). vLLM pads each recurrent page
to the attention page, groups layers by the smallest bucket, holds 1, 2 or a
context's worth of pages per recurrent group (prefix caching off, "align",
"all") plus one per speculative token, and takes a whole group block as the
null block; `vllm_hybrid_need` computes that bound from the recipe's prefix
caching, cache dtypes, speculation, block size and TP and ignores any hand
`state_bytes_per_request`, which other engines keep. A hybrid whose vLLM layout
the formula cannot express is named, with its reason, in
`VLLM_UNMODELED_HYBRID`.
For an unproven failing pool, prefer the creator context and reserve at least
one request at that context, checking the worst option. If that cannot be
admitted with selected weights and runtime overhead, cap the explicit engine
context at what fits and disclose the cap in the recipe description.
Preserve proven recipe bytes; an estimate is never physical acceptance.

The informative workload estimator (not declaration admission) uses
**130,663,231,488 total bytes − 17,179,869,184 bytes
(16 GiB estimator headroom) − 5,012,574,208 measured idle bytes =
108,470,788,096 bytes**. The 16 GiB headroom is this estimator's own rule: the
workload limit subtracts it, and a touched multi-Spark utilization-sized pool
must leave it (below). It began as the host memory guard's per-sample
MemAvailable bar (vonk-forge `040c51b8531523bebe542b5e1f7b9403e0767f72`,
`host_memory_guard_policy.rs::STALL_AVAILABLE_BYTES`), which the swap-aware
guard (vonk-forge #1936, `fef665bf820346c885d50723e3179b2335c9216a`) removed.
That guard has no per-sample bar; it stops managed workloads on
R1 MemAvailable < 1 GiB and free swap < max(2 GiB, 10% of swap) for 2 s
(without swap: MemAvailable < 256 MiB for 3 s); R2 swap > 90% used and still
filling with MemAvailable < 4 GiB for 3 s; R3 swap-ins plus refaults ≥ 50,000
pages/s with full PSI avg10 > 60% and MemAvailable + free swap < 8 GiB for
10 s; R4 full PSI avg10 > 60% for 30 s while serving, or avg60 > 80% and not
falling for 60 s while loading; R5 an NVRM allocation failure together with
R1's or R3's condition, or MemFree < 1 GiB, held for 2 s
(`host_memory_guard_policy.rs`, thresholds calibratable). Idle use is from spark-3542
2026-10-15/16 incident telemetry, **125,650,657,280 bytes idle MemAvailable**,
retained in the calculator's historical source. Runtime
overhead currently uses an **8,000,000,000-byte estimate**, not a measurement. A vLLM
speculative drafter adds nothing beside the budget, however it is loaded: the
worker loads it after its free-memory snapshot and profiles it inside
`gpu-memory-utilization` (sources in `qualification/vllm_memory_fit.py`), so its
weights count inside the budget like the target's. Pool-less single-Spark runs
with a separate drafter held at most 9.9 GB beside utilization × memory, OS
included. The 2026-10-13 DFlash2 freeze at 0.7 (image 155ce16b) is unexplained
and 0.7 has not been re-run since; the 0.6 document passed on 431593a0 with
3.7 GB beside its budget. The estimate over this limit
is a **WARNING**, not a rejection: proven singles run utilization-sized pools
above it. Never omit a failing recipe to pass the check.

**C.** Assume an existing recipe works until hardware evidence says otherwise.
Only new or changed recipe documents are subject to the need guard. An exact
`content_sha256` in the committed `historical_documents` snapshot in
`qualification/memory-inputs.json` or in any `qualification/evidence/` record
identifies existing content, regardless of its qualification outcome. The snapshot's
`historical_sources` records its source commits; it is a content baseline, never
hardware proof. Validation reads these committed identities without Git or remote
refs. Changed content and new ids must satisfy the need guard; supplying an old
digest cannot exempt modified recipe bytes or a modified parsed launch. Existing
untested recipes produce informative **WARNING** lines in
`tools/build-catalog-index` and `tools/check-recipe-memory-need`; estimates never
rewrite them or assign a “does not fit” verdict. Qualification status comes only
from `qualification/evidence/` and the shared `derive_status` reducer. There is
no second pass/failure list in memory inputs; that file contains numerical
geometry, source references and the immutable content baseline only. Preserve all
unrelated recipe bytes.

Repair unproven, unadmittable declarations by calculated need, preserving absolute
cache requirements and bounding engine allocation (an absolute pool or an
admissible engine fraction). Peak plus reserve must fit the owner's
idle capacity, which already subtracts host use. The catalog guard rejects changed
declarations that exceed capacity despite a fitting need. Preserve proven bytes,
calibrate against hardware passes, and leave unknown or oversized needs unresolved.

Proven recipes are the calibration set. Run
`tools/check-recipe-memory-need --calibrate` without exemptions before changing
an estimator. The catalog loading floor must reject no proven recipe; engines
outside its vLLM/SGLang scope are reported explicitly. A loading floor does not
prove full-context capacity, graph peaks, or runtime fit. Missing full-workload
geometry is an unknown warning, never a hardware failure.

[tests/test_vllm_memory_fit.py](../tests/test_vllm_memory_fit.py) retains the
pre-#1612 calibration guard for historical vLLM documents and options on **each
role of each node**, including dual/multi-Spark recipes. Its shared
implementation and constants live in `qualification/vllm_memory_fit.py`: utilization ×
**130,663,231,488 bytes** + **8 GB** runtime overhead must fit the declared
per-node peak; peak + the **6 GB OS floor** must fit
the **126,600,000,000-byte idle-available floor**. The 2026-10-15/16 incident also
retains its measured overhead and the lower spark-3542 idle-available value above.

[qualification/memory_need.py](../qualification/memory_need.py), exercised by
`tests/test_single_spark_memory_headroom.py`, shares those constants; its
single-node full-workload estimate is
informative. An absolute vLLM KV pool overrides cache sizing by utilization.
Changed multi-Spark documents use the per-rank demand declaration below;
a changed multi-Spark SGLang document must also fit its sharded weights, resident
expansion and capped token/state pools in `mem-fraction-static` × the lowest idle
available bytes. Its absolute `max-total-tokens` cap must hold one full-context
request, not context × `max-running-requests`: SGLang schedules requests within
it, and the proven Qwen3.8 Flash Next W4A16 dual caps one 262,144-token context
at concurrency 4. A hybrid model's `max-mamba-cache-size` must hold one
request's five state slots; at startup SGLang lowers `max-running-requests` to
slots // 5, so 5 slots serve concurrency 1. With speculation the budget also
charges (min(`max-running-requests`, slots // 5) + 1) ×
`speculative-num-draft-tokens` intermediate state slots. Geometry that names a
`runtime_image_digest` applies only to that image. A multi-Spark vLLM recipe file a change touches must bound KV
(`tools/check-recipe-memory-need --base origin/main`): an absolute
`kv-cache-memory-bytes` pool that holds one full-context request per rank, or a
`gpu-memory-utilization` whose share of physical memory plus the 8 GB overhead
leaves the 16 GiB estimator headroom. Laguna M.1 at 0.82 with no pool fell to
16 GiB MemAvailable, the host guard's per-sample bar at the time, on 2026-10-26; proven utilization-sized duals run at 0.80 or below. The
pool need not hold context × `max-num-seqs`: vLLM schedules sequences within it,
proven duals hold fewer, and the DSpark vLLM 0.25 dual pool sized for
16 × 200,000 tokens exhausted host memory on both Sparks.
Sharded weights, that pool and the 8 GB overhead are bounded by idle admission,
not by the 16 GiB planning limit: hardware-proven dual documents run above it.
Existing documents remain unchanged and receive informative warnings.

SGLang profiles loaded weights before applying `max-total-tokens`. A fraction
below the resident model size can therefore produce a negative hybrid-cache
budget even with a short context. For a large hybrid model, cap both the token
pool and `max-mamba-cache-size`, bound concurrency and disable graphs (or state
in the description why capture stays in the non-static slack), then use a
fraction that permits profiling. Budget the actual capped allocations,
including state padding, index storage, resident weight expansion and runtime
overhead; the fraction alone no longer describes that peak. Tie any geometry
bound to the exact Model identity and test the pinned engine's sizing path with
the old failing inputs. Label estimated overhead and retain physical testing as
an outstanding check. Never raise an uncapped fraction to hide a memory failure.
The a0 GLM EXL3 geometry in `qualification/memory-inputs.json` binds the exact
model configuration and image: 64 KiB/token, 150 MiB/state including the padding
slot, and a 14 GB inferred resident expansion beside the default 8 GB overhead.
The calculator charges the actual absolute token/state pools, rejects graph or
image drift, and checks the declared peak against the shared limit.

For a named tool choice that must produce one invocation, use a single native
grammar rather than a repeating triggered tag. Preserve the argument schema's
local `$ref`/`$defs` scope and the engine's automatic/parallel tool handling.
Test the old grammar accepting duplicate calls and the replacement rejecting
them, plus streaming and non-streaming parsing of the resulting native frame.
Keep the serving assertion strict; response deduplication can discard real calls.

vLLM 0.1.dev20133+gb5f995e73, in the B12X base
`sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd`, builds
its strict DeepSeek V4 tool grammar with a reasoning block even for
`thinking: false`, so a named or required call cannot stop and runs to
`max_tokens`. A `deepseek_v4` recipe on that base sets
`VLLM_ENFORCE_STRICT_TOOL_CALLING=0`
(`tests/test_deepseek_v4_b12x_named_tool_stop.py`). Newer B12X builds, such as
vLLM 8244d1c4 in base `sha256:03a956544e9f...`, pass `reasoning=False` and need
no setting.

On GB10, CPU and CUDA allocations share physical memory. For unproven
single-Spark jobs, keep models on CUDA when the complete resident weights,
startup transients, activations and runtime overhead fit `LIMIT_BYTES` from the
shared calculator. CPU/model/sequential/group offload adds device transitions
and can duplicate storage; it is not a second memory pool. Inspect the pinned
loader as well as steady inference. Retain proven execution settings and real
disk streaming where residency or loading peaks require it. A smoke estimate is
not full-length capacity or hardware acceptance.

MOVA at source `1591b2557b42030a3421442c998dd6f5602ed9bd` offers `none`,
`cpu` (Accelerate model hooks) and `group` (leaf streaming). Its step invokes
visual/audio/bridge pre-forward hooks for each sequential CFG pass; Accelerate
1.15.0 skips transfers for already-resident components, so hooks alone do not
prove the source of all observed latency. Both pinned snapshots contain
77,672,622,000 weight-file bytes, including both 28.58 GB video DiTs, T5,
audio DiT, bridge and VAEs. With one frame there are 880/3,600 visual tokens
at 352x640/720x1280 (latent stride 8, patch stride 2). Estimate 2/8 GB for
activations and decoding, plus the shared 8 GB runtime estimate and 10 GB
loading allowance (largest shard rounded up): 97.67/103.67 GB, below the
108.47 GB safe limit. These are planning estimates, not measured peaks.
`--remove_video_dit` only deletes the high-noise DiT after the low-noise switch;
it does not change the initial loading peak or inference mathematics and is
unnecessary with resident models.

The MOVA smoke keeps one frame and two steps. Budgets are 1,200/3,600 seconds:
the 2026-10-21 sweep observed 325.32 s/step at 360p with offload, so allow
650.64 seconds for two steps plus 549 seconds for cold loading and export.
For 720p, scale denoising by the 3,600/880 token ratio (2,662 seconds) and
allow 938 seconds for loading/export. This deliberately assumes no offload
speedup, uses the job's existing one-hour ceiling, and needs hardware validation;
attention can scale worse than linearly. Each reference-image case gets the
same bounded job budget.

### Multi-Spark memory demand and nonfitting recipes

`qualification.memory_need.per_rank_need` reports an optimistic loading floor:
selected weights divided over tensor/pipeline shards, explicit KV bytes, and
8 GB estimated runtime overhead. The complete selected auxiliary weights are
counted once. Engram tables are never memory usage (owner, 2026-10-26): their
files, listed by content digest in `engram_files` of
`qualification/memory-inputs.json`, are excluded for every recipe, from the
per-rank floor and from the single-Spark `calculate_need`. A recipe that
sets a positive `DSV41_CACHE_GIB` row cache is charged that size per Spark,
shared by its ranks. Replicated tensors, graph buffers, recurrent state and loading
expansion may require more. Use source-backed `calculate_need` where its
geometry is available; keep unknown full-workload terms visible.

Calibration separates physical feasibility from planning headroom. The lower
observed idle MemAvailable is **125,650,657,280 bytes** (spark-3542, 2026-10-15/16,
retained in `qualification/vllm_memory_fit.py`). Subtracting the host pressure
floor leaves **108,470,788,096 bytes** for conservative new-document planning;
that policy margin is not proof that an existing recipe cannot execute.

Utilization fractions size adaptive pools; they are not irreducible workload
demand. [vLLM's cache contract](https://docs.vllm.ai/en/v0.19.1/api/vllm/config/cache/)
states that absolute KV bytes override utilization. The
[SGLang memory guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/hyperparameter_tuning.mdx)
separates weights, KV, graphs and activations. Do not count a fraction-sized
pool as a required full-context allocation.

A different number of Sparks is always a different recipe id, never a new
revision of the same recipe: name recipes per topology (`-single`, `-dual`,
`-triple`, `-four`, `-eight` as applicable).
`tools/check-recipe-topology` fails validation when a change alters the
`node_count` of an existing recipe compared with `origin/main`, and the
Controller's catalog sync skips such a revision (`recipe.topology_changed`).

Every Model a recipe selects must exist in `models/` at the pinned content
digest, with every selected file present. `tools/check-recipe-model-references`
checks this over the whole library on every pull request and names the recipe
and the missing Model.

Runtime arguments are ordered data, not shell text. Repeated names keep their
order; an empty string remains an argv token; literal `true` emits a flag and
`false` omits it. `value: null` requires a setting binding. Structured values
render as deterministic compact JSON. The contract bounds names/values at
65,536 UTF-8 bytes, rendered argv at 1 MiB and argument count at 128
(`tests/test_contracts.py`).

Trusted recipe options pass through to the pinned engine even when the
Controller has no label, enum entry, or specialized validator for them. Known
option metadata improves editor help; it is not an exhaustive allowlist. Do
not silently remove an option or replace its value after an engine error.
Preserve structural checks and reject conflicts with platform-owned execution
and security requirements. An unfamiliar option cannot grant host access or
change mount ownership.

The platform supplies engine cache and temporary paths and creates them for
the runtime UID/GID. Recipes must not duplicate or override those defaults.
Keep non-root execution, a read-only root, and declared writable volumes. If an
engine needs an additional invariant, fix the central engine implementation
and exercise actual writes and cache reuse; do not scatter recipe workarounds.
Bind generated rank/vision/model views to their checkpoint and engine inputs,
reuse complete cached work and rebuild malformed/interrupted output
(`tests/test_tony_tensorfold_cache.py`,
`tests/test_tensorfold_vision_cache_launcher.py`, `tests/test_prewarm_cache_seed.py`).

Artifact jobs receive their declared files under `/inputs`, plus
`/inputs/manifest.json`, and write their outputs under `/outputs`. Read
the manifest with the platform's `RecipeJobInputManifest` Pydantic model and
select files by their declared slot. Do not assume the input directory contains
only the prompt. Reject undeclared files and unsafe paths; preserve valid file
names, including uppercase names. The three native LTX adapters bundle the same
`vonk-agent-protocol` wheel as the Controller and install it in the image. When
changing this shared contract, rebuild that wheel from the platform's
`agent_protocol` source, replace all three adapter copies, and run the actual
manifest-producer-to-adapter tests before rebuilding the catalog.

### Job output staging

`/outputs` is a writable artifact root with platform-owned `tmp/` and persistent
`cache/` entries, not an empty working directory. Keep incomplete artifacts in a
unique directory under `/outputs/tmp`; generate and validate there, then
`os.replace` each final file into `/outputs` on the same filesystem. Create it
with `tempfile` defaults (the platform sets `TMPDIR=/outputs/tmp`); never pass
the output root as `dir=` or make directories in it. Start upstream processes
with `cwd` in that scratch, never an image path: the image is read-only, and
libraries write cwd-relative temp files (moviepy's mux audio). A killed job runs no
cleanup, and any non-file entry left there fails output collection. Export only
plain files matching signed output slots. Clean up only the scratch directory
you created; preserve caches, other scratch and existing final output on failed
generation, validation or rename. Retries may retain all those entries.
`tests/test_adapter_output_staging.py` rejects output-root emptiness checks and
newly added output-root scratch or image working directories, and exercises
atomic publication, killed jobs, failures and retries with retained entries.
`tests/test_job_writable_paths.py` catches cache/lock/temp writes into image paths.

A delivery-only retry observes a durable exit receipt and retained outputs; it
cannot launch a second process. Missing local evidence does not authorize a
rerun. These are platform lifecycle rules, not extra Recipe fields; the pinned
implementation evidence is [job receipt persistence](https://github.com/CarstVaartjes/vonk-forge/blob/486f7ff1e6dedb2b77d03bee36c8120e8bc5d1c6/rust/crates/vonk-agent/src/oci/jobs.rs)
and [delivery retry dispatch](https://github.com/CarstVaartjes/vonk-forge/blob/486f7ff1e6dedb2b77d03bee36c8120e8bc5d1c6/rust/crates/vonk-agent/src/executor/jobs.rs).
The output collector excludes platform-owned scratch/cache entries; see its
[pinned implementation](https://github.com/CarstVaartjes/vonk-forge/blob/486f7ff1e6dedb2b77d03bee36c8120e8bc5d1c6/rust/crates/vonk-agent/src/executor/jobs/outputs.rs).

### Tool calling, reasoning and images: claim only what the launch serves

vLLM, SGLang and TensorRT-LLM parse tool calls and reasoning only when the
launch passes the parser. A recipe follows its playbook or kit: if upstream
passes the flags, carry them (`--enable-auto-tool-choice` with
`--tool-call-parser` on vLLM, `--tool-call-parser` on SGLang, `tool_parser` on
TensorRT-LLM; `--reasoning-parser` or `reasoning_parser` for reasoning); if it
does not, the recipe must not tag `tool-use` and its `smoke_cases` must hold no
tool case, because the engine answers HTTP 400 or returns the raw `<tool_call>`
text in `content`. A reasoning model served without a reasoning parser uses the
`A323_UNPARSED`/`A391_UNPARSED` case; use its actual functional and quality
assertions rather than assuming that every kind of leaked markup is allowed. Image cases need a
multimodal recipe. When a kit's entrypoint injects the flags from inside its
image, record them with their source in `qualification/kit-engine-flags.json`.
`qualification/capabilities.py` derives all of this and
`tests/test_recipe_capabilities.py` fails CI when a recipe's tags or smoke cases
disagree with its launch. An SGLang recipe with `--reasoning-parser inkling`
copies and runs the shared `inkling-forced-tool-grammar.py` build patch, so a
required or named tool choice is one call; without it the reasoning wrapper
defers the grammar past Inkling's first call and forces a second
(`tests/test_inkling_sglang_forced_tool.py`).

### Options: user-selectable runtime variants

When users can reasonably choose between ways to run the same weights, declare an
`options` list instead of publishing near-duplicate recipes (contract 2.1.0).

| Field | Rule |
| --- | --- |
| option `name` | Slug, unique in the recipe. |
| option `label`, `help` | What the user sees in the web select and `vonkctl`; say what changes and the trade-off. |
| `choices` | 2 to 16 fixed named values. There are no free-form values. |
| choice `value` | Slug, unique within its option. |
| choice `label`, `help` | Shown next to the value; state measured or upstream-documented effects only. |
| choice `default` | Exactly one choice per option is `true`. It applies whenever the user does not choose. |
| choice `args` | Literal runtime arguments (`name` and `value`, never a `setting` binding). |
| choice `env` | Environment variables as `{NAME: value}`. |

**Deployment choices only.** An option must be a real deployment choice: it changes
what is loaded or how the server runs (drafter, vision on or off, quantization or
projection, context size when it changes memory, server flags a request cannot
set). Anything a client chooses per API call (thinking or reasoning on or off,
reasoning effort, sampling parameters such as temperature, chat template kwargs)
is not an option; leave the engine's default in `runtime.arguments` and make sure
requests can still override it.

**The default is the recipe as it runs today.** The recipe's own `runtime.arguments`
and `runtime.environment` hold the upstream defaults, so the default choice
normally adds nothing (`args` and `env` left out); when upstream's default
changes, change the base runtime and the default choice together. Choices carry
only what differs from it.

**Binding.** The chosen choice is merged into the runtime before the platform
compiles it, identically on every rank. An argument or variable with the name
the recipe already sets replaces it in place; a new name is appended. So an
option that needs a different CUDA graph list carries the whole replacement
`compilation-config` argument, and a choice that turns a feature off writes the
variable's off value instead of leaving it out. Two options may not change the
same argument or variable. Options cannot change weights, the image or the
build: different weights are a different Model and recipe, and anything a choice
needs at run time (for example a patch applied at build and switched by an
environment variable) must already be in the image. A choice may not download or
build anything.

**Forbidden.** A choice is trusted recipe data with the same limits as
`runtime.arguments` and `runtime.environment`: the platform still owns
security, users, networks, mounts, ports, writable paths, caches and telemetry.
The contract checks the structure (bounds, names, one default, unique names and
values, no conflicting options, every choice merges into a valid runtime) and
the library validation compiles every choice against the platform policy, so a
choice that sets a platform-owned variable such as `HOME` or a cache path, a
reserved `VONK_*` name or a loader-injection variable fails the library check.
Run it as in section 5.

Example (a real option from the GLM 5.3 Flash EXL3 recipe, abridged):

```json
"options": [
  {
    "name": "verification",
    "label": "Draft verification",
    "help": "How many drafted tokens the target verifies per step.",
    "choices": [
      {"value": "standard", "label": "Standard", "default": true,
       "help": "Verify all 7 drafted tokens every step (upstream default)."},
      {"value": "adaptive-k", "label": "Adaptive length",
       "help": "Verify a per-step prefix; needs extra CUDA graph sizes.",
       "args": [{"name": "compilation-config",
                 "value": "{\"cudagraph_capture_sizes\":[1,2,3,4,5,6,8,9,10,12,15,16,20,24,32]}"}],
       "env": {"GLM53_ADAPTIVE_K": "ema"}}
    ]
  }
]
```

Do not offer an option whose extra inputs the recipe cannot declare (for
example a preset that needs downloaded vectors, or one upstream documents as
ineffective): leave it out and say so in the version note.

## 4. Version and explain what changed

`release.version` and `release.released_at` name what the recipe runs. When
the upstream project publishes versions (GitHub releases or tags at the pinned
commit, or a versioned changelog entry), use the upstream version as upstream
spells it, without a leading `v`, and its release date. Otherwise keep the
recipe's own semantic version and bump it with each change. The recipe
document carries no release history; the Git history and the pull request
hold it.

For an update, write a short summary and a few useful highlights in the pull
request: new model capabilities, corrected behavior, engine compatibility,
memory or startup changes, and changes an operator will notice. Separate Model
changes from Recipe/runtime changes. Use release notes where available and
inspect commits between the exact old and new pins. Link each claim to its
supporting release, comparison, commit, or diff. Label performance claims as
upstream claims until measured here. Do not infer a speedup or quality gain
from a commit title. Missing or incomplete upstream notes do not prevent a
usable recipe from publishing.

## Refresh selection and repair

Follow newest applicable upstream revisions, not a history of old revisions
under one id. Reuse an exact Model already catalogued; delete an old Model only
when no recipe references it. A distinct weight/runtime variant or topology is
still a distinct recipe. `release` contains only the current version/date;
`tests/test_recipe_releases.py` rejects embedded history.

Fix a broken creator recipe instead of preserving a broken launch for fidelity.
Keep required specialized implementation, but repair missing imports, invalid
chat weights, output handling or unreproducible build closure with a regression
that fails before the fix. Do not hide defects with non-executable placeholders
(`tests/test_recipe_executability.py`). A chat validation needs an explicit
runtime chat template, a selected template file, or a served Model declaring
`chat`; a drafter's capability and an embedded tokenizer template alone cannot
make base weights chat-trained (`tests/test_chat_configuration.py`).

`tools/check-upstream-drift` inventories configured watches and embedded inputs;
unknown/mutable/transitive inputs remain evidence gaps, not proof of freshness.
The hourly `.github/workflows/refresh-upstream.yml` and `tools/refresh-upstream`
implement automatic selection (`tests/test_check_upstream_drift.py`,
`tests/test_refresh_upstream.py`): newest release/tag or head, preserving inferred
non-default branches and declared release lines. Never downgrade an ahead pin.
A mechanical move requires compatible vendored/used bytes, retained referenced
launch names, complete selected Model files and passing producer checks. Changes
to patched runtime files or missing selected files need review; metadata-only
changes do not independently justify rebuilding identical content. Shared build
contexts refresh together. Human repairs may intentionally diverge with evidence.
A proven recipe moves too: the refresh rewrites its entry in
`tests/fixtures/proven-recipe-bytes.json` and the sweep re-checks it. A refresh
branch only moves forward (never force-pushed); a PR whose checks failed on an
unchanged refresh is reported in the recipe's review issue and moves forward again
only when upstream or main's checks (`tools/`, `tests/`, `.github/`) change.

Respect `base-image-holds.json`: a held base and Dockerfile must agree until the
hold is updated/removed with working evidence (`tools/check-base-image-holds`,
`tests/test_base_image_holds.py`). `model-revision-holds.json` declines one Model
revision that cannot run on the recipe's Sparks, with its reason; a newer
upstream revision is assessed again (`tools/check-model-revision-holds`,
`tests/test_model_revision_holds.py`). Recipes in `published-images.json` follow the
published image digest and its recorded build commit, not unrelated newer Git
heads. `tools/check-image-provenance --base origin/main` checks changed recipes.
An unavailable recorded commit needs the explicit unverified provenance mark
and review; do not claim a byte comparison that could not run. Outages wait;
missing source is unverified (`tests/test_published_image_follow.py`).

### Kit-built recipes: the kit declares what it builds on

A recipe built from a creator's kit (a patch stack, launch scripts, a configuration that
names its base engine version, the model it serves, the image it runs on) pins **only the
kit**: `provenance.source_reference` (a `tree` URL, or a `blob` URL of the declaring file).
What the kit builds on is the kit's declaration at that commit, so the recipe never carries
a second, hand-authored pin for it. The patches are written for that exact base; the base
repository's newest release is information, not a candidate (TensorFold v0.6.2 existing
says nothing about a kit whose patches apply to v0.6.0), and neither is a model
repository's newest revision or an image repository's newest tag.

- **Declaration.** The adapter's `vendored-upstream.json` names where the kit declares each
  dependency, as a Renovate-style regex manager: `datasource`, `file` and a `pattern` with a
  named group `version`. Exactly one distinct value must match. The datasources:
  - `git-ref`: a tag, branch or commit of a GitHub repository (`TF_VERSION=...`).
  - `hf-revision`: a revision of a Hugging Face model repository, for the model and for a
    drafter. `binds` names the recipe model (`models[].id`) it pins.
  - `oci-image`: a tag (or digest) of a container image repository, locked as the digest of
    its `linux/arm64` manifest.
- **Lock.** `tools/kit-pins lock <recipe>` derives `kit-lock.json` from the kit commit the
  recipe pins: each declared value resolved once to an immutable one (a Git commit, a Hugging
  Face revision, an image digest), recorded with what it named (`"kind"`: `tag`, `branch`,
  `commit` or `digest`). `kit-lock.json` is generated output and is never edited by hand.
  Builds read only the lock and never resolve a tag. `lock` also moves what follows it: the
  vendored base archive and the Dockerfile's commit, archive file name and digest for a
  `git-ref` base; the recipe's Model binding (and the disk envelope, and removal of the old
  Model once no recipe uses it) for an `hf-revision`, which must be catalogued first with
  `tools/catalog-hf-model`; the Dockerfile's `FROM` digest and `base_image` for an
  `oci-image`. The Dockerfile header comment and licence labels are the author's to review.
- **Declared repository.** A kit that also declares *which* repository (a creator's fork,
  e.g. `ARG TF_REPO=`; a model id; an image name) gives `repo_pattern` (group `repo`),
  or a `repo` group in `pattern` itself when one line pairs the repository with the value
  (a shell `case` on the model id), instead of `repo`. The lock records the repository with
  the value, and a kit that moves to another repository is a review like a moved ref.
- **Branches.** A kit may declare a branch (a creator's fork branch, a Hugging Face `main`)
  instead of a tag or commit. `lock` resolves the head once and locks it as
  `"kind": "branch"`, as a Nix flake locks a branch input; the build and `check` use only the
  locked value, so a moving branch changes nothing by itself. The scanner resolves the head
  again: a moved head is a change of the kit's dependency, so a review.
- **Check.** `tools/kit-pins check` (CI) is the "lock is up to date" gate and enforces, per
  datasource, that the recipe carries no hand-authored pin for a declared dependency: it
  re-reads the declaring file at the pinned kit commit and fails when the committed lock
  differs, when the manifest keeps a hand-authored `source` pin beside a `git-ref`
  declaration, when the Dockerfile and vendored archive do not follow a locked base, when
  the Model bound to an `hf-revision` is not that repository at that revision, or when the
  Dockerfile's `FROM` or `base_image` for an `oci-image` is not the locked digest.
- **Override.** A recipe may deliberately deviate (a known-bad upstream, a mutable alias the
  kit names, a security fix) with `"override": {"ref": ..., "reason": ...}` on the
  dependency (`revision`, `digest` and `repo` are accepted where they apply). The lock records
  it, refresh reports it as overridden and proposes nothing.
- **Monorepo kits.** A kit that is one directory of a creator's repository, of which the
  recipe uses only the declared values (NVIDIA's playbooks name their TensorRT-LLM image in a
  README), sets `"whole": false` on the `kit` block: only the declarations are checked, and
  the repository's other changes are judged as for any source.
- **Refresh.** The scanner watches the kit and what it declares, never the dependency's own
  channel. A kit move that changes a declared value is a review (the patches were rebased,
  the image is another build; run `tools/kit-pins lock`), except one case: a kit that moves a
  declared model to another revision of the same Hugging Face repository whose files, apart
  from metadata, are the catalogued bytes (weights, configuration and tokenizer) moves the
  recipe's Model, its lock and the kit commit together. A different repository, different
  weights, a changed configuration or tokenizer is a review. A kit that moves its image, or
  a declared tag that now names another digest, of the same image repository moves the
  recipe mechanically (the lock, the Dockerfile's `FROM` and `base_image` digest; the thin
  layer on top is rebuilt on it); another image repository is a review. A kit move that keeps every declared value moves the recipe's
  source reference, the Dockerfile's kit label and the lock together. Recipes whose kit
  declares nothing machine-readable (or only a floating tag such as `latest`) keep explicit
  pins in the manifest or the Dockerfile and the scanner's newest-upstream rule: there is no
  duplicate to derive.
- **Prebuilt image rule.** If a creator's kit provides a prebuilt image, the recipe runs
  that image, pinned by digest, and does not rebuild it from the kit's sources: the
  adapter's Dockerfile is `FROM <image>@sha256:...` plus, at most, a thin layer (the
  Controller's launcher wrapper, licence texts, the non-root user). Declare the image as an
  `oci-image` dependency, and the image scanning, Trivy and supply-chain path that already
  covers every recipe image covers it. Self-building stays only for kits that publish no
  image. `tools/kit-pins check` scans every cited kit at its pinned commit for the container
  images it runs or pulls (Dockerfile `FROM`/`ARG` image defaults, `IMG=`/`IMAGE=`
  assignments, `docker run|pull|create`, compose `image:`; docs, tests and CI files are not
  scanned; engine and base images such as `vllm/vllm-openai`, `lmsysorg/sglang`, `nvcr.io/*`
  are not creator images) and fails when the recipe builds `FROM` none of them. A recipe that
  must not run the kit's image records why in `kit-image-exceptions.json`
  (`{"<recipe id>": "reason"}`); the check names an entry the recipe no longer needs (it runs
  a kit image, or the kit references none) so it is removed. `tools/kit-pins images [<recipe>...]` shows what
  the scan finds.
- **Judging kit changes.** An adapter built from a whole kit (declared, or one whose files
  cite `github.com/<kit>/tree/<commit>`) is judged by every changed kit file, not only by
  files named by URL. Not runtime: READMEs, changelogs, notices, docs, tests and client-side
  helper scripts such as a new `tools/*.py`. Runtime, so a review: patches, launch
  scripts (`*.sh`, `start`/`serve`/`run` Python), configuration and dependency files,
  Dockerfiles, and any file the adapter ships a copy of at the same path. A truncated or
  diverged comparison is never trusted.
- **Review issues** record the upstream heads (commit and timestamp) they were computed
  from; when a later run sees newer heads the body says "Upstream moved since the last
  check" with the old and new heads. A Model revision that differs from the catalogued
  one only in metadata files (a README edit) is not a move.

## 5. Validate and exercise the result

HTTP fixtures use the same typed request models in their handlers and tests.
Declare required, optional, nullable and extension fields explicitly, preserve
supported defaults and extensions, and keep deterministic content assertions
separate from structural validation. Raw dictionary equality is not a request
parser. Exercise both streaming and non-streaming behavior when supported.

Use the current [producer validation workflow](../.github/workflows/validate.yml)
for the exact commands and dependencies. Use a writable task-specific cache
for `uv`. Never fall back to older schemas to make a consumer or test pass.

Author representative `validation.serving.checks` in the Recipe and reviewed
qualification definitions in `qualification/recipes/<slug>.json`. The latter has
`id: "<publisher>/<slug>"` and the applicable `recipes` (artifact) and/or
`service_recipes` (HTTP) section; either section may be omitted. It is assembled
with shared fixtures and templates from `qualification/shared.json` by
`qualification.definitions_loader.load_definitions`. Duplicate ids fail. Add one
file for a new recipe; do not edit a generated plan, authority or catalog count.
Declare aliases matching `ENDPOINT_ALIAS_PATTERN` (lowercase profile identifiers);
`qualification.service_aliases` and `tests/test_qualification_service_aliases.py`
validate them. A colliding active alias receives the sweep's stable suffix;
use the reviewed alias in response assertions.

Service entries choose applicable `smoke_cases` and request templates from the
shared definitions; job entries declare real fixture slots, timeout and output
assertions. Use `qualification/case_assertions.py` and the typed contract's
assertion variants rather than inventing an ignored assertion. A fixture's
extension, media type and decoded bytes must match the declared slot; include
all referenced fixtures in the package. `tests/test_qualification_index.py`,
`tests/test_qualification_case_assertions.py`, `tests/test_optional_inputs.py`
and `tests/test_media_types.py` enforce closure and slot consistency.

A media smoke proves one small, safe output, not full-quality generation. Keep
native resolution, topology and output codecs; use the smallest frame count the
model accepts and a few denoising steps (retain required distilled schedules).
Declare per-job controls as Recipe knobs bound to runtime arguments, then select
small values in the qualification `parameters`, or use an existing request slot.
Do not add undeclared parameters: the Controller refuses them. Qualification
JSON is outside `document_sha256(recipe)`, although fixture/package bytes and
image source identities can change. Adding knobs changes the Recipe document;
preserve hardware-proven documents and their complete image build contexts.
Review every consumer of a shared context against hardware passes before editing
it. LTX BF16 distilled and TRELLIS retain their existing smoke settings because
their contexts are shared with proven LTX FP8 and Pixal3D respectively; do not
fork these adapters to introduce smaller smokes.
An explicitly requested source change, such as a reviewed protocol-wheel
refresh, updates the reviewed source pin without creating hardware evidence.
Keep prior passes and unknown image provenance unchanged; reviewed source bytes
do not certify the new runtime image. `tests/test_qualification_case_assertions.py`
guards every reviewed shared context and verifies that historical passes do not
become public proof of the refreshed context.

For pipelines that select native buckets, metadata assertions declare
`native_dimensions` (kind, target size and T2V aspect ratio) instead of fixed
width/height. The shared loader uses a stdlib translation of the pinned upstream
bucket rule and reads the digest-checked PNG fixture's IHDR dimensions for I2V.
Only resolved width/height
reach consumers. `tests/test_native_dimension_expectations.py` guards the
source pins, automatic-sizing contexts and wrong-dimension rejection.

MOVA's pinned temporal rule is `4k+1`, including one frame; smoke uses one frame
and two steps at 640x352 or 1280x720, with synchronized 48 kHz audio at 24 fps.
The target is a few minutes on one GB10, with a 600-second job budget including
adapter preparation, generation and ffprobe. The adapter reads the platform's
`VONK_JOB_TIMEOUT_SECONDS`, shares one monotonic deadline, and kills torchrun's
process group on expiry with stderr evidence. The read-only 2026-10-21 sweep
results record the 360p 193-frame/50-step attempt timing out after 3,606 seconds;
there is no completed 360p timing from which to estimate a measured speedup.
The few-minute target remains unverified until a separately authorized GPU run.

### What the sweep checks

The sweep process and verdict meanings live in
[AGENTS.md](../AGENTS.md#hardware-sweep-operation). Author for its actual executors:

- **HTTP service:** for chat, stream once to measure TTFT and tokens/second, then
  run reviewed cases; without reviewed service cases, use the Recipe's serving
  checks. Functional `assertions` gate execution (response shape, alias, choice,
  parsers, bounded content). `quality_assertions` record expected-answer misses
  as notes without failing a runnable recipe or filing a defect. Generic chat
  cases switch thinking off per request, while markup rules still apply. Keep
  the two assertion classes separate (`tests/test_sweep_quality.py`,
  `tests/test_qualification_chat_cases.py`, `tests/test_sweep_vonkctl_smoke.py`).
- **Artifact job with `--job-recipes`:** create/upload/submit one job against the
  held run, follow its receipt, download outputs and evaluate output count,
  names, media/format, size, image/audio/video metadata, GLB structure or declared
  document/transcript assertions. `spark_sweep/smoke_job.py` and
  `spark_sweep/job_assertions.py` are the executors. Host-unobservable assertions
  are recorded as `not_evaluated` with a reason, never silently passed. Video
  inspection needs ffprobe; inspect any missing observation before claiming
  coverage (`tests/test_sweep_job_smoke.py`, `tests/test_sweep_job_media.py`,
  `tests/test_image_output_contracts.py`). Without deployed job-profile support
  and the flag, jobs are skipped (`tests/test_sweep_job_recipes.py`).
- **Other non-OpenAI services:** `readiness-only` proves health/routing only;
  never call it successful generation (`spark_sweep/smoke.py`).

The sweep records telemetry as total minus available memory per node, including
the OS. A run observed alone above its declared bytes plus
`spark_sweep.memory.OS_ALLOWANCE_BYTES` gets a quality note, not a failure
(`tests/test_sweep_quality.py`). These observations and upstream performance
are not measurements of every topology/configuration. Test the selected runtime,
not merely a Model capability. Full qualification also needs restart/cache reuse
and relevant fault recovery; structural checks and hosted builds cannot prove
CUDA inference or fabric behaviour (`tests/test_family_aware_coverage.py`,
`tests/test_qualification_authority_contract.py`).

### Catalog guard repair map

Run the full suite; the named guards below are not a substitute for it. Each
failure should identify the recipe/input and be repaired at its source, never
by deleting coverage or weakening an assertion. Source-backed version notes,
newest-applicable selection, fix-over-fidelity and truthful evidence are author
review policies; no static guard can prove upstream completeness or honesty.

| Guard/test | Required invariant and repair |
| --- | --- |
| `tests/test_contracts.py`, `tests/test_contract_examples.py` | Strict root/field/relationship validation, bounded paths/argv, source/file identities. Correct data to the current definitions; regenerate examples/schemas after a contract change. |
| `tools/check-recipe-model-references`, `tests/test_recipe_model_references.py` | Every pinned Model/file exists at the exact digest. Catalog the missing immutable Model or update all bindings together. |
| `tools/check-model-families`, `tests/test_check_model_families.py` | One consistent upstream lineage. Reuse its family identity instead of republisher/format families. |
| `tools/check-recipe-topology --base origin/main`, `tests/test_recipe_topology_check.py` | Existing id keeps Spark count. Create a distinct topology id. |
| `tests/test_recipe_executability.py` | Unique normalized titles, executable metadata/entrypoint, actual shebang/ELF for copied executables. Repair the launcher or metadata; do not publish placeholder tags. |
| `tests/test_ray_poll_interpreter.py` | A launcher that starts a Ray head polls the cluster with `sys.executable`: the image's `python3`, the interpreter a build-time `import ray` check runs under. Never name an interpreter path the image may lack. |
| `tests/test_recipe_model_mounts.py`, `tests/test_adapter_model_mounts.py` | Model mounts and adapter paths agree and all ranks receive files. Repair targets/roles and local loader paths. |
| `tools/check-build-context-sources`, `tests/test_check_build_context_sources.py` | Every COPY source is shipped. Add the actual local file or fix the context-relative reference. |
| `tests/test_recipe_build_inputs.py`, `tests/test_build_network_hosts.py` | All external build stages are immutable and declared once; URL hosts are allowed. Pin every external base and declare real egress. |
| `tools/check-vendored-upstream`, `tests/test_check_vendored_upstream.py` | Declared vendored files match immutable upstream bytes. Re-vendor them; carry intentional changes as explicit patches/overlays. |
| `tools/check-model-construction-device --base origin/main`, `tests/test_check_model_construction_device.py` | New adapter runtime code (added lines, code only; vendored archives and `vendored-upstream.json` verbatim creator copies excluded) builds no full model as real CPU tensors (`init_weights_on_device(cpu)`, `low_cpu_mem_usage=False`) and puts no CUDA device in a model constructor. Build on meta, assign the checkpoint, fail on any parameter left on meta, then let the loader place the model. Upstream source a patch script rewrites is text to this guard: prove it with the adapter's patched-source behaviour test. |
| `tools/kit-pins check`, `tests/test_kit_pins.py`, `tests/test_kit_prebuilt_images.py` | Kit declarations, lock, image/base/model consumers agree. Regenerate the lock; use a justified override/image exception only for an intentional deviation. |
| `tools/check-image-provenance`, `tests/test_published_image_follow.py` | Claimed build commit matches the selected source image. Fix provenance or supply the supported explicit unavailable-commit evidence. |
| `tools/check-base-image-holds`, `tests/test_base_image_holds.py` | A held digest stays consistent. Change/remove the hold only with replacement evidence. |
| `tools/check-model-revision-holds`, `tests/test_model_revision_holds.py` | No recipe uses the Model revision its hold declines. Remove the hold only with evidence that the declined revision runs. |
| `tests/test_memory_envelope.py`, `tests/test_single_spark_memory_headroom.py` | Memory consistency, component accounting and need checks. Repair the reported launch/envelope inconsistency; follow the memory section above. |
| `tests/test_recipe_capabilities.py`, `tests/test_chat_configuration.py`, `tests/test_served_model_name.py` | Served Model, parser flags, aliases, tags and cases agree. Enable the real launch support or stop claiming the capability; use chat-trained weights or an explicit supported template. |
| `tests/test_recipe_option_controls.py`, `tests/test_qwen36_fp8_options.py` | Deployment choices and base defaults execute as declared. Bind actual runtime differences, keep request controls per request. |
| `tests/test_recipe_releases.py`, `tests/test_recipe_refresh_repairs.py` | Current release metadata, valid dates/version shape and bounded source closure. Correct release data or repair the specific refresh regression. |
| `tests/test_job_writable_paths.py`, `tests/test_adapter_output_staging.py`, `tests/test_local_media_path_exists.py` | Managed writable paths, atomic outputs, existing local media inputs. Fix path handling; do not add root/broad-write workarounds. |
| `tests/test_job_runtime_toolchain.py`, `tests/test_cuda_extension_builds.py`, `tests/test_torchaudio_torchcodec.py` | Required compilers, ABI-compatible media/native dependencies and build flags exist. Fix the locked image environment and import/build smoke. |
| `tools/prepare-remote-code-smoke`, `tests/test_remote_code_build_smoke.py` | Pinned selected remote Python modules that the recipe's engine imports pass check_imports in the installed image. Add hash-locked dependencies and build-time native import checks, not runtime installation. |
| `tests/test_recipe_packages.py`, `tests/test_recipe_package_closure.py`, `tests/test_exact_model_file_closure.py`, `tests/test_explicit_hf_file_manifest_closure.py` | Self-contained source/fixture/Model closure, deterministic bytes, no weights/OCI payloads. Include required local inputs and recalculate generated manifests through the builder. |
| `tests/test_qualification_definitions_loader.py`, `tests/test_qualification_index.py`, `tests/test_qualification_case_assertions.py`, `tests/test_qualification_service_aliases.py` | Every recipe is defined/scheduled; ids, aliases, fixtures and assertions resolve. Repair the per-recipe source definition, not generated outputs. |
| `tests/test_family_aware_coverage.py`, `tests/test_qualification_coverage_identity.py` | Coverage equivalence binds actual runtime/topology/content identities. Rebuild coverage and fix false equivalence. |
| `tests/test_prebuilt_images.py`, `tests/test_publish_image_guard.py`, `tests/test_release_bundle.py`, `tests/test_release_assets.py` | Plan/source/build keys, digests, signed index and packaged bytes agree. Rebuild/resolve the correct artifacts before signing. |
| `tests/test_guidance_files.py` | Guidance stays in the allowlisted files and local Markdown links resolve. Consolidate the new rule here or in AGENTS; fix links in the same change. |
| `validate-recipe-library` in `.github/workflows/validate.yml` | Platform source/security/launcher/option compilation. Repair the concrete refusal; producer-only success is insufficient. |

Recipe-specific regressions under `tests/test_<family or adapter>*.py` additionally
pin supported launch, model, cache, protocol and media behaviour. Find them with
`rg -l '<recipe-slug>|<adapter-path>' tests`; run them after touching that runtime,
and run the full suite to catch shared-context consumers. Never treat a narrow
subset as every catalog guard.

The required checks cover:

- Shared `ModelDefinition` and `RecipeDefinition` Pydantic validation for the
  catalog, exact Model resolution, and selected file IDs.
- Complete build and serving-fixture closure. Each recipe archive contains
  exactly one `recipe.json`, its exact Model snapshots, and required supporting
  files; member bytes and manifest digests agree.
- Deterministic package/index generation: CI builds the release asset set twice
  and requires identical bytes, derives family-aware coverage offline from
  the current catalog, and runs the platform validator on the built outputs.
  Contract changes also regenerate/check schemas and examples and exercise the
  standalone package.
- Focused adapter/build tests relevant to the change and `git diff --check`.

Retain representative serving tests. Vision tests need an image; audio/video/3D
jobs need valid input fixtures and output assertions. Decode encoded fixtures
to the payload expected by the runtime and bind the correct input slot.
Health checks alone do not demonstrate inference.

For runtime or filesystem changes, use the applicable container/Spark test
lane. On macOS, inspect `docker context show` and `docker info` and use the
intended OrbStack engine for container checks. Check non-root cache/temp writes
with a read-only root and restart reuse. Real GPU execution, multi-Spark
communication, and model quality require the relevant hardware. Record tests
that could not run and their missing inputs; do not manufacture passing
evidence or add an unrelated formal approval gate.

### Dependency locks and build-time smokes

For an adapter that installs Python dependencies, keep its authored input and
generated hash lock together, target the image's Python/ARM64/CUDA environment,
and install with `--require-hashes`. Regenerate with the command recorded in the
lock header rather than editing resolved hashes. For example,
`adapters/video/moss-vl-realtime/requirements.lock` records its exact
`uv pip compile` invocation for Python 3.12/aarch64 and cu130; it is not a macOS
lock. Include the lock and any vendored wheels/sources in build closure. Existing
creator images may already supply dependencies; do not reinstall a different
engine over them merely to impose a lock format. There is no global lock-format
guard: lock/content checks are recipe-specific, including
`tests/test_remote_code_build_smoke.py` and `tests/test_nemotron_omni_sglang_audio_lock.py`.

Python CUDA runtime bases need the host compiler/libc and Python headers that
Triton uses at inference time (`tests/test_job_runtime_toolchain.py`). Native
CUDA extensions need explicit target architecture/CUDA build verification
(`tests/test_cuda_extension_builds.py`). Audio paths using torchaudio 2.9+ need
a compatible pinned TorchCodec or an evidence-backed unused-audio exemption
(`tests/test_torchaudio_torchcodec.py`). Import native decoders during the image
build to catch missing wheels/shared libraries before qualification. Driverless
build smokes must not replace runtime engine-schema validation: retain the
installed engine's refusal before serving (`tests/test_trtllm_speculative_build_guard.py`).

The hosted image build runs `tools/prepare-remote-code-smoke` against its plan.
For every image whose Models select Python modules, it calls Transformers
`check_imports` inside the recipe's
installed environment on modules named by `auto_map` in the selected `config.json`,
`tokenizer_config.json`, `processor_config.json`, `preprocessor_config.json` and
`video_preprocessor_config.json`,
plus their relative imports (including processor dependencies), before adding
the platform layer or pushing. vLLM and SGLang build registered architectures
natively, so their `AutoModel*` entries are checked only when a base or option
argument selects `model-impl` `transformers` or SGLang's ModelOpt quantization
load (`modelopt-*`, or `quantization` `modelopt`, `modelopt_fp8` or
`modelopt_mixed`); other engines check every entry. An unregistered
architecture's Transformers fallback is not checked and fails at load.
Engine plugins outside that graph belong to the engine's build/import checks.
For every vLLM image, with or without remote code, the installed vLLM also
parses the served Model's pinned `config.json` with `get_config`. The served
path is the one after `serve` or the `model` argument; an argv `/models` is the
endpoint role's first mount, as the Controller rewrites it, and a launcher's own
`vllm serve /models` stays `/models`. It is parsed with each `trust-remote-code`
value its base or option choices can start with, under the interpreter the
adapter's `/opt/vonk/bin/vllm` launcher names, then the `vllm` script's shebang,
then `python3`. Only `get_config` itself fails the build: an exception (with
vLLM's traceback) or a hang after vLLM imported (with its dumped stack, at the
parse bound). An interpreter that cannot run vLLM, or hangs before importing it,
yields to the next, and when none can the build prints a `::warning::`
naming the image, the interpreters and their statuses. A recipe whose served
path mounts no `config.json` is a plan `::notice::` and is not parsed. Downloads
retry transient network, 429 and 5xx errors with backoff; changed bytes fail at
once. (vLLM 0.30.0's own LagunaConfig rejected Laguna's nested RoPE under
Transformers 5.17 at start on hardware; this parse catches that class at build.)
The check fetches only pinned configs and source code and verifies the Model's
hash and size; it loads no weights and requires no GPU. Keep remote-code
dependencies hash-locked
and import native decoders during the Dockerfile build so missing wheels or
shared libraries fail before qualification.

### TensorFold CPU checks on Apple hosts

A Spark-built ARM64 binary may use SVE instructions that an Apple/OrbStack CPU
does not expose. If `tensorfold-native capabilities --json` exits with an illegal
instruction at an SVE instruction, run that same binary under QEMU's SVE-capable
CPU emulation. For the Mia DeepSeek 1017 image, set `RECIPE_IMAGE` to the locally
built image or its immutable published reference, then run:

```bash
docker run --rm --user 0 --entrypoint sh "$RECIPE_IMAGE" -ec '
  apt-get update -qq
  apt-get install --no-install-recommends -y qemu-user >/tmp/qemu-install.log
  runuser -u vonk -- qemu-aarch64 -cpu max \
    /opt/tensorfold/native/bin/tensorfold-native capabilities --json
'
```

This disposable diagnostic container installs the emulator and then executes
as the recipe's non-root user. It does not change the recipe image or production
CPU target. A successful capability response proves the binary can execute this
CPU path; CUDA, model loading, RoCE and inference still require Spark execution.

## 6. Generate, publish, and report

Commit authored sources only. `catalog-index.json`,
`qualification/qualification-index.json`, `packages/` and
`qualification/coverage/` are build outputs: Git ignores them, CI builds and
validates them for every pull request, and publication builds them from the
release commit. Build them locally with one command when a tool or test outside
the suite needs them:

```bash
uv run --python 3.14 --no-project --with-editable contracts tools/build-catalog-index
```

It writes them in place, recording the checkout's `HEAD` as `source_commit`.
`tools/build-family-aware-coverage` then derives the coverage matrix and report
into `qualification/coverage/`. The test suite builds its own copy once per
session in a temporary directory and never reads outputs from the checkout.

For CI, signing, release bundle verification and integration procedure, follow
[AGENTS.md](../AGENTS.md#ci-and-publication). Build assets are not release evidence
until that procedure verifies the published bytes.

### Prebuilt runtime images

Recipe runtime images are built in CI, not on a Spark.
`.github/workflows/recipe-images.yml` runs inside every publication (and by hand
for a backfill: `gh workflow run recipe-images.yml -f slugs=all`, or a list of
recipe ids). Vonk Forge's `scripts/recipe-image-plan` (at the `platform_ref`
`publish.yml` validates the library with: one pin for validation and images)
derives each recipe's executable build key (source bundle,
Dockerfile, pinned base images, build options and capabilities, runtime
adapter) and the exact Spark build flags and platform adaptation stage. Recipes
that build the same adapter directory the same way share one key and one
build. Every publication reconciles all current keys against the registry,
including the first release of a new contract version. For each missing key,
a hosted `ubuntu-24.04-arm` runner builds the image with those flags
and pushes it to the public package
`ghcr.io/<owner>/vonk-forge-recipe-<recipe-id>` as `build-<key>` and
`<version>`; a key another recipe already has is copied, not rebuilt.

`tools/prebuilt-images resolve` then looks up every recipe's `build-<key>`
digest, and the release index records it beside the recipe:

```json
"prebuilt_image": {
  "reference": "ghcr.io/carstvaartjes/vonk-forge-recipe-<recipe-id>@sha256:<digest>",
  "build_key": "<64 hex>"
}
```

The field is optional and covered by the same signed `SHA256SUMS` as the rest
of the index; Controllers that do not know it ignore it. A Controller whose own
key for the revision matches pulls that digest (verified once, at ingress)
instead of building on a Spark, so updating a running recipe needs no Spark
build memory. A recipe whose image is unavailable publishes without this field
and can use a Controller build. Publication reports each such recipe as image
pending in its job summary. The signed bundle includes `plan.json`, binding the
pending keys and recipe content identities to the release. Deferred batches and
the daily scheduled reconciliation build missing keys and republish the pins.
Each build has a 360-minute deadline (a retry starts only if it still fits) and
failures remain visible in the run.
Publication still fails for missing artifacts, mismatched source identities,
stale build keys, invalid digests or an index that differs from a resolved pin.
The plan, index and publication must name the same source commit; verification
runs before signing or changing the release. A backfill run republishes when it
finishes so the index picks up the new digests. The recipe `content_sha256`
identifies its JSON document, not the adapter source or image; an adapter-only
fix changes the package identity and the signed index image reference while the
recipe JSON digest can remain unchanged. The signed index and package manifest
carry `source_bundle_sha256`, the normalized source manifest digest verified at
package ingress and carried into build and qualification identities. A context
change invalidates an unchanged recipe's old build and verdict, including
recipes built on a Spark without a published image. Acceptance revalidates that
context identity so a concurrent library refresh requires a fresh plan.

Roll out this format by publishing the signed library first, then deploying
the Controller that requires the context identity. The preceding Controller
accepts the additional index and manifest fields and derives the same context
identity at ingress; the new Controller rejects a publication missing that
identity. Verify the new library syncs on the running Controller before its
upgrade. Keep one current contract; this deploy order needs no alternate reader.

The qualification authority is derived, never committed. Coverage builds it in
memory from the generated catalog; `tools/build-qualification-authority` can
write it with `--output-dir`. Author its inputs as specified in section 5.

Use [the repository gates](../AGENTS.md#setup-and-producer-gates), then push and
integrate only within the requested scope. Report added/updated/unchanged
recipes, checked/retained pins and reasons, checks, exact source/package/image
identities and remaining hardware work. Include sources that could not be checked in the report.
