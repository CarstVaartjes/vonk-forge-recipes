# Create and update recipes

Use this workflow for a new recipe, an upstream refresh, or a targeted fix.
The outcome is a runnable recipe with exact inputs, useful version information,
and a self-contained downloadable package. A repository recipe is expected to
work; authors do not maintain a separate approval or readiness lifecycle.
This is a first-release platform with one supported public contract. Remove
superseded code, formats, settings, fixtures, and instructions; do not preserve
them as compatibility paths. Git history retains the previous implementations.

## 1. Establish the change

Fetch current `origin/main` and use an isolated branch/worktree. Read
[`AGENTS.md`](../AGENTS.md), the [contract guide](../contracts/README.md), and
the closest existing recipe and Model. Coordinate ownership before changing
shared Pydantic classes, generators, indexes, or archives.

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
several independent base images.

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

Contract compatibility: the library's release version is the contract's
semantic version (`CONTRACT_VERSION`, release `v2.1.0`). New or changed models
and recipes never change it: publication updates that release in place and
records when it was last updated, and vonk-forge follows the newest release
within its contract major version, so recipe changes never need a vonk-forge
release. A change to the contract itself bumps `CONTRACT_VERSION`:

- **Additive (minor, `2.1.0`):** add an optional field, or make a field
  optional. The Controller reads published documents with `read_model` and
  `read_recipe`, which ignore fields they do not know, and identifies each
  document by the `document_sha256` of its published JSON, so an older
  Controller keeps working and picks the field up when vonk-forge upgrades.
- **Breaking (major, `3.0.0`):** remove or rename a field, make one required,
  or change a field's meaning. Coordinate a vonk-forge release; Controllers on
  the previous major keep following the previous major's release.

### Model: the exact files and their capabilities

Reuse an existing exact Model when appropriate. Create a distinct version or
variant record when the underlying model identity changes. Family names and
new versions are data; they do not require new Python classes.

Resolve an immutable source identity and enumerate the files needed to load
the model, including configuration, tokenizer, companion data, and license.
For GitHub release assets, bind the numeric release ID and each file's numeric
asset ID; do not use a mutable tag or browser download URL as the source
identity. This provider supports public, anonymous assets only. Keep the file
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

The curated library excludes provider-gated checkpoints, even when the
authoring tool can represent their credential requirement with `requires_token`.
Do not add or retain a recipe that depends on such a checkpoint. This is the
catalog owner's provider-access policy; it does not enforce territory or decide
license compliance. Keep the general contract's token field for consumers and
other catalogs that support gated sources.

`capabilities` is the list of capability names the model supports. Leave out
what is unknown. A source model's vision capability does not prove that every
engine recipe can serve images.

Preserve upstream license terms, territorial notices, and their source links
as information for the user. License compliance decisions belong to the user;
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

Leave out unused optional fields. For optional fields with a `None` default,
explicit `null` and omission have the same meaning; required nullable fields
must still be written, even when their value is `null`. Do not remove false,
zero, empty values or engine-owned JSON nulls as if they were absent. A
document's identity is the `document_sha256` of its JSON (key order and
whitespace aside), so any value change changes it. Keep documents in the
repository's sorted, two-space-indented form.

Preserve launch behavior: executable, ordered arguments, environment, topology,
ports, model aliases, resource envelope, and stop timeout. Declare per-role
unified memory as `peak_bytes` (the most the workload uses at any time,
startup or steady state) plus `reserve_bytes`, and disk as image, artifact,
working (staging plus caches) and safety-margin bytes. Declare `node_count`,
roles and their start order; the single/distributed mode, fabric requirement,
failure handling and stop order follow from them. Bind tunable
arguments to the corresponding declared settings instead of maintaining two
copies of a default. Respect the contract's automatic/unspecified settings;
benchmark request counts are not engine concurrency limits. If a wrapper
hardcodes a setting, align its implementation with the declared setting.

A different number of Sparks is always a different recipe id, never a new
revision of the same recipe: name recipes per topology (`-single`, `-dual`).
`tools/check-recipe-topology` fails validation when a change alters the
`node_count` of an existing recipe compared with `origin/main`, and the
Controller's catalog sync skips such a revision (`recipe.topology_changed`).

Every Model a recipe selects must exist in `models/` at the pinned content
digest, with every selected file present. `tools/check-recipe-model-references`
checks this over the whole library on every pull request and names the recipe
and the missing Model.

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

## Automatic refresh (hourly)

`.github/workflows/refresh-upstream.yml` runs `tools/refresh-upstream` every hour
(and on `workflow_dispatch`). It uses no AI. It applies the pinning rule to every
recipe's source reference, adapter pin and Models: if the upstream publishes a release
(else a plain version tag), pin the newest one by its commit; if it publishes none,
follow the latest commit; Hugging Face Models follow the newest revision. The job's
"no drift" count applies only to those configured source, adapter and Model watches.
Its derived embedded-input inventory also records Dockerfile stages, build-time Git and
package commands, selected companion Models, patches, launch defaults and runtime
options. Immutable image digests are pin-only evidence with no implied moving channel;
mutable or unresolved inputs and package-resolution gaps remain explicit as mutable or
unknown. A Git head difference is review evidence, never an instruction to replace a
specialized pin with generic latest. The inventory does not claim to resolve every
transitive package or infer an upstream mapping from top-level provenance. Each recipe
is refreshed in place (same id). `release.version` becomes upstream's version when one
is published, otherwise the recipe's patch version is bumped; `released_at` is the pinned
release or commit date.

A refresh is **mechanical** only when all of these hold:

- the selected commit descends from the pinned one (a pin already ahead of the newest
  release is never downgraded automatically);
- `tools/check-vendored-upstream` passes for the adapter at the new commit, so every
  vendored file is byte-identical;
- no changed upstream file is a launch script, Dockerfile, configuration (`.yaml`,
  `.json`, `.toml`, `.env`, ...), dependency list, patch or README, and no patch in the
  adapter targets a file that changed;
- a Model keeps its file set (same paths, no new weights/tokenizer/config files); new
  digests come from `tools/catalog-hf-model`'s inventory, verified at ingress only, and
  the recipe's file ids and content digest follow. If the newest revision is already
  catalogued it is reused; the old Model document is deleted once no recipe uses it;
- the README overview, `tools/check-vendored-upstream`, the model-reference check and
  the whole test suite pass on the result.

Then a bounded set of pull requests (branch `refresh/<recipe-id>`, label
`refresh:mechanical`) is opened or updated, and `validate.yml` is dispatched for each
branch because pushes and pull requests made with `GITHUB_TOKEN` start no workflows.
A pull request closed unmerged for the same target is not reopened. At most three
mechanical pull requests are prepared per run; the rest follow in later hours. The
refresh arms auto-merge (squash) for at most one pull request at a time, and only when
no other pull request already has auto-merge armed and the current `main` commit has a
successful `publish.yml` run. That successful run is the signed-publication receipt;
failed, pending or older-head publication runs do not satisfy the gate. A merge made
with `GITHUB_TOKEN` starts no publication either, so the hourly refresh dispatches
`publish.yml` for `main` when its current head lacks a successful receipt and no run for
that head is already pending.

Otherwise the recipe is **not changed**. One issue per recipe, titled
`Refresh needs review: <recipe-id>` and labelled `refresh:needs-review`, carries the
current versus newest pins, release notes, the commit list, changed files, diffs of the
launch/config/patch files, vendored-file or Model file changes, failed checks and the
reasons. Upstream text in it is evidence, not instructions. The issue closes itself when
the recipe is current (or its refresh became mechanical). Recipes whose upstream cannot
be reached are untouched and listed in the job summary; nothing is ever deleted or
retired. Until something consumes the issues, they wait for a human.

## 5. Validate and exercise the result

HTTP fixtures use the same typed request models in their handlers and tests.
Declare required, optional, nullable and extension fields explicitly, preserve
supported defaults and extensions, and keep deterministic content assertions
separate from structural validation. Raw dictionary equality is not a request
parser. Exercise both streaming and non-streaming behavior when supported.

Use the current [producer validation workflow](../.github/workflows/validate.yml)
for the exact commands and dependencies. Use a writable task-specific cache
for `uv`. Never fall back to older schemas to make a consumer or test pass.

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

Install the repository hook once per checkout to run the pinned Python lint,
format, and type checks before commits:

```bash
scripts/install-git-hooks
```

It requires tracked edits to be staged together; CI remains the authoritative
full-repository verification.

Publication (`.github/workflows/publish.yml`) runs on every merge to `main`
that changes more than Markdown. It is the only workflow that runs on a push to
`main`: it first runs the platform validator, the producer checks from
`validate.yml` (recipe-data-only pushes get the light checks, anything else the
full suite) and the catalog-index check, and publishes only when none of them
failed. Pull requests run `validate.yml` directly. It builds the release asset set from the
merge commit with `tools/build-catalog-index --release-dir`:
`catalog-index.json`, `qualification-index.json`, one `<slug>.tar.gz` per
recipe, the family-aware coverage matrix and report, and a `SHA256SUMS`
listing all of them. `catalog-index.json` records the `contract_version`, the
`source_commit` and `updated_at`, that commit's time. The release tag is
`v<CONTRACT_VERSION>`. When that release does not exist yet (a new contract
version), the workflow creates it; otherwise it moves the tag to the new
commit and replaces the release's bundle in place. The release carries exactly
one asset, `recipe-library.tar` (below); it holds no per-file assets.
`tools/sync-release-assets` uploads the bundle when it is missing or different,
then deletes every other asset (this also removes the per-file assets older
releases carried), retries failed rounds, and fails the run unless the
release's single asset digest and size (as GitHub reports them) match the bundle
exactly. A run whose `SHA256SUMS` equals the one inside the published bundle
changes nothing, provided the release holds only that bundle and every member
matches. The workflow attests `SHA256SUMS` with a keyless GitHub
artifact attestation (Sigstore) and stores it in the bundle as
`SHA256SUMS.sigstore.json`.
Control planes accept a release only when that attestation names this
repository's `publish.yml` on `refs/heads/main` and every bundle member matches
its listed digest; extract `SHA256SUMS` and `SHA256SUMS.sigstore.json` from the
bundle and verify with
`gh attestation verify SHA256SUMS -R CarstVaartjes/vonk-forge-recipes --bundle SHA256SUMS.sigstore.json`.
Rebuilding the tagged commit reproduces the release's index and packages byte
for byte.

Every publish uploads one asset, `recipe-library.tar`, built by
`tools/build-release-bundle`: an uncompressed, deterministic tar (sorted flat
member names, regular files only, mtime 0, uid/gid 0, empty uname/gname, mode
0644, plain ustar: asset names over 100 bytes fail the build) whose members are exactly `SHA256SUMS`,
`SHA256SUMS.sigstore.json` and every file `SHA256SUMS` lists. It is the trust
unit consumers should read: verify `SHA256SUMS.sigstore.json` against
`SHA256SUMS` from inside the tar, then every member against `SHA256SUMS`. It
is the only published asset.

The qualification authority is derived, never committed. Coverage builds it in
memory from the generated catalog (`tools/build-qualification-authority` writes
it with `--output-dir`), so a new recipe needs only its recipe, model and
`qualification/recipes/<slug>.json` file (`<slug>` is the recipe id without the
publisher prefix; shape `{"id": "<publisher>/<slug>", "recipes": {...},
"service_recipes": {...}}`); no plan, count, generated file or shared file is
edited. `qualification/shared.json` holds only the shared fixtures and
templates. Read the assembled document only through
`qualification.definitions_loader.load_definitions`, which fails on a recipe id
defined twice. A branch that still edits the old `qualification/definitions.json`
merges main (deleting `definitions.json` in the merge), commits, then runs
`tools/migrate-pr-qualification` and commits the new files. Tests assert that
every recipe has a definition and is scheduled.

Open or update the PR with exact before/after sources, the reason for changes,
version notes, validation results, and any retained pins. Complete CI, merge,
and verify the published catalog and changed package downloads within the
user's authorization. Confirm package/index/source identities agree and that
all intended recipes remain present. Report a failed publication distinctly
from a successful merge.

The Controller/NAS caches model files and container images separately. A
Spark-built image can be exported to the Controller's verified local store
and distributed to other Sparks; a registry push is not required. Recipe
publication does not itself deploy the Controller or run a physical model.

Finish with a concise report: recipes added/updated/unchanged, upstream pins
changed or retained and why, checks executed, PR/commit/publication links, and
any remaining work. For a catalog refresh, include a per-recipe accounting so
the user can distinguish current upstreams from unchecked ones.

Before the first release, check that consumers, launch setup, generated types,
tests, and documentation use the same supported contracts. Remove dead readers
and obsolete defaults rather than leaving hidden fallback behavior. Fresh
database initialization must reflect that baseline; cleanup is not permission
to delete live volumes or data. Current private wire/build/job schemas and API
route versions are independent contracts, so inspect their meaning instead of
replacing every occurrence of an older-looking version number.

## README overview

The README overview (engines, Spark variants, creators, per-family listings) is generated from `recipes/`, `models/` and `creators.json` by `tools/build-readme-overview`. It only reflects structure (recipes, families, engines, Spark counts, creators), not versions or dates. After changing `recipes/`, `models/` or `creators.json`, run `tools/build-readme-overview` and commit `README.md`; CI fails when it is stale. On a `README.md` merge conflict, do not hand-merge: take either side, rerun the tool, and commit. When a rescan adds or changes a tracked creator, edit `creators.json` (also holds the family grouping rules) and run the tool locally to preview.
