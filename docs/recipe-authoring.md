# Create and update recipes

Use this workflow for a new recipe, an upstream refresh, or a targeted fix.
The outcome is a runnable recipe with exact inputs, useful version information,
and a self-contained downloadable package. A repository recipe is expected to
work; authors do not maintain a separate approval or readiness lifecycle.

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
- **Convert:** change document structure while preserving execution intent.

Do not report a conversion or repair as a complete upstream refresh. For a
catalog-wide request, account for every recipe, including unchanged recipes.

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

### Model: the exact files and their capabilities

Reuse an existing exact Model when appropriate. Create a distinct version or
variant record when the underlying model identity changes. Family names and
new versions are data; they do not require new Python classes.

Resolve an immutable source revision and enumerate the files needed to load
the model, including configuration, tokenizer, companion data, and license.
Record actual file paths, IDs, SHA-256 hashes, byte sizes, and purposes. Preserve
legitimate empty supporting files with their real empty-content digest. Do not
invent hashes, sizes, capabilities, context limits, or memory measurements.

Describe capabilities with evidence and keep unknowns honest. A source model's
vision capability does not prove that every engine recipe can serve images.

Preserve upstream license terms, territorial notices, and their source links
as information for the user. License compliance decisions belong to the user;
do not add location setup, territorial admission gates, or tests requiring
Vonk Forge to deny downloads or runs based on geography. A provider may still
require authenticated access to download gated files: preserve those actual
credential requirements and report provider access errors accurately.

### Recipe: a complete way to run those files

Reference exact validated Model documents and their file IDs; do not duplicate
their manifests. Select read-only mounts for the required topology roles.
Choose either a pinned final image or a recipe-owned source build. Include all
required local build inputs, patches, entrypoints, wrappers, and test fixtures.
Model weights and container image bytes stay outside the recipe package.

Preserve launch behavior: executable, ordered arguments, environment, topology,
ports, model aliases, resource envelope, and lifecycle intent. Bind tunable
arguments to the corresponding declared settings instead of maintaining two
copies of a default. Respect the contract's automatic/unspecified settings;
benchmark request counts are not engine concurrency limits. If a wrapper
hardcodes a setting, align its implementation with the declared setting.

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

## 4. Explain what changed

For an update, write a short summary and a few useful highlights: new model
capabilities, corrected behavior, engine compatibility, memory or startup
changes, and changes an operator will notice. Separate Model changes from
Recipe/runtime changes. Use release notes where available and inspect commits
between the exact old and new pins. Link each claim to its supporting release,
comparison, commit, or diff. Label performance claims as upstream claims until
measured here. Do not infer a speedup or quality gain from a commit title.

Store notes in the shared contract's optional changelog metadata when that
field is available. If the current contract does not yet support it, include
the same evidence in the PR and report that catalog notes are pending; do not
invent an unvalidated JSON field or a third authored catalog document. Missing
or incomplete upstream notes do not prevent a usable recipe from publishing.

Keep notes bounded; link to the full history instead of embedding it. If two
pins cannot be compared, explain the source change without inventing a commit
range. A notes-only edit must not invalidate cached model files or runtime
images, although the document/package content digest will change.

## 5. Validate and exercise the result

Use the current [producer validation workflow](../.github/workflows/validate.yml)
for the exact commands and dependencies. Use a writable task-specific cache
for `uv`. Never fall back to older schemas to make a consumer or test pass.

The required checks cover:

- Shared `ModelDefinition` and `RecipeDefinition` Pydantic validation for the
  catalog, exact Model resolution, and selected file IDs.
- Complete build and serving-fixture closure. Each recipe archive contains
  exactly one `recipe.json`, its exact Model snapshots, and required supporting
  files; member bytes and manifest digests agree.
- Deterministic package/index generation and a non-mutating
  `tools/build-catalog-index --check`. Contract changes also regenerate/check
  schemas and examples and exercise the standalone package.
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

Inspect the scoped diff and commit authored sources first. Generate packages
and indexes with `tools/build-catalog-index --source-commit` bound to that
actual source commit, then commit the generated outputs. Do not bind an index
to a commit that predates its authored inputs or invent a digest to avoid
regeneration. Follow the generator and publication workflow if their procedure
changes. Re-run freshness checks before pushing.

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
