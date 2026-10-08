# Hardware sweep

`tools/sweep-recipes` tests as many recipes as possible on the real Sparks,
fast and unattended, and records what it finds. It drives the fleet only
through `vonkctl` (JSON mode), needs no Controller credentials of its own, and
runs from the operator's machine. The entry point uses `uv` to supply Python
3.14 and pinned Pydantic for the shared sweep observation contracts; the code
is `spark_sweep/`. `tests/test_sweep_*.py` run it against a fake `vonkctl` and never touch a fleet.

It lives in this repository because what to test is already here: the
reviewed smoke cases and fixtures in `qualification/shared.json` and
`qualification/recipes/<slug>.json`, and the results log has the same shape as
the platform's `vonk-fleet-qualify-campaign`. That tool runs one batch at a time
through the Controller API and needs the derived authority; the sweep adds
prefetching, independent lanes and resumability, and talks to the fleet only
through `vonkctl`.

## Run it

```bash
tools/sweep-recipes plan                       # read-only: unique bytes, download time, order
tools/sweep-recipes run --yes --restore-owner 2 \
    --client-key-file ~/.vonk/sweep.key        # stops what the Sparks run; loads profile 2 at the end
tools/sweep-recipes status                     # the live status page
tools/sweep-recipes report                     # markdown + JSON report from the state file
tools/sweep-recipes export-evidence            # qualification/hardware-evidence/<slug>.jsonl
tools/sweep-recipes file-issues --dry-run      # one `hardware-test` issue per failure (or: run --file-issues)
```

Ctrl-C cancels the current load, hands the recipes back to the queue and exits
130. Rerunning the same command resumes. `--only`, `--limit`, `--spark`,
`--seed-list FILE` (slugs to test first, for example models already
prefetched) and `--boost WORD` (default `glm-5-3`) narrow or steer a run.

The state directory (default `~/.vonk-sweep`, `--state-dir`) holds:

| file | content |
|---|---|
| `state.json` | the resumable record: results, slots, download operations, pins, learned timings |
| `results.jsonl` | one line per outcome, appended |
| `status.json`, `status.md` | live: lanes, queue, throughput, ETA, counts, failure clusters |
| `report.json`, `report.md` | the final summary |
| `evidence/<slug>-<n>.json` | `vonkctl fleet evidence` bundles of failed operations |

## Lanes: one sweep profile, not one profile per lane

Profiles cover the whole enrolled fleet: a Spark with no assignment is idle and
a load stops its workload (`docs/runbooks/vonkctl.md` in the platform: "Profiles
cover the entire enrolled, non-revoked fleet… an unassigned Spark is explicitly
idle"; the preview scope is the roster, `idle_node_ids`). A profile per lane
would therefore stop the other lane. The sweep uses **one profile** (default
`--sweep-profile 10`) holding every lane's assignment. A lane swap is
`profile remove <old>` plus `profile add <new> --spark X --as <alias> --state
running`, then `profile load --review`. Before loading, the review must show
no `stop` effect for a lane that is still running; then `profile load --yes
--detach`. A fresh whole-fleet application adopts equivalent unchanged loading
assignments and fills a free Spark while the other lane copies or smokes. The
review must bind each borrowed application, plan digest, ordinal and complete
assignment/node scope; a missing binding defers the new admission. A swap never
restarts the other lane. Duals use both Sparks and run as a
window (several cached duals back to back) rather than interleaved with
singles. Each single recipe owns one Spark; dual recipes are scheduled only when
both Sparks are free.

In-flight applications are persisted in a `loads` map keyed by request UUID.
Old state files with one `load` are adopted without changing that UUID. Admission
and progress replies in `observing` or `backoff` keep being polled. Operator waits
and removal gates end the owned load and requeue its recipes without blame.
Downloads and lane observations have a 24-hour total budget; an unreadable or
stalled download has a one-hour observation budget. An owned download exceeding
that budget keeps its original request and resumes paced observation after
cooldown. The read budget never authorizes cancellation or a replacement download.

Concurrent accepted snapshots use the Controller's bound whole-assignment
adoption contract (`effects.adopted`), while original applications keep their
children and progress. The sweep observes each original request independently.
An application whose own lane has ended still retains its observation receipt
while an adopted lane is active: cancelling that aggregate application would
also stop its adopted effects. These observer receipts never occupy a Spark or
prevent scheduling. This coordinator must run after the companion Controller
adoption change is deployed; without its preview bindings, it safely defers an
overlapping load.

**Aliases.** `--as` is the assignment name, the lane's client-facing model name.
The profile contract takes a lowercase identifier (`ENDPOINT_ALIAS_PATTERN` in
the contracts package), so a reviewed `service_recipes.alias` in
`qualification/recipes/*.json` must already be one: the index build and the
tests reject any other spelling, and the sweep derives its name from it the way
the Controller derives a default (lowercase, separators to `-`). A profile
cannot hold two running assignments of one name, so a variant that shares its
alias with a lane already in the profile (the Inkling duals) gets a short
stable suffix. A profile edit the Controller refuses is named by its step
(`profile-edit`): a refused sweep field (name, Sparks, state) pauses the sweep
as an infrastructure problem, and only a refused recipe field (`recipe_selector`,
`option_choices`, `model_variant`) fails the recipe, as `recipe-data`.

## Prefetch

The next models are downloaded to the NAS while the Sparks test.

* **Per model, not per recipe.** Recipes are grouped by the exact set of model
  digests they need. A model is downloaded once; every recipe using it is then
  tested back to back, engine and base image grouped, so downloads, installs and
  compile caches are reused.
* **Value per byte.** Groups are ranked cached first (small first), then by
  recipes unlocked per new byte (a recipe needing models A and B is unlocked as
  soon as both are present, so shared parts cost nothing twice). Owner-priority
  families (`--boost`, default GLM 5.3) and `--seed-list` recipes are boosted.
  `tools/sweep-recipes plan` prints this order, the unique model bytes (the
  library's `disk_bytes` summed over distinct model digests; an upper bound if
  variants share files) and the download time at `--rate-mb-s` or the rate
  measured in a previous run.
* **Already there is present.** Cached and in-flight (`preparing`) recipes and
  models are never requested again; a download that someone else started is
  followed until it lands. Request keys are deterministic, so a restart
  reconnects to its own operations.
* **Download done means the assets are there.** A recipe is ready to place when
  its download operation succeeded, or the library lists it cached, or the
  library's own assessment says `Exact NAS assets: ready`. It never depends on
  fleet fit or readiness: those cannot hold while someone else's workload fills
  the Sparks, and `profile load --review` on the sweep profile decides fit at
  placement. A finished download is not requested again unless a library read
  made after it still says the cache is empty, and then at most twice.
* **The library is read a page at a time.** The listings are cached; a refresh
  reads one page per pass of the loop (several at start), retries a failed or
  timed-out page, and restarts from the first page when the Controller
  invalidates the cursor. Scheduling reads the cache, so one failed page never
  stalls it, and rows missing from a complete pass are dropped.
* **Bounded.** `--max-model-downloads` (default 3) operations that fetch a
  model, `--max-image-pulls` (default 2) operations that only pull an image,
  and `--nas-budget-tib` (default 2) for models that are downloaded or
  downloading and not yet tested. One lead recipe fetches a model; its siblings
  follow once it is cached and need only their image. `recipe download` fetches
  the model (Hugging Face) and the runtime image (GHCR) as children of one
  operation, so the two sources already run concurrently; there is no separate
  image verb to schedule.
* **NAS free space is not exposed.** Nothing in `vonkctl` or the Controller API
  reports the NAS's free bytes (only per-Spark disk, and `free_bytes` or
  `shortfall_bytes` on a failed operation). The byte budget is therefore the
  operator's, and a `free_space` or capacity failure also pauses new model
  downloads for ten minutes.
* **Eviction pins.** The Controller evicts unused models LRU but "nothing is
  removed while a saved profile (loaded or not) points to it". The sweep keeps a
  never-loaded **pin profile** (default `--pin-profile 13`) naming one recipe
  per model set that is downloaded or downloading and not yet tested
  (assignments with `--state installed`), within the byte budget, and removes
  it once the set is tested. If the Controller refuses the pin edits the sweep
  carries on and says so in the status page.
* **Measured throughput.** The summed `smoothed_bytes_per_second` of the
  active operations feeds the ETA on the status page.

## What is tested

* chat, code, reasoning and vision recipes: a streamed completion measures time
  to first token and tokens per second, then the recipe's reviewed qualification
  cases run with their assertions (the image cases carry their small PNG). A
  recipe without reviewed cases runs its own declared `validation.serving` checks.
  The sweep proves a recipe **runs**. A case's `assertions` are functional and
  gate it: the server answers (HTTP 200), `model` is the alias, one choice, a
  well-formed number, colour word or tool call, no leaked think or control tags.
  Its `quality_assertions` name the exact expected answer (391, Amsterdam, red):
  a miss is recorded as `quality: {case, expected, got, ok}` in the result and the
  recipe still passes, reported as "ran; N quality notes" on the status page and
  in the report, never filed as an issue. Answer quality, like speed, belongs to
  the model and the recipe's creator. `export-evidence` keeps the notes. Results
  that failed a smoke assertion under the older rules are requeued once.
* generation and other non-OpenAI recipes (image, video, audio, 3D jobs):
  readiness only (the run is healthy and its route published), recorded as
  `readiness-only`. Their reviewed job fixtures are not submitted by the sweep yet.

Requests go to the endpoint `vonkctl profile endpoint` reports, with the key in
`--client-key-file` (`--ca-file` or `--insecure-tls` for the gateway's
certificate).

## Timeouts, failures, fixes

* Load timeouts are learned per engine from completed loads (seconds per GiB of
  model, times 3, between 20 minutes and 2 hours); before any sample, 60 minutes
  allow a first-start compile. The limit only runs once the bytes are in place
  (install, start, readiness): while the profile application is distributing or
  copying (child phase `target-copy`, `transfer`, `model-download`,
  `container-download`, or a copying operation) it does not count, and the learned
  timings leave the copy out. During a copy only a lack of progress fails the
  load: no change in the completed bytes for `--copy-stall-minutes` (default 10),
  which is class `copy-stalled`, deferred and requeued on a platform change or the
  watch cooldown.
  Load timeouts recorded before this rule (copy time counted) are retried once.
* While the Controller holds the application back with error blockers (capacity, stale
  inventory, a phase it keeps retrying; `progress.blockers`), those seconds are not load time
  either, and a hold that lasts `--blocked-minutes` (default 15) defers the lane as class
  `admission-stalled` (phase start). A proven change to the typed fault owner or watch cooldown schedules a retest; it
  is never recorded as an engine `load.timeout`.
* A restarted sweep adopts a load it submitted itself: the saved lane slot and the
  saved application are picked up, nothing is cancelled or placed again, and the
  time the sweep was away counts neither as load time nor as a stalled copy.
* Failures carry a phase (download, review, build, install, start, readiness,
  smoke, timeout) and a class (network, oom, capacity, build-policy,
  model-integrity, …). Only network-like failures are retried, once. A
  typed model-integrity download failure fails the current attempt; every sibling
  receives its own attempt and evidence. Unknown platform outcomes are deferred.
  Failures with the same normalised signature form
  one cluster in the report, so a fix lands for all of them.
* Each result is bound to the recipe document digest (`content_sha256`). When a
  recipe's digest changes (the hourly refresh, a fix), a failed recipe is
  requeued and a passed one is retested last (`--no-revalidate` keeps old passes).
* Each new result records the authenticated deployed API source when known
  (`controller_release` in `results.jsonl`, `release` and `release_authority`
  in state). `vonkctl platform` reports packaged API and fresh worker identities;
  signed publication and installed CLI facts are recorded separately. A signed
  publication alone never proves deployment or resets a deferred result. Historical
  results recorded from a publication proxy remain unverified and group as unknown.
* Automatic recovery requires an exact typed fault with a proven changed owner
  (source and contract fingerprint). Generic capacity, application and preparation
  faults retain their normal scheduled retest; missing, stale or mixed provenance
  cannot manufacture a recovery trigger. Retests append history and preserve
  cumulative attempts, cause, evidence, cooldowns and existing request identities.
  Active preparations reconnect with their original UUID. A new download attempt
  requires a cooldown-qualified retest (or the bounded transient retry) and a fresh
  exact terminal-failure receipt;
  the complete previous attempt remains in download history and cache bytes are reused.
* `run --retry-failed [--only SELECTOR]` requeues failed recipes (all of them, or
  those whose selector contains the text), whatever the cause. The status
  page groups failures by Controller release, newest first.
* A recipe that does not fit the declared memory still gets its own review: the
  platform's admission decides, and a blocked review is a `review` failure.
* Recipes needing more Sparks than the fleet has are `skipped`, not failed.

## Owner safety

Profiles 1-3 are never written (the wrapper refuses it, and `vonkctl run` is
never used); `--sweep-profile` and `--pin-profile` cannot be 1-3. The sweep
refuses a profile that already holds assignments it did not create. At start it
stores a read-only export of profiles 1-3. If an owner profile load starts, the
sweep submits nothing until it ends plus a hold (30 minutes), and requeues its
in-flight tests without blame. `--restore-owner 2` loads profile 2 at the end.
`run` refuses to start without `--yes`, because the sweep replaces whatever the
Sparks run: at start it loads the empty sweep profile once (again after an
owner load, or when a review is blocked by a workload that is not ours), so
every review sees idle Sparks. Every request key carries a
per-state-directory nonce, so a fresh state directory never replays an older
run's requests, while a resumed run reconnects to its own. Downloads that
someone else started (library state `preparing`) are waited for, not repeated.

## Infrastructure errors, the client, and one sweep at a time

* A client, protocol or transport error is not a recipe's failure:
  `controller.protocol_invalid`, `controller.transport_*`, a
  `controller.unavailable` without a structured platform reason, a timeout or
  unreadable output, a command the installed `vonkctl` does not know, and a
  generic `controller.invalid_request` from a review or profile edit. The sweep
  pauses that kind of work, retries with backoff (30 s doubling to 10 min) and
  shows the problem on the status page until the Controller answers. A
  structured refusal such as `dockerfile.heredoc_forbidden: ...` still fails
  its recipe.
* At start `vonkctl --version`, `vonkctl platform` and `vonkctl update` record
  installed, deployed and signed-publication facts independently. The signed
  updater installs only a client whose packaged contract matches the observed API;
  unknown or different compatibility remains pending. `--allow-version-skew`
  keeps the installed client with a warning. Unknown provenance does not pause
  otherwise valid Controller scheduling.
* A state directory has an exclusive lock (`run.lock`, an `flock` that names the
  holder's pid and host and disappears with the process). A second `run` on the
  same directory refuses to start.
* SIGINT and SIGTERM stop the sweep even when it was started with SIGINT
  ignored (a background job of a non-interactive shell inherits that, and Python
  then installs no handler). Every `vonkctl` call runs in its own session and is
  killed with its process group on an interrupt or timeout; smoke requests run on
  daemon threads, so a hung request cannot keep the process alive.

## The sweep's own loads

Every load the sweep submits itself (clearing the fleet, placements, the final stop,
`--restore-owner`) has its request key recorded in `state.json` before the call and its
application id after, so the owner guard never mistakes it for an owner's load, in this
process or the next. A later run that finds the previous run's restore still
coming up waits for it ("our own load") but does not hold or requeue anything; only
an application the sweep did not submit counts as the owner loading a profile.

The sweep places only on online Sparks. Empty takeover loads whose preview
would stop an offline or unselected Spark are skipped. Clearing admission gets
three attempts spaced by 30 seconds; the budget survives restart. After that,
free online lanes are selected from a fresh fleet observation and submitted
through the normal Controller review. Unknown cleanup bookkeeping cannot reserve
every lane forever. Idle alerts include the clearing blocker and offline Spark names.

The current CLI applies profiles to the whole fleet and offers no node-scoped
load. Placement loads can therefore still carry stops for offline workloads;
excluding those effects requires a Controller/CLI scope contract. The sweep
neither waits for offline-only cleanup adoption nor counts it as lane occupancy.

Cleanup observes the original accepted application's canonical per-effect graph.
An observation timeout never cancels it or invents a new clearing request. Each
online stop scope remains occupied until its exact successful typed stop receipt
while clearing observation is within its budget; absence from `fleet.loaded`
never rewrites a receipt. After budget exhaustion, observed fleet occupancy and
a fresh Controller review govern placement. A healthy independent lane can then
receive a fresh whole-fleet admission that explicitly adopts every remaining
pending cleanup on online lanes. Borrowed cleanup retains its original root, plan, ordinal,
request and child operation, and keeps the aggregate observer alive. A dual stop
scope remains atomic even when one member is healthy.

A genuinely terminal failed or cancelled cleanup retains its physical-claim
history. Normal admission may accept a new stop only for the exact residual run
and topology; only its matching successful stop receipt reconciles the older
unconfirmed effect. Unreadable or changed bindings defer safely. Stop and restore
loads remain ordinary authorized whole-fleet intents with durable request keys.

## Evidence

`results.jsonl` follows the platform campaign's log shape (`recorded_at`,
`authority_id`, `batch`, `recipe`, `step: "smoke"`, `status`, `run_id`,
`node_ids`, `result: {endpoint_alias, cases[]}`), so
`vonk-fleet-qualify-campaign status` reads it for the same authority id. Each
line adds the hardware facts: `phase`, `failure_class`, `signature`, `cluster`,
`content_sha256`, `recipe_revision_id`, `library_commit`, `engine`, `timings`
(download, load, smoke), `result.perf` (`ttft_ms`, `tokens_per_second`) and the
local `evidence` bundle. The qualification definitions say what to test and
have no slot for results, and the derived authority is never committed, so
`export-evidence` writes the latest outcome per recipe document to
`qualification/hardware-evidence/<slug>.jsonl`. The loader does not read that
directory; a test checks that every file names a real recipe.

## Assumptions to confirm on the first real run

The first real run should use `--limit 2` and watch the first load and review.
Each tick makes roughly 8-12 `vonkctl` calls (owner polls, fleet, application and
download progress, pin edits), so the real cadence is 15-30 seconds, not
`--poll-seconds`.

The tests use a fake `vonkctl` modelled on the Controller's OpenAPI schemas and
on real library and fleet output. Bound adoption links and staggered lanes are
covered by the fake; their physical acceptance still requires the companion
Controller deployment. Also not yet seen on a real fleet: that the
Controller accepts many `--state installed` assignments in the pin profile; and
the exact gateway `api_base` and key handling (the key is sent as
`Authorization: Bearer`, the usual OpenAI-compatible form; the platform's own
campaign smoke sends no header).

Not built: Spark-side staging of a queued recipe before its lane is free,
submitting the reviewed
job fixtures for generation recipes, and reading the NAS's free space (not
exposed).

## Failure evidence and tool outages

Every failed result row carries a structured `evidence` object (phase, code, detail,
plus: the smoke case id, HTTP status, a redacted response excerpt of at most 2 KB and the
offending assertion value; or the Controller's reason and the operation id with the
`vonkctl fleet evidence <op>` command for load, start, download and build failures).
The `error` text summarises it and `status.md` shows one evidence line per failure
cluster. The downloaded bundle path is `evidence_bundle`. A missing or unspawnable
`vonkctl` (for example during `vonkctl update --apply`) is infrastructure: the sweep backs
off and retries. An unexpected exception in a pass is logged as an infrastructure event
and the loop continues; only an explicit stop, SIGINT or SIGTERM ends it.

### Recovery outcomes

A typed recipe-data failure or failed output assertion records `failed`. Unknown
Controller, transport, HTTP and process outcomes record `deferred` with evidence;
they never count as a recipe defect or inherit a sibling defect. `--watch SECONDS`
keeps the coordinator alive after a pass and retests deferred/failed results after
24 hours. With `--watch 0` (the default) one pass ends after its owned attempts
settle. A stuck owner intent ends the sweep after its observation budget, leaving
its owned lanes released for a new operation. Unreadable state is preserved beside
`state.json` and disposable bookkeeping is rebuilt. Client skew applies the
accepted CLI update automatically and retries a failed update with backoff.

### Gone cleanup reconciliation

An exact Controller not-found receipt is reconciled against a fresh, structurally
validated fleet observation. If the target runs are absent and their nodes are
online, the local clearing record ends with `CleanupEndReason.TARGET_ABSENT`.
The shared Pydantic consumer models live in
`contracts/src/vonk_forge_contracts/sweep.py`; these local records do not change
the public Model/Recipe document contract or assert a successful Controller stop.
Ended records survive restart and contribute neither occupancy nor required
cleanup adoption to a fresh load. Unavailable observations retain the original
request and use persisted exponential delays of 30 to 600 seconds. A new load
still requires the Controller's current admission review.

The former clearing counter was an infrastructure submission-error counter,
not an observation counter. Progress observation kept running without updating
or clearing that old entry. Reconciliation now removes the stale entry and its
backoff; the local observation budget records successive unavailable reads.
