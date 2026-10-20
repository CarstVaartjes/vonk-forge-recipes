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

## Verdict identity

A verdict records the accepted recipe digest, resolved runtime image build key
and image digest, and model digests. Reconciliation, runtime preparation reuse,
model removal and evidence export compare this shared identity. Image-only
changes requeue the recipe, newest publication first. Status and reports retain
both the tested identity and the previous verdict when a retest is queued.

The signed catalogue's `prebuilt_image` pin must be exposed in the Controller's
recipe library row for an exact image identity. When a Controller omits it,
the image stays explicitly unknown and the accepted library commit serves as a
conservative invalidator: any new library publication triggers revalidation.
It cannot distinguish republished images at the same library commit. Existing
saved verdicts without identity requeue failed/deferred recipes; passed recipes
are preserved with `--no-revalidate`. No sweep state needs manual editing.

## Lanes: one sweep profile, not one profile per lane

Profiles cover the whole enrolled fleet: a Spark with no assignment is idle and
a load stops its workload (`docs/runbooks/vonkctl.md` in the platform: "Profiles
cover the entire enrolled, non-revoked fleet… an unassigned Spark is explicitly
idle"; the preview scope is the roster, `idle_node_ids`). A profile per lane
would therefore stop the other lane. The sweep uses **one profile** (default
`--sweep-profile 10`) holding every lane's assignment. Each new single lane is submitted as its own application, retaining the same
whole-fleet profile and bound adoption of unchanged lanes. A lane swap is
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

Placement consumes the Controller review's disk decision, which counts free
space, reservations, reusable objects and evictable installations. A typed
insufficient-disk decision never submits that lane, even when the aggregate
review allows waiting for preparation. Its node scope and evidence survive
restart until the next check; other placements are reviewed again immediately.
The next placement can use another Spark, and the original is reconsidered after
the check interval without spending a recipe attempt.

Single placement prefers a free Spark holding every model digest, then the exact
installed recipe revision (and its built runtime), then the usual usage/id order.
These hints join the existing fleet `installed` rows to the current recipe
catalog; incomplete, unknown and multi-node installations provide no model hint.
The fleet does not expose model-store or image digests, so shared base images
are not treated as proof of a shared runtime. The first fitting queued recipe
always places immediately. Later variants may yield once within the placement
call so the other Spark can take its next model or download; a fallback pass
fills any remaining lane. No affinity reservation survives the call.

In-flight applications are persisted in a `loads` map keyed by request UUID.
Old state files with one `load` are adopted without changing that UUID. Admission
and progress replies in `observing` or `backoff` keep being polled. Operator waits,
removal gates and every other lane wait use a bounded local observation budget.
Expiry releases the lane and requeues the case for a later fresh review. A download whose own progress (bytes,
items, phase, last-progress time) is silent for an hour is shown as stalled and keeps
its original request; nothing cancels or replaces it, and a download someone else
started is followed until it lands.

Controller application stall alarms retain their original request for observation,
but release their local lane when its budget expires.
The status alerts name the application ids; independent lanes continue through
fresh canonical review. An unreadable application receipt consumes that lane's wait budget. Owner safety is observed every tick, including
while paused, so an old unreadable owner profile cannot latch a global hold.
See the [global gate audit](sweep-global-gates.md) for the complete safety boundary.

Concurrent accepted snapshots use the Controller's bound whole-assignment
adoption contract (`effects.adopted`), while original applications keep their
children and progress. The sweep observes each original request independently.
A snapshot still waiting for admission, with no child operation or accepted
effects and a measured storage shortfall, releases its local lane slots for
re-planning. Its original receipt stays observed; the next reviewed snapshot
omits the blocked assignment and supersedes that intent. The blocked recipe
waits until its next check while a fitting sibling gets a separate application.
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
* **Fleet model removal.** After every catalog recipe referencing a model has
  a passed or terminally failed verdict for its current content, the sweep
  reviews and requests `model remove DIGEST --fleet`. All model roles count;
  pending, retrying, deferred, skipped and stale-content results keep their
  models. A complete catalog and readable owner profiles are required. Every
  saved owner profile number is protected, including 20 and higher; only this
  sweep's own sweep and pin profiles are excluded from that owner check.
  The Controller still owns reference and shared-byte retention.
  `state.json` and `status.json` retain the review, request key and receipt;
  `status.md` shows the outcome, request key, last message and next check.
  Keys are saved before submission and reused after a disconnect or restart.
  Terminal failures re-plan with a fresh review and a new key. Uncertain
  outcomes keep reconnecting to the original key until the Controller returns
  a receipt or a typed terminal answer. Checks use capped exponential backoff
  with jitter (at most one hour) and a 30-second CLI timeout; no attempt,
  observation or elapsed-time budget ends reconciliation. If a consumer becomes
  pending or retrying again, or an owner profile starts using the model, the
  sweep drops its pending removal intent immediately. This does not cancel
  remote work already accepted by the Controller. Removal never
  reserves a lane or delays sweep completion: unfinished cleanup remains in
  state for a later invocation, and a refused removal never stops recipe work.

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
  Every pass also records `memory: {declared_bytes, nodes: [{node_id, used_bytes,
  alone}]}` from the fleet telemetry while the run serves (total less available,
  the OS included). A run alone on its Spark that holds more than its declared
  envelope plus a 6 GB OS allowance gets a `memory-envelope` quality note: the
  envelope is judged by what the run holds, not by the engine's settings.
* generation and other non-OpenAI recipes (image, video, audio, 3D jobs):
  readiness only (the run is healthy and its route published), recorded as
  `readiness-only`. Their reviewed job fixtures are not submitted by the sweep yet.

Requests go to the endpoint `vonkctl profile endpoint` reports, with the key in
`--client-key-file` (`--ca-file` or `--insecure-tls` for the gateway's
certificate). The key file is read on every request. A request the gateway
refuses with 401 or 403 is the sweep's own key failing (for example an expired
one), never the recipe: the sweep reports an infrastructure problem and smokes
again with backoff within its phase budget, then releases and requeues. It carries
on as soon as the file holds
a working key (`vonkctl key roll NAME --output NEW`, then move `NEW` over the key file).

## Timeouts, failures, fixes

* The sweep follows the Controller's verdicts, while bounding local lane occupancy.
  `bounded_wait` is the sole `waiting_on` producer. Every wait uses the lane budget
  from admission (or the smoke phase start); changing reasons, progress signatures,
  heartbeats, backoff and process restarts cannot extend that deadline. Expiry
  records `sweep.wait_budget_exhausted` with the typed reason, detail, start,
  deadline, observation time and last Controller receipt. The local lane releases,
  no recipe attempt is spent, and the case requeues after `defer_delay`.
  Fresh work goes first; deferred cases are reconsidered newest revision first.
  Original remote receipts remain observed; fresh canonical review decides which
  effects can be adopted or superseded. Expiry never asserts remote cancellation.
  `status.md` marks any over-budget lane explicitly until it releases.
  Own-work inactivity still stalls independently; moving copies may exceed a load
  budget, but a dependency wait during copying cannot hold the lane past it.
* The window does not run while the profile application is distributing or
  copying (child phase `target-copy`, `transfer`, `model-download`,
  `container-download`, or a copying operation) it does not count, and the learned
  timings leave the copy out. During a copy only a lack of progress stalls the
  load: no change in the completed bytes for `--copy-stall-minutes` (default 10),
  which is class `copy-stalled`, observed once more and then deferred, and requeued
  on a platform change or the watch cooldown.
  Load timeouts recorded before this rule (copy time counted) are retried once.
* Preparation blockers use the shorter of `blocked_release_seconds` (default
  five minutes) and the lane budget. Other application waits use the lane budget.
  Download queue/backoff and unreadable-operation waits use a one-hour observation
  bound. Downloads own no Spark lane: expiry records a deferral and keeps observing
  the same remote request with cooldown, preserving download concurrency accounting.
* After the load succeeded, a healthy run whose route the Controller has not
  published yet, and an endpoint not yet published, are waits; a run that is missing
  or unhealthy is the recipe's own work and stalls after the window (`run.not_published`).
* A restarted sweep adopts a load it submitted itself: the saved lane slot and the
  saved application are picked up, nothing is cancelled or placed again, and the
  time the sweep was away counts neither as load time nor as a stalled copy.
* Failures carry a phase (download, review, build, install, start, readiness,
  smoke, timeout) and a class (network, oom, capacity, build-policy,
  model-integrity, …). `disposition` decides what an outcome spends: a `WAIT` (a
  busy 429/503 smoke, a refused client key, a full NAS, any vonkctl error that is not
  an explicit recipe-data code) spends no attempt and the recipe is asked again; an
  `OBSERVE` (an uncertain outcome: a stalled copy, a transport failure, a smoke socket
  timeout, a shared load whose failure names no recipe) is retried once; a `FAIL` is
  recorded at once with its evidence. A refused load is recorded, never resubmitted
  with the same plan. Every outcome, wait and requeue writes a `results.jsonl` row
  with the Controller document it ended on; the log is rotated, never trimmed. A
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

**Testable is the enrolled topology; an offline lane is a wait.** "Not testable on
this fleet" is decided from the Sparks the `fleet` listing enrolls (online or not):
a recipe needing more Sparks than are enrolled is skipped, with the listing as its
evidence. A recipe needing N Sparks with N enrolled but fewer online stays pending
("waiting for 1 more lane" in the queue) while the online lanes carry on with work
that fits them. An enrolled Spark that goes offline keeps its lane in the status,
shown with the Controller's reason and last-seen time; each offline/online change
writes one `lane` row with the fleet facts. Skips an older sweep wrote from a
momentary online count heal to pending on the next pass. A run its failed
application left behind on that Spark is stopped by the next sweep-profile load once
the Spark answers (its assignment is already gone from the profile).

The sweep places only on online Sparks. Empty takeover loads whose preview
would stop someone else's workload on an offline or unselected Spark are skipped.
A stop of the sweep's own workload on an offline enrolled Spark does not hold the
clearing: the Controller queues it and delivers it when the Spark answers, while
the clearing frees the online lanes. Clearing admission gets
bounded backoff from 30 seconds to 10 minutes; retry history survives restart.
Free online lanes are selected immediately from a fresh fleet observation and
submitted through the normal Controller review while clearing keeps retrying.
Unknown cleanup bookkeeping cannot reserve every lane forever. Idle alerts
include the clearing blocker and offline Spark names.

A load applies the whole sweep profile, so before any load (placement, clearing,
final stop) the sweep removes every assignment of its own that no live lane needs,
judged by the exact profile export rather than its bookkeeping: a lost cleanup
record can never make a clearing keep, or a stop start, a workload. Each new
clearing decision (a review blocked by a different set of workloads, or a capacity
refusal) gets fresh retry history. A clearing that is not submitted
records why (`clearing refused: …` in the status and once in the events, with the
review's blocker codes); a review still blocked by the same workloads is a wait on
their owner only after a clearing load was actually submitted for them, otherwise
the clearing is decided again and the waiting result names the refusal.

The current CLI applies profiles to the whole fleet and offers no node-scoped
load. Placement loads can therefore still carry stops for offline workloads;
excluding those effects requires a Controller/CLI scope contract. The sweep
neither waits for offline-only cleanup adoption nor counts it as lane occupancy.

Cleanup observes the original accepted application's canonical per-effect graph.
An observation timeout never cancels it or invents a new clearing request. Each
online stop scope remains occupied until its exact successful typed stop receipt
while its projection is known; absence from `fleet.loaded` never rewrites a
receipt. When clearing cannot proceed, observed fleet occupancy and a fresh
Controller review govern placement immediately. A healthy independent lane can
receive a fresh whole-fleet admission that explicitly adopts every remaining
pending cleanup on online lanes. Borrowed cleanup retains its original root, plan, ordinal,
request and child operation, and keeps the aggregate observer alive. A dual stop
scope remains atomic even when one member is healthy.

A genuinely terminal failed or cancelled cleanup retains its physical-claim
history, but cannot own a future stop. A fresh review either observes its exact
targets absent on all their online nodes and retires the local record, or admits
the sweep's own stop for the remaining run. The recovery clears the review
incident once; it never rewrites the old Controller receipt. Active cleanup may
also retire when a fresh fleet observation proves its exact targets gone.
Unavailable fleet facts cannot certify absence. Stop and restore
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
settle. An owner load (or ownership the sweep cannot read) is waited for while it
lasts; it never ends the sweep. Unreadable state is preserved beside
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

### Bounded wait inventory

All `waiting_on` producers in `spark_sweep/` use `SweepWaitReason` and
`bounded_wait` (there is no `tools/spark_sweep` directory):

| Producer | Reasons | Bound |
|---|---|---|
| `application_signal` | application queued, application backoff, operator, admission, controller blocker, controller retry | Lane budget; preparation blockers at most five minutes |
| `download_signal` | operation queued, operation backoff | One-hour observation bound; no Spark lane |
| `progress_verdict(None)` | observation unavailable | Lane budget, or one hour for downloads |
| `_observe_readiness` | route publication | Lane budget |
| `_controller_stalled` | controller stalled, with application and alarm detail | Lane budget |
| `_submit_smoke` | endpoint publication, including unreadable endpoint observations | Smoke phase budget |
| `_smoke_wait` | client key renewal, smoke busy | Smoke phase budget |
| `_advance_smoking` | smoke response (including SSE heartbeats) | Smoke phase budget; job-specific budget when declared |

The phase budget defaults to the learned lane window; job smoke uses its declared
job budget including grace. Expired streams also receive a cooperative stop signal
so a continuing SSE heartbeat closes its response on the next line.
`tests/test_sweep_wait_budget.py` guards every `waiting_on` write and contract reason,
and drives every reason beyond its budget to require lane release and a requeue.

Gateway authorization refusals (including a route denied to the client key)
end the smoke case immediately, retaining its case, status and redacted body.
Only typed LiteLLM key expiry/revocation waits for renewal. That wait uses the
smoke phase budget, persists its deadline across restart and shows
its elapsed time and budget in status. At expiry the lane is released without
a recipe attempt; the recipe is requeued at the configured defer cadence.
A successful renewed key is read from `--client-key-file` on the next request.
