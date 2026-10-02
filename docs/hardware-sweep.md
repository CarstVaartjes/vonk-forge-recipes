# Hardware sweep

`tools/sweep-recipes` tests as many recipes as possible on the real Sparks,
fast and unattended, and records what it finds. It drives the fleet only
through `vonkctl` (JSON mode), needs no Controller credentials of its own, and
runs from the operator's machine. The code is `spark_sweep/` (stdlib only);
`tests/test_sweep_*.py` run it against a fake `vonkctl` and never touch a fleet.

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
--detach`. Applications are serialised (one load in flight), while the other
lane's smoke test runs beside it. The planner keeps unchanged assignments, so
a swap never restarts the other lane. Duals use both Sparks and run as a
window (several cached duals back to back) rather than interleaved with
singles. Two singles share a Spark when their declared `memory_bytes` plus the
reserve fit, ports and aliases differ, and the review agrees.

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
  models are never requested again. Request keys are deterministic, so a restart
  reconnects to its own operations.
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
  never-loaded **pin profile** (default `--pin-profile 13`) naming every recipe
  that is downloaded or downloading and not yet tested (assignments with
  `--state installed`), within the byte budget, and removes each recipe once
  tested. If the Controller refuses the pin edits the sweep carries on and says
  so in the status page.
* **Measured throughput.** The summed `smoothed_bytes_per_second` of the
  active operations feeds the ETA on the status page.

## What is tested

* chat, code, reasoning and vision recipes: a streamed completion measures time
  to first token and tokens per second, then the recipe's reviewed qualification
  cases run with their assertions (the image cases carry their small PNG). A
  recipe without reviewed cases runs its own declared `validation.serving` checks.
* generation and other non-OpenAI recipes (image, video, audio, 3D jobs):
  readiness only (the run is healthy and its route published), recorded as
  `readiness-only`. Their reviewed job fixtures are not submitted by the sweep yet.

Requests go to the endpoint `vonkctl profile endpoint` reports, with the key in
`--client-key-file` (`--ca-file` or `--insecure-tls` for the gateway's
certificate).

## Timeouts, failures, fixes

* Load timeouts are learned per engine from completed loads (seconds per GiB of
  model, times 3, between 20 minutes and 2 hours); before any sample, 60 minutes
  allow a first-start compile.
* Failures carry a phase (download, review, build, install, start, readiness,
  smoke, timeout) and a class (network, oom, capacity, build-policy,
  model-integrity, …). Only network-like failures are retried, once. A
  model-integrity download failure fails the lead; its siblings are recorded as
  inheriting it, not attempted. Failures with the same normalised signature form
  one cluster in the report, so a fix lands for all of them.
* Each result is bound to the recipe document digest (`content_sha256`). When a
  recipe's digest changes (the hourly refresh, a fix), a failed recipe is
  requeued and a passed one is retested last (`--no-revalidate` keeps old passes).
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
`run` refuses to start without `--yes`, because the first load replaces whatever
the Sparks run.

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

The tests use a fake `vonkctl` modelled on the Controller's OpenAPI schemas and
on real library and fleet output. Not yet seen on a real fleet: that a load
submitted while another lane's workload is up reports `keep` for it; that the
Controller accepts many `--state installed` assignments in the pin profile; and
the exact gateway `api_base` and key handling. Not built: Spark-side staging of
the next recipe ahead of its load (it would rely on the same unverified
concurrent-load behaviour), submitting the reviewed job fixtures for generation
recipes, and reading the NAS's free space (not exposed).
