# Sweep global gate audit

Scope: `spark_sweep/run.py`, `owner.py`, `prefetch.py`, `cli.py` and their
state/report paths. All profile loads apply the entire enrolled fleet. A lane
being locally free never authorizes bypassing a canonical review or adoption
binding. This scheduler change builds on the ended cleanup-owner reconciliation
in recipes PR #1557; it does not change cleanup receipts or declare old stops
successful.

| Gate or idle path | Scope and behaviour |
| --- | --- |
| `paused`: followed application `progress.stalled` | Item problem. Previously stopped scheduling and cleanup everywhere. Now retains the original lane/receipt, emits an observation without spending an attempt, and alerts with ids. Other lanes place through canonical adoption. |
| `paused`: platform `stalled_applications` | Application alarm, including unrelated applications. Now an alert listing ids, never a fleet scheduling veto. The actual affected lane remains accounted for by its slots, cleanup and fresh review. |
| Owner progress unreadable (profiles 1–3) | Genuine ownership safety boundary: cannot prove the latest owner request permits takeover. Re-read all profiles every tick, even without clock advancement; fresh success clears the hold immediately. No old exception is consulted. |
| Owner application active, including our own restore | Genuine whole-fleet request, since profiles cover the fleet. Hold names profile and application id. Re-evaluate each tick. |
| Owner preemption and post-load hold | Genuine owner priority. Preempt affected sweep work without blaming recipes. Hold has an explicit persisted expiry shown in status, checked every tick; new owner activity steers. |
| Startup fleet unreadable or no online Spark | Whole-fleet observation unavailable. Retry at the poll cadence, saving status with the exact startup step and error; no increasing global startup sleep. |
| Startup sweep/pin profile unreadable or foreign assignments | Ownership security boundary before editing those profiles. Unknown ownership is observed again; foreign assignments fail explicitly. Profile claiming is not inferred from other successful API calls. |
| Startup owner exports | Optional backup reads: failures do not gate startup. Owner progress still supplies the fresh ownership check. |
| Startup takeover/clearing | Item/cleanup problem. Eager clearing is attempted once without a startup wait gate; normal ticks own retries and observation while independent lanes remain runnable. |
| Infrastructure `backoff_until` | Removed the shared scheduler timer. Review, clearing, edit, submission, endpoint, smoke, credential, updater, observation, tick/loop, state-save, results-log and status failures cannot lend their retry deadline to unrelated cached work. |
| Profile edit fails in a placement batch | Item failure. Roll back, defer that recipe with capped infrastructure backoff, and re-plan the smaller batch in the same tick. Each re-plan removes a candidate. |
| Rollback and review refusal | Only rolled-back candidates receive a retry delay. Structured admission decisions already re-plan the admitted subset immediately. Deterministic recipe-data errors retain their existing evidence/verdict path. |
| Whole-profile export/prune or rollback cleanup fails | Canonical consistency boundary: an unaccounted saved assignment could otherwise start on every subsequent whole-fleet load. Retry the exact removal with its capped cadence; never submit a profile whose effects cannot be accounted for. A mere infrastructure record is no longer a veto. |
| Review unanswered, would stop a live lane, or lacks exact loading/cleanup adoption | Security/effect ownership edge for that proposed whole-fleet mutation. Roll back and re-review; never manufacture adoption. A fresh valid review is used regardless of an old infrastructure deadline. An actually invalid canonical whole-fleet review can still prevent submission: safe node-scoped loads require a Controller contract, not a local bypass. |
| Clearing refusal: unanswered/denied review, foreign offline stop, live-lane stop, unbound adoption, residual-stop mismatch | Refusal applies to the clearing proposal. Immediately use fresh fleet occupancy and ordinary canonical placement reviews for free online lanes; retry clearing with 30-second to 10-minute capped backoff. No three-attempt latch and no global wait for that retry deadline. |
| Ended cleanup owner with residual occupancy | Keep the physical scope held. Prefer a free lane's fresh reviewed placement to an empty clearing that would add another adoption dependency. When no lane is free, normal clearing recovery still runs. No receipt is rewritten. |
| Unknown cleanup projection, pending exact stop, terminal unconfirmed stop | Cleanup/item occupancy. Exact online stop scopes, including whole gangs, remain held. Clearing fallback uses fresh fleet occupancy; the final canonical review still governs every mutation. Cleanup observation retains the original request and bounded retry budget. |
| Released/orphan load | Only its actual scopes are held. Original observers needed by adopted live lanes remain. Other lanes place; absence and cancellation keep their existing ownership rules. |
| One application progress read fails | Only that request's load window freezes. Other observed loads can advance; the next fresh receipt clears its uncertainty. A failed fleet read freezes fleet-dependent load judgements and fresh takeover observation gates new submissions. |
| Dual recipe at queue head | Healthy lanes still drain for a dual. A stalled/unreadable peer, cleanup scope or released workload cannot reserve an independent free lane: ready singles may pass it. |
| No catalog/ready candidate, missing assets, recipe delay, capacity/ports, offline topology, selection/limit | Availability constraints, not global safety pauses. Catalog is paged against cached rows; per-recipe delays and offline topology do not reserve other lanes. A truly unavailable whole fleet has no placement. |
| Download/NAS/prefetch pressure, pin failure or admission backoff | Download/pin work only. Cached recipes remain schedulable. Download and pin retries keep their existing budgets and original accepted requests. |
| Client update or provenance unavailable | Informational/retry state; valid canonical work continues. Signed updater compatibility remains authoritative. No local label substitutes for contract compatibility. |
| Tick/loop exception, state or log write failure | Costs that pass/operation and remains visible; no scheduler deadline is inherited. Required durable intent writes still precede submission. |
| Idle watchdog, done/watch, finish/interrupt | Idle alerts do not gate scheduling. Done/watch waits only after the selected campaign is complete; interrupt is explicit owner intent. Finishing cleanup/restoration is not a running-lane scheduling gate. |

`tests/test_sweep_noblock.py` reloads saved state, injects receipt, edit and
cleanup faults, and requires independent placement within one tick. It also
checks that a fresh owner observation removes a global hold without waiting
for the old polling interval. Existing adoption/gang tests continue to enforce
the whole-fleet security boundaries.
