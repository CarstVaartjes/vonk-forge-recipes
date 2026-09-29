# Postmortem — N1/N3 freeze during phase-9 benchmarks (2026-09-13)

## Impact

Both N1 (node0, head) and N3 (node2) became unresponsive during
concurrent phase-9 benchmark lanes (~12:30–13:49 CDT). Both nodes were
power-cycled at ~14:29–14:30 CDT. The dsv41-rank container exited 255 on N1;
six qwen38fn containers on N1 had been SIGKILLed in the preceding 26 h.
N2 and N4 never froze.

## Root cause

**Unified-memory (GB10) DRAM exhaustion on under-prepared nodes, wedging the
NVIDIA kernel driver — triggered by three concurrent benchmark clients.**

Chain of failure, from kernel/journal evidence of boot -1 on both nodes:

1. The sglang rank reserves `mem_fraction_static=0.80` of the GB10 *unified*
   pool (weights 70.2 GB + draft 2.7 GB + KV/graph pools → ~84 GB/rank
   "GPU-side"). On GB10 that reservation competes with host DRAM for the same
   physical pages.
2. Host-side residents on the frozen nodes were not accounted for:
   - N1: ~27 GB stale page cache from previously loaded weight shards
     (`sar` 11:50: kbmemused ~93 GB, kbcached ~29.7 GB), the Hermes gateway +
     desktop stack (~10+ GB), plus three bench clients (NIAH ~510 K-token
     prompts, cvbench ×4 workers, Q200) each holding multi-GB Python prompt
     buffers.
   - N3: k3s (leftover from the July k3s campaign, ~2 GB RSS) on top of its
     rank container.
3. Physical DRAM ran out. `MemAvailable` on N1 fell 24.6 → 19.5 → 16.7 →
   14.5 GiB across 11:50–13:00 while swap-in hit ~3 000 pages/s. The kernel
   logged `Under memory pressure, flushing caches` storms on both nodes
   (N3: continuous 19:24–19:29 UTC).
4. NVRM could no longer satisfy allocations: `NVRM: ... Out of memory
   [NV_ERR_NO_MEMORY] (0x00000051) returned from _memdescAllocInternal`
   repeatedly (N1: 17:53 Sep 12 → 13:13 Sep 13; the failures began the
   previous evening as page cache accumulated).
5. The sglang scheduler thread then spun in `serverAllocResource` holding the
   driver's global API rw-semaphore while retrying the failing allocation.
   Kernel hung-task reports name the owner explicitly:
   `task nvidia-smi ... blocked on an rw-semaphore likely owned by task
   sglang::schedul`.
6. Every GPU-touching process on the node then D-blocked behind that
   semaphore: `nvidia-smi` blocked 1 105+ s, docker health checks timed out,
   runc/containerd streams died, journald was watchdog-SIGKILLed. That
   unresponsiveness is the observed "freeze". SSH console on N1 stopped
   responding to even trivial commands (`sleep 300; docker logs` timed out
   after 420 s); the box was power-cycled by the operator.

## Differential (why N2/N4 survived)

N2/N4 were serving the same rank footprint with (a) no leftover k3s, (b) no
desktop/Hermes stack (idle, ~4–7 GB used), and (c) no bench clients or stale
weight-shard cache. Kernel logs for their boot -1 show **zero**
`NV_ERR_NO_MEMORY` events in the same window.

## Contributing factor

Three benchmark clients ran concurrently against the single serve
(NIAH-512k + cvbench×4 + Q200), all on the head. The interleaving starved the
NIAH prefill (throughput fell 130 → 57 tok/s), and the combined client-side
prompt buffers on the head accelerated the DRAM exhaustion. The agent's
mid-run `kill -STOP` of the vision bench was a symptom-level mitigation, not
a fix.

## Fix (root-cause, no envelope reduction)

The serving envelope (`mem_fraction_static=0.80`, `max_total_tokens
=1 100 000`, chunked prefill 4096) is unchanged — capacity claims depend on
it. Instead the nodes are now prepared like the survivors:

1. **`scripts/node_prepare.sh`** (run on every rank before launch):
   fail-closed checkpoint presence, **MemAvailable ≥ 24 GiB admission floor**,
   page-cache drop (`sync` + `drop_caches`), and a pre-launch state record
   into `logs/node-prepare.log`.
2. **`serve.sh`**: ships + runs the preparer on every rank before container
   start, and passes `--weight-loader-drop-cache-after-load` so each
   safetensors shard calls `posix_fadvise(DONTNEED)` after loading — the
   stale 27 GB shard cache never accumulates in the first place.
3. **`scripts/bench_orchestrator.sh`**: phase-9 lanes run **serially** — one
   lane at a time, MemAvailable floor (12 GiB) on the head before every lane,
   scheduler-drain cooldown between lanes, per-lane receipts.
4. **N3 k3s disabled** (`systemctl disable --now k3s`): a July leftover with
   stock manifests and no user workloads, consuming unified memory for nothing.

## Ongoing (noted, not blocking)

- The 1 M-context multineedle case on the long profile runs after
  publication; tracked as ongoing in the package.
