# Reproducing the overlay-v4 production campaign

## Identity and boundaries

This package serves the official `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint on four GB10/SM121 nodes with SGLang TP4/EP4, vision enabled, static DSpark K5, and native file-backed Engram lookup. It does not redistribute or quantize the checkpoint. The final runtime is the existing immutable image recorded in `evidence/final/RUNTIME-PROVENANCE.json`; use its published registry manifest digest, not a Docker config ID as a registry digest.

The embedded source revision is a local build-provenance commit, not a promise that that revision is checkoutable from this public repository. The manifest proves byte equality between the public adapter/sandbox files, the build input, and files extracted from the actual immutable image. Publication-only commits parameterize host inventory and sanitize evidence without rebuilding or changing those image bytes. Do not transfer results to a newly rebuilt image merely because it uses this Dockerfile.

## Four-node inventory

Install Docker with NVIDIA runtime and working RDMA on each node. The measured fabric used two logical NIC halves per node and native NCCL IB transport. Supply your own correctly configured fabric; do not copy unrelated ring-network settings. The checkpoint must already be present at the same absolute path on all four nodes. Verify its pinned revision and shard integrity before launch; no automatic download or weight replacement is performed.

Copy `cluster.example.json` to ignored `cluster.local.json`, then replace every null with real local values:

- `ssh_hosts`: four distinct `user@host` SSH destinations in rank order, including a working self-SSH destination for rank0.
- `hostnames`: the corresponding exact `hostname` results.
- `fabric_ips`: four rank-ordered IPv4 addresses, one on each rank's intended fabric interface.
- `dist_init_addr`: rank0 fabric IPv4 followed by a port, without a `tcp://` prefix.
- `model_path`: absolute checkpoint directory, identical across ranks.
- `ssh_identity_file`, `ssh_known_hosts_file`: absolute paths to pre-authorized SSH material. Strict host verification is mandatory; keys are never copied by the launcher.
- `nccl_env`: absolute path to your application-scoped NCCL environment file. It must select `NCCL_NET=IB` and `NCCL_IB_DISABLE=0` and your actual HCAs; each rank's socket/Gloo interface is resolved from `fabric_ips`.

Alternatively set `DSV41_CLUSTER_CONFIG` to another private inventory path. The same runtime and support-directory paths must be writable on every rank. The configured user needs non-interactive Docker access and narrowly authorized cache-reclaim/kernel-log access. The guard's peer SSH key and known-hosts paths must exist on every rank. Missing inventory fails before Docker or model load; copying the null-valued example cannot target a real or invented host accidentally.

## Long-context qualification warning

The configured 512k production window is retrieval-qualified on overlay-v4 (exclusive C1, chunk 256, K-slice cap 2048) for exactly one ordered two-key 33/66 case. Overlay-v2 512k on the historical C8 profile remains INFRA_FAILURE. 1M is deferred. Q200/vision/throughput stay bound to overlay-v2 and are not transferred to overlay-v4. Launching `prod` reproduces the qualified 512k envelope, not the historical C8 quality profile (`prod-c8`).

## Launch and stop

Pull the recorded registry digest on all ranks, and verify that `docker image inspect` returns the recorded config ID everywhere. Launch under tmux from a clean checkout:

    tmux new-session -s dsv41 'bash scripts/serve.sh prod --image-id sha256:<verified-config-id> --run-dir .hermes/runs/prod-01'

A new run directory is required for every epoch. The launcher admits at least 24 GiB available host memory per rank, refuses any running container, verifies checkpoint-index equality and host identity, reclaims page cache only before launch, starts rank-local guards, and records exact container IDs and ownership labels. READY is API/model/context admission, not semantic qualification. Run the narrow semantic checks before measurements.

For an orderly stop, require idle running/queued counters and use the launch record:

    python3 scripts/final_runtime.py stop --run-dir .hermes/runs/prod-01

Do not kill a long request client and leave its server work orphaned. Never stop or remove unowned containers. A stopped hard-abort epoch must be preserved and diagnosed rather than resumed against stale IDs.

## Evaluation

Run one heavy lane at a time. Safety guards pause new admission on pressure and hard-stop on low memory, fresh NVRM allocation failures, or cgroup OOM. Long retrieval additionally requires at least10GiB physically free (`MemFree`) on every rank at lease acknowledgement; run it in a dedicated cold epoch rather than following a long workload mix. Kernel error monitoring requires a privileged, nonempty journal: unprivileged `journalctl` can return success with no visible system records. A large `MemAvailable` value alone does not prove that the GPU allocator can obtain memory; the final campaign includes an explicitly retained BFCL infrastructure-abort attempt. Do not weaken guards to turn such an attempt into a score.

Host tooling: Python 3.12, NumPy, requests, Pillow and pytest, plus `bfcl-eval==2025.12.17` for the frozen BFCL subset. The vision collector additionally uses the separately installed r0b0bench-vision source and dataset. See its own licensing and data setup instructions; data is not bundled here.

The Q200v2 kit is public at https://github.com/r0b0tlab/r0b0bench/tree/main/subsets/q200v2 . Clone that repository and set `Q200_KIT` to its `subsets/q200v2` directory. Verify `MANIFEST.sha256` from inside that directory. The frozen text180 dataset SHA256 is `74623ab9b075120cd6f7a93059cc16d8817a6039dd20118b8f0350279f8b1ed6`.

    python3 scripts/final_eval.py primary --launch-dir .hermes/runs/prod-01
    python3 scripts/final_eval.py quality --launch-dir .hermes/runs/prod-01
    python3 scripts/final_systems.py --launch-dir .hermes/runs/prod-01

These are separate explicit phases, not permission to repeat previously completed lanes. Primary measures SHORT/MEDIUM/PROSE. Quality uses thinking enabled, effort low, one worker and an 8192-token cap. HumanEval code executes only in the image's bounded, network-free sandbox. The twenty hard-reasoning answers require independent response-hash-bound review; generation completion alone is not a Q200 score. Re-run the identical text run ID with `--manual-evidence` to recompute grades without regenerating saved responses, then use the kit's `close_q200_v2.py` to verify and merge text180 with BFCL structural-hard20. Model errors are scores; infrastructure failures remain separate attempts. The subset is not an official full BFCL leaderboard category.

The operator's final retrieval protocol is exactly one ordered two-key case at 33%/66% per window, after pre-NIAH publication. It is not the former 25/50/90 ladder:

    python3 scripts/niah.py --depths '' --twokey --window 522174 --out <new-512k-result.json> --identity <launch.json> --admission-config <admission-config.json>

For 1M, first stop the idle prod epoch and launch a fresh `1m` profile on the same image. This separate C1 profile keeps native graphs/precision and uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,garbage_collection_threshold:0.6` to proactively reclaim unused native allocator blocks; it does not change the production profile or inherit a full-model pass from the tiny CUDA oracle. Recheck actual effectiveness; recheck physical capacity and semantic readiness. Use target 1046462 with the same command form. The client checks actual native chat-token counts, ordered exact answers and completion/speculation reserve. Budget a multi-hour request and keep its owner durable. 1M is deferred on overlay-v4 production; a configured 1M window is not a result.

## Public evidence

`evidence/final/RESULTS.json` is the source for generated human-facing results. Sanitized per-row score/timing ledgers retain usage, scores and content hashes without publishing questions, answers, images, tool arguments, host identities or raw operational logs. Raw evidence is preserved privately. Historical `evidence/phase*` summaries describe older or diagnostic profiles and never substitute for final-image qualification. `ARCHIVE-POLICY.json` records the raw files withdrawn from the current public tree without rewriting Git history.
