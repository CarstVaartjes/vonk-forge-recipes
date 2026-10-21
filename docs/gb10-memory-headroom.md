# Single-Spark engine memory headroom

The 2026-10-21 Llama 3.1 8B NVFP4 failure is a recipe reservation problem
(a). It is not evidence that installed model files are concurrent workloads.
Production was inspected through Controller GETs only; the sweep was unchanged.

## Captured failure and lifecycle evidence

The supplied `ev-llama-mem.json` and `loginfo-3542-mem.txt` show:

- Recipe `llama-3-1-8b-instruct-nvfp4-nvidia-vllm-single`, spark-3542.
- Start operation `580c36e7-a2d9-48e7-979b-147bb0fc704d` began at
  00:50:47.565673Z. Its vLLM log explicitly reports utilization **0.85**,
  a **93.28 GiB** KV cache, **1,528,320** cache tokens, and 131,072-token
  context (11.66 times that context in the pool). Reducing context alone
  does not bound this automatically sized pool.
- At 00:52:21.864065Z, the host guard recorded 15,852,806,144 bytes
  available (14.76 GiB), full-memory PSI 41.25%, pressure sustained for
  3,005 ms, and an NVIDIA allocation failure.
- Controller activity records the prior stop
  `f278c2ff-a271-46cf-9da2-4e5e55f97f94` on this Spark as successful at
  00:40:27.491558Z. No successful intervening start appears in the bounded
  activity history. The failed run/switch operation
  `1c8c40a6-adc2-49c2-805a-d4985120a8c8` has prepare/transfer/verify/prepare
  completed before start, with no conflicting-run stop phase.
- Fleet detail distinguishes `installed` (disk artifacts) from `loaded`
  (running ranks). Its many installed recipes do not establish memory
  residue. This history supports (a); historical per-process memory is not
  available to prove there were zero residual bytes.

The contemporaneous successful sibling run/switch
`61f8c35a-e3d8-40a0-bc65-377b1f2e61d4` maps installation
`b65b94f0-1bde-452e-a096-8bcccfb6e3ea` to **D3 NANO vLLM** on spark-2297.
Its final receipt reports healthy running ranks and a published route;
the recipe uses utilization **0.20**, context **32,768**, concurrency **1**,
and no absolute KV override. This is runtime evidence, not a new sweep
serving-test verdict. Smaller model weights alone do not explain safe memory
use: a fraction-sized KV pool still grows to the configured budget.

**D3 LITE vLLM** subsequently started successfully on spark-3542 itself:
job `47a8244e-a041-4d19-9c32-e12b12096a2c`, child
`1a55a6ff-9615-48a1-b5d2-6948a4d1f268`, completed 00:54:51.391374Z.
Fleet detail identifies its loaded rank. It uses utilization **0.15**,
context **32,768**, concurrency **1**, and no absolute KV override.
The 00:54:52.697744Z telemetry reports 103,687,192,576 bytes available.
This provides a same-host startup comparison and argues against persistent
memory residue as the main explanation for the Llama failure.

## Bound and upstream comparison

At platform commit `040c51b8531523bebe542b5e1f7b9403e0767f72` (also this
repository's validation platform pin),
`rust/crates/vonk-agent-helper/src/host_memory_guard.rs::Decision::observe`
uses the policy re-exported from
`rust/crates/vonk-agent-protocol/src/host_memory_guard_policy.rs`:

- `STALL_AVAILABLE_BYTES = 16 * 1024**3`;
- `FULL_PSI_PERCENT = 30`, `SUSTAINED_PRESSURE = 3 seconds`;
- the separate exhaustion threshold is only 256 MiB. The 16 GiB threshold
  is conditional on sustained pressure, not an unconditional kill floor.

The old catalog test budgeted only a 6 GB OS floor. The headroom test instead
uses the lower recorded idle MemAvailable, 125,650,657,280 bytes, and the
existing runtime overhead estimate of 8 GB (another 17 GB for a separately
loaded vLLM draft model):

```text
fraction * 130,663,231,488 + runtime_overhead
    <= 125,650,657,280 - 17,179,869,184
```

This gives maximum fractions 0.768929 (ordinary) and 0.638823 (separate
vLLM drafter). The catalog uses 0.75 and 0.60 respectively, retaining lower
existing fractions. SGLang's static pool includes its model weights; its
budget also includes the 8 GB allowance outside that pool. Recheck the
platform policy when updating the validation platform pin.

The [pinned NVIDIA playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/98b3a473f30916ec05eef3e7a01b781aaf4daa8a/nvidia/vllm/README.md)
launches the ordinary model with `vllm serve ${HF_MODEL_HANDLE}`, without a
memory or context flag. Our pinned image is NVIDIA NGC `26.05.post1-py3`,
vLLM `0.21.0+2325b6f0`, ARM64 digest
`9204569b17ee4c0eff75194b8e6e458479c8aee18953b5ab9cf359fcdac659e2`.
The [upstream v0.21.0 cache configuration](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/config/cache.py)
declares default utilization 0.92 and documents that absolute KV-cache bytes
bypass utilization. This is upstream source evidence, not execution of the
NGC container default. The actual failing container's explicit 0.85 is
established by its captured log; neither an assumed 0.90 default nor an
NGC-specific default is needed to explain this failure.

## Repair and limits

Every single-Spark vLLM/SGLang recipe now has an explicit bounded fraction;
option overrides are checked through the contract's real option merger.
The redundant SGLang 0.78/0.85 fraction option becomes one 0.75 setting.
No model, image or context pin changes. Existing resource envelopes remain
conservative demand estimates; they are not physical memory limits.

Three SparkInfer wrappers also pinned a different, higher utilization in
their shell validation, rejecting even main's earlier lowered arguments.
Those three equality checks are removed; the authored memory argument is
still forwarded to the engine and exported as `GPU_MEMORY_UTILIZATION`.
The two DSpark wrappers build/select a separate draft internally, so they
use the 0.60 cap even without an authored `speculative-config` argument.
The target-only wrapper retains the ordinary 0.75 cap.
`tests/test_sparkinfer_memory_forwarding.py` executes all three wrappers
with local stand-ins for GPU preparation and the final engine, checking
the actual argv/environment handoff and draft-model budget detection.

Absolute KV overrides are additionally checked as selected weight bytes
(including drafter weights) plus KV bytes plus 8 GB overhead. DeepSeek REAP
162B drops from 14 GiB to 6 GiB; Laguna's low-memory canary drops from 3 GB
to 512 MB. REAP 180B removes its 6 GiB override because its selected weights
plus overhead already exceed this budget. Its automatic cache sizing may
reject startup for insufficient capacity; this change does not establish
that the original context can fit. No unsafe absolute override is retained
to force allocation beyond the fractional budget.

`tests/test_single_spark_memory_headroom.py` fails for 86 recipes on main.
It also injects unsafe/implicit fractions and an oversized absolute KV pool.
The two single-Spark draft recipes that now fit the existing overhead test
are removed from its stale `PENDING_CREATOR` exceptions. Hardware execution
of the repair remains pending; this is an estimated catalog budget, not a
claim that runtime overhead can never exceed the allowance.

The platform already requires inventory collected after conflicting-run
stops in `run_switch_operations/executor_preflight.py` and accounts for
memory use/residual uncertainty in `run_switch_operations/resource_fit.py`.
No platform residue fix is supported by this incident's evidence, so there
is no `fix/admission-memory-residue` branch.
