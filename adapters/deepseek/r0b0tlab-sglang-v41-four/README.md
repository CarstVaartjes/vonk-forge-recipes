# DeepSeek-V4.1-Flash on 4×GB10 / SGLang TP4

r0b0tlab · @mr_r0b0t — official checkpoint, vision retained, local file-backed Engram lookup, static DSpark K5. Original adapter and reproducible runtime package; no model weights distributed.

Campaign state: 512K_RETRIEVAL_QUALIFIED. This is a measured profile, not a fastest-hardware or long-term-stability claim.

## Final-image results

| Lane | Result | Scope |
|---|---|---|
| Q200v2 | 189/200 (94.50%) | Text180 + official BFCL structural-hard20; thinking on, effort low |
| Text180 | 175/180 | All rows normally terminated; independent manual review included |
| cvbench | 39/60 (65.00%) | Thinking off; one worker; bounded CV-Bench first60 / full MMVP300 |
| mmvp | 205/300 (68.33%) | Thinking off; one worker; bounded CV-Bench first60 / full MMVP300 |
| MMVP paired | 74/150 | Both answers correct in each pair |
| NIAH 512k | PASS | Exactly one ordered two-key 33%/66% case; not a 25/50/90 ladder |
| NIAH 1m | DEFERRED | Deferred; not run on overlay-v4 production |

| Q200 family | Correct | Total |
|---|---|---|
| bfcl_hard20 | 14 | 20 |
| gsm8k | 77 | 80 |
| hard_reasoning | 19 | 20 |
| humaneval | 40 | 40 |
| ifeval | 39 | 40 |

Manual review corrects erroneous frozen references for three cases. One answer is rejected for an incorrect additional continuous-time claim despite a correct discrete recurrence. See MANUAL-REVIEW-POLICY.json; scores are bound to unchanged response hashes. Historical overlay-v1 192/200 is not the score of this image.

## Throughput: read each workload separately

Custom primary/counting rates use server usage tokens and real complete-client elapsed time. Thinking is off. Primary/counting values are medians of three measured repetitions after one excluded warmup. They include request prefill, not pure decode. Fixed-token synthetic rows are intentionally capped and are not semantic-answer quality scores.

| Workload | K5 aggregate output tok/s | K3 comparison tok/s |
|---|---|---|
| short_c1 | 30.75 | 25.87 |
| medium_c1 | 11.22 | 13.41 |
| prose_c1 | 6.77 | 9.69 |

K5 retained by the predeclared no-regression rule; K3 has higher medium/prose and geometric-mean throughput but regresses short. Not a universal speed winner.

| Counting workload | Aggregate output tok/s |
|---|---|
| counting_c1 | 58.33 |
| counting_c4 | 112.75 |

| Requested concurrency | Sustained observed | Completed | Errors | Aggregate output tok/s |
|---|---|---|---|---|
| 1 | 1 | 17 | 0 | 55.55 |
| 2 | 2 | 28 | 0 | 85.94 |
| 4 | 4 | 36 | 0 | 110.48 |
| 8 | 8 | 48 | 0 | 136.67 |

Load-counter caveat: Pinned load_inquirer.py:100 counts len(get_running_batch().reqs), without filtering finished requests. Raw c4 count transiently exceeded client concurrency; raw maxima retained and only sustained exact levels qualified. Original rc1 receipt retained.

## Native SGLang bench_serving

| Cell | Completed | Sampled input/output maxima | Retokenized E2E tok/s | Harness nominal tok/s | Mean TTFT ms |
|---|---|---|---|---|---|
| bench-short-c1 | 20 | 128/128 | 8.36 | 8.37 | 1869.7 |
| bench-short-c8 | 20 | 128/128 | 11.40 | 11.43 | 10687.1 |
| bench-medium-c1 | 20 | 2048/512 | 10.16 | 10.19 | 6798.5 |
| bench-medium-c8 | 20 | 2048/512 | 7.07 | 7.08 | 31598.6 |

Native cells use seed42 and random_range_ratio0: uniformly sampled lengths from1 to the displayed maxima, with ShareGPT-derived text, not fixed-length shapes. Harness nominal output counts can fall back to requested lengths if stream usage is absent; the decoded-response retokenized E2E rate is shown separately. Nominal input totals exclude any extra chat-template tokens. These rows are not interchangeable with the custom primary or quality workload rates.

## Telemetry and reliability

| Lane | Rank | Coverage | GPU °C mean/max | NVML mean W | Min available GiB |
|---|---|---|---|---|---|
| excluded-text-epoch-bfcl | 0 | FULL_WINDOW_SAMPLED | 50.0/56.0 | 11.4 | 22.00 |
| excluded-text-epoch-bfcl | 1 | FULL_WINDOW_SAMPLED | 55.6/63.0 | 16.6 | 24.04 |
| excluded-text-epoch-bfcl | 2 | FULL_WINDOW_SAMPLED | 56.1/62.0 | 18.5 | 23.98 |
| excluded-text-epoch-bfcl | 3 | FULL_WINDOW_SAMPLED | 50.4/54.0 | 10.5 | 26.99 |
| text180 | 0 | FULL_WINDOW_SAMPLED | 60.2/70.0 | 26.7 | 22.95 |
| text180 | 1 | FULL_WINDOW_SAMPLED | 64.1/73.0 | 31.3 | 24.87 |
| text180 | 2 | FULL_WINDOW_SAMPLED | 63.5/72.0 | 29.6 | 24.83 |
| text180 | 3 | FULL_WINDOW_SAMPLED | 61.8/71.0 | 28.5 | 27.64 |
| bench-medium-c1 | 0 | FULL_WINDOW_SAMPLED | 57.6/64.0 | 27.6 | 19.53 |
| bench-medium-c1 | 1 | FULL_WINDOW_SAMPLED | 63.6/71.0 | 32.6 | 23.41 |
| bench-medium-c1 | 2 | FULL_WINDOW_SAMPLED | 60.8/67.0 | 30.2 | 23.18 |
| bench-medium-c1 | 3 | FULL_WINDOW_SAMPLED | 59.1/65.0 | 29.1 | 26.15 |
| excluded-bfcl-systems-epoch-bench-medium-c8 | 0 | FULL_WINDOW_SAMPLED | 50.6/55.0 | 12.1 | 19.21 |
| excluded-bfcl-systems-epoch-bench-medium-c8 | 1 | FULL_WINDOW_SAMPLED | 59.6/63.0 | 21.3 | 23.44 |
| excluded-bfcl-systems-epoch-bench-medium-c8 | 2 | FULL_WINDOW_SAMPLED | 58.1/61.0 | 20.0 | 23.22 |
| excluded-bfcl-systems-epoch-bench-medium-c8 | 3 | FULL_WINDOW_SAMPLED | 56.1/59.0 | 19.1 | 26.17 |
| bench-short-c1 | 0 | FULL_WINDOW_SAMPLED | 54.1/60.0 | 21.7 | 19.43 |
| bench-short-c1 | 1 | FULL_WINDOW_SAMPLED | 60.1/66.0 | 25.8 | 23.35 |
| bench-short-c1 | 2 | FULL_WINDOW_SAMPLED | 58.9/63.0 | 24.4 | 23.11 |
| bench-short-c1 | 3 | FULL_WINDOW_SAMPLED | 56.5/61.0 | 23.3 | 25.97 |
| bench-short-c8 | 0 | FULL_WINDOW_SAMPLED | 53.1/64.0 | 21.7 | 19.61 |
| bench-short-c8 | 1 | FULL_WINDOW_SAMPLED | 60.9/67.0 | 26.8 | 23.41 |
| bench-short-c8 | 2 | FULL_WINDOW_SAMPLED | 59.3/65.0 | 25.4 | 23.21 |
| bench-short-c8 | 3 | FULL_WINDOW_SAMPLED | 57.0/66.0 | 24.1 | 26.15 |
| excluded-bfcl-systems-epoch-bfcl | 0 | NOT_CAPTURED | unavailable | unavailable | unavailable |
| excluded-bfcl-systems-epoch-bfcl | 1 | NOT_CAPTURED | unavailable | unavailable | unavailable |
| excluded-bfcl-systems-epoch-bfcl | 2 | NOT_CAPTURED | unavailable | unavailable | unavailable |
| excluded-bfcl-systems-epoch-bfcl | 3 | NOT_CAPTURED | unavailable | unavailable | unavailable |
| bfcl-hard20 | 0 | FULL_WINDOW_SAMPLED | 53.8/62.0 | 23.2 | 20.42 |
| bfcl-hard20 | 1 | FULL_WINDOW_SAMPLED | 58.8/68.0 | 27.3 | 22.79 |
| bfcl-hard20 | 2 | FULL_WINDOW_SAMPLED | 57.2/67.0 | 25.7 | 22.56 |
| bfcl-hard20 | 3 | FULL_WINDOW_SAMPLED | 55.6/64.0 | 24.7 | 26.01 |
| bench-medium-c8 | 0 | FULL_WINDOW_SAMPLED | 54.6/61.0 | 23.4 | 20.54 |
| bench-medium-c8 | 1 | FULL_WINDOW_SAMPLED | 60.3/69.0 | 27.7 | 24.15 |
| bench-medium-c8 | 2 | FULL_WINDOW_SAMPLED | 57.4/64.0 | 26.0 | 24.05 |
| bench-medium-c8 | 3 | FULL_WINDOW_SAMPLED | 56.4/64.0 | 24.8 | 27.14 |
| excluded-first-long-epoch-niah-512k | 0 | FULL_WINDOW_SAMPLED | 67.0/72.0 | 44.3 | 16.76 |
| excluded-first-long-epoch-niah-512k | 1 | FULL_WINDOW_SAMPLED | 74.7/80.0 | 51.4 | 18.32 |
| excluded-first-long-epoch-niah-512k | 2 | FULL_WINDOW_SAMPLED | 68.1/74.0 | 47.1 | 18.42 |
| excluded-first-long-epoch-niah-512k | 3 | FULL_WINDOW_SAMPLED | 70.2/75.0 | 46.9 | 21.30 |
| excluded-cold-retry-epoch-niah-512k | 0 | FULL_WINDOW_SAMPLED | 66.0/74.0 | 40.7 | 12.00 |
| excluded-cold-retry-epoch-niah-512k | 1 | FULL_WINDOW_SAMPLED | 68.2/80.0 | 45.9 | 13.99 |
| excluded-cold-retry-epoch-niah-512k | 2 | FULL_WINDOW_SAMPLED | 66.4/74.0 | 43.7 | 13.65 |
| excluded-cold-retry-epoch-niah-512k | 3 | FULL_WINDOW_SAMPLED | 68.2/75.0 | 42.8 | 16.74 |
| excluded-1m-epoch-niah-1m | 0 | FULL_WINDOW_SAMPLED | 67.5/74.0 | 42.4 | 11.75 |
| excluded-1m-epoch-niah-1m | 1 | FULL_WINDOW_SAMPLED | 74.7/82.0 | 49.5 | 14.25 |
| excluded-1m-epoch-niah-1m | 2 | FULL_WINDOW_SAMPLED | 68.2/74.0 | 45.8 | 14.24 |
| excluded-1m-epoch-niah-1m | 3 | FULL_WINDOW_SAMPLED | 70.9/77.0 | 45.1 | 17.11 |

| Evaluation | Request-E2E output tok/s | Total lane wall seconds |
|---|---|---|
| text180 | 17.43 | 6696.2 |
| bench-medium-c1 | 10.19 | 645.3 |
| bench-short-c1 | 8.37 | 215.4 |
| bench-short-c8 | 11.43 | 165.7 |
| bfcl-hard20 | 16.82 | 4879.6 |
| bench-medium-c8 | 7.08 | 898.0 |

NVML samples are not wall-outlet power. HTTP output throughput includes prefill and reasoning; not pure decode. No prefill-only rate inferred. Stopped-epoch samples excluded by exact lane bounds.

A real NVRM allocation failure occurred during the first BFCL attempt after text180. All four guards stopped the runtime; those BFCL outputs are infrastructure-invalid, not scored model failures. Text180 was preserved unchanged and unfinished lanes were retried in a fresh identical-image/profile epoch. This recovery does not claim to repair the underlying allocation failure or establish indefinite service stability.

The first medium-c8 warmup later failed with an asynchronous CUDA illegal-memory-access error reported by rank0 NCCL. The three preceding native cells and completed Q200 were preserved. The missing cell is tested in another identical-image/profile epoch; the failure is retained and no originating kernel or stability repair is claimed.

Overlay-v2 512k (same logical case, two execution attempts) returned no answer after GPU allocation errors and remains INFRA_FAILURE on that image and C8 profile. Overlay-v4 production (exclusive C1, chunk 256, independent K-slice cap 2048) completed the same logical case: usage.prompt_tokens=522174, finish_reason=stop, exact two-key 33/66 match, NVRM 0 on all ranks. Q200/vision/throughput stay bound to overlay-v2 and are not transferred. 1M is deferred.

Optional dedicated 2h mixed-workload soak: NOT_RUN. Completed serial evaluations are not relabeled as that soak.

Production profile is the exclusive C1 512k retrieval envelope on overlay-v4. Overlay-v2 remains the measured quality/throughput/vision image. Historical C8 profile is profiles/dsv41-prod-c8.env. 1M is deferred. Optional dedicated 2h mixed-workload soak: NOT_RUN.

## Runtime and reproduction

| Identity | Value |
|---|---|
| Production image config ID | sha256:c010623e97f75bd2be82fda556c667dad64839242f46d58e55be576a16c05d32 |
| Production embedded source | cc38a672aa78ced3702addc10b321b2e87a9773f |
| Upstream SGLang | da64c5cbb8cf6bfd39be19da43573fdfd484c43a |
| Production profile SHA256 | b424efadb052a7477a8db43f6c138aff01000dd33a2127e1cf0252a9ecc7f3fe |
| Advertised prod window | 524288 |
| Historical overlay-v2 image | sha256:5b246919f183289ab2a147f0ba7f22c54a72c52748082ce4cd2ec8303d38fa1f |
| Historical C8 profile SHA256 | 24aaa458977e94a7ce6b21548cbf55fba4c10baa99a2709fd41686ab7c71967a |

Verified registry reference: ghcr.io/r0b0tlab/dsv41-flash-tp4-sglang-sm121@sha256:537a7bc04a1de6cdaf2086556984a9ee26d15f479dbb2e7799f88bd52a321740

```bash
docker pull ghcr.io/r0b0tlab/dsv41-flash-tp4-sglang-sm121@sha256:537a7bc04a1de6cdaf2086556984a9ee26d15f479dbb2e7799f88bd52a321740
```

Follow docs/REPRODUCIBILITY.md for private inventory, image verification, guarded launch/stop and serial evaluation. Public inventory is intentionally empty; no private host topology is embedded. The Dockerfile and adapter source match the immutable runtime as recorded in RUNTIME-PROVENANCE.json. Host-only publication changes are not a runtime rebuild.

Full machine-readable scores, timing/usage and source hashes: evidence/final/RESULTS.json, TEXT180-SCORES.json, VISION-SCORES.json, PERFORMANCE-ROWS.json, TELEMETRY.json and MANIFEST.sha256. Raw prompts/responses, host logs and credentials remain private. Historical phase directories are not current qualification.

Credit: DeepSeek, SGLang, FlashInfer, PyTorch/Triton, NVIDIA CUDA/CUTLASS/NCCL and the upstream benchmark authors. See THIRD_PARTY_NOTICES.md. Package code is MIT; model/base-image/data retain their own terms.

Results JSON SHA256: 663c2b6566f5ea84e5b8ee7a84c0047f0bb13da7b44ae9b0aec09bbcd926a368
