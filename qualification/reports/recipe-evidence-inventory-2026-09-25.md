# Per-recipe qualification evidence inventory

Report snapshot updated: **2026-09-25T07:29:27Z**. Recipe `main` is
`506a4af6f9c8c52acf0213aba7f2ab63bbad254f`; accepted release is `v1.0.17` at
`efbbba29bd4c706c73d295d24047787be3f36d78`. The 85/85 structural run remains
bound to recipe checkout `784bda637a45c9fab6b18fc0cd2faa2669ab9800` and platform
`4c8cf45540bca1f32c6c9b1b963c68783aa644bb`, captured at
`2026-09-25T03:29:03.058743+00:00`. Structural success is not physical qualification.

The JSON file has one row per exact recipe/package identity, with source audit,
structural result, latest and historical cache observations, fit state,
inference status, recovery references and blockers.

## Evidence state

- Source freshness remains the plan's historical upstream audit at
  **2026-09-23 19:22:37 UTC** (178 watched sources; 119 matched and 59 advanced).
  No upstream refresh was performed for this report; present source-channel
  freshness is unknown.
- The latest complete exact-identity cache assessment finished at
  **2026-09-25 07:03:08.790476 UTC**. Its 85 summary rows span
  **06:59:39.098475–07:03:06.507613 UTC**; all recipe content hashes matched
  (85/85), with zero mismatches and zero command errors. The snapshot reports
  **3 ready and 82 blocked**. Its ready recipes are Step1X geometry, Step1X label
  geometry and GLM-5-3 Flash EXL3 DFlash2. Each row retains the observation
  timestamp, content hash, revision, reason details and separately recorded fit
  state. This is the latest supplied scan, not availability after that window.
- The superseded Sep 24 cache matrix remains in `previous_cache_snapshot` and in
  each row's `previous_exact_observations`: its matrix reported 1 ready, 38
  blocked and 46 unavailable; earlier merged exact observations were 2 ready,
  38 blocked and 45 unknown. The newer full scan supersedes those counts without
  erasing their timestamps. Unknown was not treated as missing.
- **Current-identity inference: 0/85.** Historical GLM 1.6.6 streaming and
  non-streaming results remain bound only to recipe digest
  `4e5255a78123f3054a4cbff993996a89262082de5c518c7e02eb906eee3e8665`, image
  `sha256:fa1868a9403baee4527b113fc84a6cfb89cf227595ced71205d1d5e19f7be950`,
  install `503907a4-58e5-4fef-863f-5701f7a08675` and run
  `c848e6b8-fe97-4fdc-b935-3647c41d16aa`. They do not qualify the changed
  GLM 1.6.7 identity.
- Step1X label geometry now has a later exact cache-ready observation, but its
  no-force preparation request `8800c9b1-39f8-4a37-9c65-998cf780919f` retains
  failed parent operation `ed53ea64-ae62-490c-9b44-1a787d3d09ab`. The stored
  parent receipt is non-retryable with PostgreSQL `55P03` while locking
  `model_cache_operations`. At `06:45:47.300430 UTC`, later progress recorded
  both child tasks succeeded: the runtime image at 13,611,009,024 bytes and the
  model cache at 8,781,511,993 bytes across 14 items, totaling 22,392,521,017
  bytes. The subsequent cache summary at **06:59:39.098659 UTC** and detail
  assessment at **06:59:40.790032 UTC** report this same recipe identity ready.
  The parent remains failed; child completion and cache readiness are not
  inference or recovery evidence. Step1X geometry's earlier successful
  no-force image/model preparation stays attached to its separate recipe row.
- Source PR #894 merged as `1b1e10f9847ed185ab1e2f4f006e4c797ca9b5a1`, with final
  source head `e7188a946219684fa2c24b4a8f4a66dce212876d`. Required CI run
  `36106444427` passed. Six focused availability cases included one
  real-PostgreSQL contention/recovery case; the other five covered explicit retry
  and model-child behavior. There were also 26 provenance tests, one
  receipt-builder test and 106 earlier identity tests. Type checking reported one
  existing exception; no database schema changed. Development-image workflow
  `36106900488` and installer acceptance workflow `36107593799` succeeded.
- Installer workflow `36107593799` accepted signed generation
  `bad7e9e7d903eac0db3acb41ad21d444f61283a2f696c66cb4f01a4d0c6f4abf` for source
  `1b1e10f9847ed185ab1e2f4f006e4c797ca9b5a1`, release
  `0.1.1~dev.611+gf2be83c2f165`. The accepted API, Hermes, LiteLLM and worker
  digests are recorded in the JSON report. The signed agent package
  `0dd85a2fc642fb5143a8c7fc8f04430108286e27c7e579d0035b427c7a168c71` matches
  the previous package; no Spark-agent rollout is needed. Staging preserved `.env`
  and all 62 secret files with no Compose diff. Deployment was recorded at
  **07:38:55 UTC**: 11 healthy services, only API/worker replaced, and all prior
  volume mounts preserved. The CLI updated to the same source and its installed
  helper validated fresh Controller provenance for both Sparks at **07:39:16 UTC**,
  including current build/binary identity with optional package receipts absent.
- The previously accepted PR #893 publication run `36090614107` and 03:45 UTC
  NAS/CLI deployment describe the prior baseline. Its verified API digest was
  `sha256:519cf084dba79fd79c35f89053bd000f10e4f05d290d769e8ea50dfff5e16344`; all
  11 NAS services were healthy. Do not treat this baseline as deployment of PR
  #894; the follow-up deployment and corrected identity boundary are verified
  separately above. The Controller's publication projection remains unknown;
  accepted publication is independently proved by the signed release.
- A fresh no-force Step1X label request succeeded at **07:40:23 UTC**, operation
  `93795a0e-3f4e-44b8-b92a-1c152d0732f8`, reusing the same model child and runtime
  build for **22,392,521,017 bytes** of completed assets. The original failed
  receipt remains failed. The 07:41:14 UTC assessment is cache-ready and fits
  Spark 2297 only; this is not physical inference or recovery evidence.
- Following the operator's Docker update, the **07:51:08 UTC** Fleet read showed
  both Sparks online with no loaded workloads. NAS verification found 11 healthy
  services, the accepted Vonk images intact, and all prior volume mounts
  preserved; Docker reported 29.6.2. These are timestamped observations.
- No current exact-identity physical recovery receipt is recorded, and no
  physical fault/restart was injected. The identity-bound source correction and
  later CLI provenance check are separate from actual physical recovery receipts.
  Recovery definitions in canonical coverage are plans, not completed receipts.
- **81** recipes fit the two-Spark campaign scope; **4** need more than two
  Sparks and remain catalog-audit-only.

Five installed plans on Spark 3542 fail canonical parsing because required
`memory_floor_bytes` and `memory_kind` fields are absent. Their retained
reservations total **1,320,491,003,815 bytes**, with 0 bytes proven discountable;
this blocks admission on that Spark. No supported repair was identified.

## Provider blockers recorded in the plan

| Blocker | Affected recipe rows | Recorded condition |
|---|---|---|
| `ltx25-authentic-per-file-hashes` | `ltx-2-5-22b-distilled-fp8-cast-diffusers-single`, `ltx-2-5-22b-distilled-bf16-diffusers-single` | 28 selected files require authentic per-file SHA-256 values; the pinned official audio_vae/config.json returned HTTP 401. Aggregate checksums do not close this gate. |
| `dinov3-provider-approval` | `trellis-2-4b-pytorch-single`, `pixal3d-pytorch-single` | Requires provider approval and authenticated access; a public mirror is not an approved substitute. |
| `pixal3d-naf-cache-contract` | `pixal3d-pytorch-single` | The pinned official checkpoint is a GitHub release asset; the documented model-cache provider resolves Hugging Face model files, so a supported cache contract is required. |

These conditions are carried from the checked-in plan; provider approval and
cache-provider capability were not refreshed for this inventory.

## Recipes outside two-Spark capacity

| Recipe | Nodes | Reason |
|---|---:|---|
| `glm-5-2-aqlm-vllm-triple` | 3 | requires more than two Sparks; catalog audit only |
| `glm-5-2-quanttrio-vllm-four` | 4 | requires more than two Sparks; catalog audit only |
| `glm-5-3-flash-nvfp4-vllm-four` | 4 | requires more than two Sparks; catalog audit only |
| `inkling-975b-a41b-nvfp4-sglang-eight` | 8 | requires more than two Sparks; catalog audit only |

This is a timestamped progress report, not the final post-campaign report.
Refresh source, cache, publication/deployment and physical evidence when new
accepted evidence is available.
