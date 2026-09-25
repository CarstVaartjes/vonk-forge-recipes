# Per-recipe qualification evidence inventory

Report snapshot updated: **2026-09-25T04:07:27Z**. Recipe checkout `main` is
`784bda637a45c9fab6b18fc0cd2faa2669ab9800`; accepted release is `v1.0.17` at
`efbbba29bd4c706c73d295d24047787be3f36d78`. The merged platform structural run is
`4c8cf45540bca1f32c6c9b1b963c68783aa644bb` at `2026-09-25T03:29:03.058743+00:00`:
**85/85 passed**, with no physical claim.

The JSON file contains one row per recipe identity, including recipe and package
digests, pinned model sources, topology, batch/lane, structural result, cache evidence,
inference status, recovery references and blockers.

## Evidence state

- Source freshness comes from the plan’s historical upstream audit at **2026-09-23
  19:22:37 UTC** (178 watched sources; 119 matched and 59 advanced). No upstream refresh
  was run for this report, so present channel freshness is unknown.
- The identity-matched cache matrix has 85 rows: 1 ready, 38 blocked by
  `library.cache_missing`, and 46 unavailable after the shared assessment budget. Those
  counts are historical through **2026-09-24 23:07:49 UTC**; they are not a
  post-deployment full-cache refresh. The exact GLM 1.6.7 row also has a targeted ready
  record while its broad page timed out. Merged historical state: **2 ready, 38 blocked,
  45 unknown**. Unknown does not mean missing. A later exact Step1X detail at **03:48:58
  UTC** reports the cache ready and fit eligible only on Spark 2297. Its no-force
  preparation operation succeeded at **03:49:48 UTC**, yielding image digest
  `sha256:c8e6ac563cfa95c1f3a26bec630d8928aebc0b0fe994c2c7dea3a08794b1acbf` and reusing
  11 cached model artifacts totaling 7,249,194,310 bytes. This preparation is not
  inference or recovery evidence.
- **Current-identity inference: 0/85 executed.** Historical GLM 1.6.6 streaming and
  non-streaming results stay tied to recipe digest
  `4e5255a78123f3054a4cbff993996a89262082de5c518c7e02eb906eee3e8665`, image
  `sha256:fa1868a9403baee4527b113fc84a6cfb89cf227595ced71205d1d5e19f7be950`, install
  `503907a4-58e5-4fef-863f-5701f7a08675` and run `c848e6b8-fe97-4fdc-b935-3647c41d16aa`.
  They do not qualify the changed GLM 1.6.7 identity.
- No current exact-identity physical recovery receipt is recorded. Recovery
  definitions in canonical coverage are plans, not completed receipts. End-to-end
  recovery receipt consumption remains unverified while the cross-artifact digest
  comparison is reviewed. The canonical recovery contract binds the agent build digest;
  a recipe tar digest and agent package digest identify different artifacts. Both
  authenticated agents report binary and build digests matching the signed target. Their
  optional package receipt remains an evidence gap, not an established campaign blocker.
- **81** recipes are in the two-Spark scope; **4** require more than two Sparks and
  remain catalog-audit-only.
- Accepted publication **36090614107** succeeded for signed generation
  `f057a9e59058d7beb7098d721fd21c07fbddd9ca39b1327babd9557e4756095e`, source
  `4c8cf45540bca1f32c6c9b1b963c68783aa644bb`. The NAS redeployment and Controller CLI
  update completed at **03:45 UTC**. Deployment capture verified the API image digest
  `sha256:519cf084dba79fd79c35f89053bd000f10e4f05d290d769e8ea50dfff5e16344`; all 11 NAS
  services were healthy. At the 03:45:45 UTC Fleet snapshot both Sparks were online,
  ready and had no loaded workloads. These observations establish deployment and
  readiness only; there are still no new physical inference or recovery receipts.

Five installed plans on Spark 3542 fail canonical parsing because required `memory_floor_bytes` and `memory_kind` fields are absent. Their retained reservations total 1,320,491,003,815 bytes, with 0 bytes proven discountable; this blocks admission on that Spark. No supported repair was identified.

## Provider blockers recorded in the plan

| Blocker | Affected recipe rows | Recorded condition |
|---|---|---|
| `ltx25-authentic-per-file-hashes` | `ltx-2-5-22b-distilled-fp8-cast-diffusers-single`, `ltx-2-5-22b-distilled-bf16-diffusers-single` | 28 selected files require authentic per-file SHA-256 values; the pinned official audio_vae/config.json returned HTTP 401. Aggregate checksums do not close this gate. |
| `dinov3-provider-approval` | `trellis-2-4b-pytorch-single`, `pixal3d-pytorch-single` | Requires provider approval and authenticated access; a public mirror is not an approved substitute. |
| `pixal3d-naf-cache-contract` | `pixal3d-pytorch-single` | The pinned official checkpoint is a GitHub release asset; the documented model-cache provider resolves Hugging Face model files, so a supported cache contract is required. |

These conditions are carried from the checked-in plan. Provider approval and
cache-provider capability were not refreshed for this read-only inventory.

## Recipes outside two-Spark capacity

| Recipe | Nodes | Reason |
|---|---:|---|
| `glm-5-2-aqlm-vllm-triple` | 3 | requires more than two Sparks; catalog audit only |
| `glm-5-2-quanttrio-vllm-four` | 4 | requires more than two Sparks; catalog audit only |
| `glm-5-3-flash-nvfp4-vllm-four` | 4 | requires more than two Sparks; catalog audit only |
| `inkling-975b-a41b-nvfp4-sglang-eight` | 8 | requires more than two Sparks; catalog audit only |

The JSON inventory is the per-recipe record for this draft:
`recipe-evidence-inventory-2026-09-25.json`. Delivery, cache, capacity and physical
evidence are point-in-time observations; refresh them as the parent execution proceeds.
