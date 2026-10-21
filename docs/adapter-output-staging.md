# Job output staging

`/outputs` is a writable artifact root, not an empty working directory. The
platform creates `tmp/` and mounts the installation's persistent cache at
`cache/`. Its output collector excludes these runtime-owned directories and
requires the exported entries to be plain files matching the signed output
slots. Keep incomplete files below `tmp/`, not beside final artifacts.

For atomic publication, create a unique temporary directory below
`/outputs/tmp`, generate and validate the artifact there, then `os.replace`
the artifact into its final name under `/outputs`. Both paths stay on the
output filesystem. Clean up only that temporary directory; preserve other
scratch work and caches. A failed generation, validation or rename must leave
an existing final artifact unchanged.

Retries do not promise a clean directory. The agent retains the job's exit
receipt and outputs for reobservation and delivery; a delivery-only retry
cannot launch a new process. Do not infer permission to rerun from missing
local evidence, or clear the output root to prepare an adapter.

Verified read-only against `CarstVaartjes/vonk-forge` main at
`486f7ff1e6dedb2b77d03bee36c8120e8bc5d1c6` on 2026-10-21:

- [Runtime writable-path contract](https://github.com/CarstVaartjes/vonk-forge/blob/486f7ff1e6dedb2b77d03bee36c8120e8bc5d1c6/docs/runtime-writable-path-contract.md).
- [OCI startup](https://github.com/CarstVaartjes/vonk-forge/blob/486f7ff1e6dedb2b77d03bee36c8120e8bc5d1c6/rust/crates/vonk-agent/src/oci/lifecycle.rs#L100-L143)
  and [scratch preparation](https://github.com/CarstVaartjes/vonk-forge/blob/486f7ff1e6dedb2b77d03bee36c8120e8bc5d1c6/rust/crates/vonk-agent/src/oci/storage.rs#L129-L152).
- [Output collection](https://github.com/CarstVaartjes/vonk-forge/blob/486f7ff1e6dedb2b77d03bee36c8120e8bc5d1c6/rust/crates/vonk-agent/src/executor/jobs/outputs.rs#L180-L207).
- [Durable exit receipt](https://github.com/CarstVaartjes/vonk-forge/blob/486f7ff1e6dedb2b77d03bee36c8120e8bc5d1c6/rust/crates/vonk-agent/src/oci/jobs.rs#L6-L86)
  and [delivery retry dispatch](https://github.com/CarstVaartjes/vonk-forge/blob/486f7ff1e6dedb2b77d03bee36c8120e8bc5d1c6/rust/crates/vonk-agent/src/executor/jobs.rs#L271-L332).

The adapter-wide audit found MOVA's empty-root requirement (shared by its 360p
and 720p recipes). Other job adapters already tolerate existing root entries.
The similar `glm/0xsero-exl3-k2-single/tools/prepare_mtp_draft.py` check protects
creation of an offline checkpoint view; it is not a job artifact output root.
`tests/test_adapter_output_staging.py` guards job entrypoints against output-root
emptiness checks and exercises MOVA preparation, atomic publication, failure
cleanup and retries with retained scratch and cache entries.
