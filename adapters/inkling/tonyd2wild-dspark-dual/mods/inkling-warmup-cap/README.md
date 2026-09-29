# Inkling SM121 warmup cap

With `max_num_seqs: 6`, vLLM's synthetic startup warmup sends six uniform
requests through Inkling's vendored SM120/SM121 relative-attention FA4 kernel.
The tested build fails deterministically with `cudaErrorIllegalAddress` even
though KV allocation succeeds and more than 11 GiB of system memory remains.

This opt-in patch reads `INKLING_WARMUP_MAX_NUM_SEQS` and caps only that
synthetic warmup batch. It does not alter scheduler admission or the server's
runtime `max_num_seqs`; real C1-C6 requests must still pass the included
concurrency benchmark before a build is called healthy.
