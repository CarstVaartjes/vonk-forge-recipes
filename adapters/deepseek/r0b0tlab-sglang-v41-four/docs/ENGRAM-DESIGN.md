# Engram file-store design record (implementation, 2026-09-12)

## Decision
Serve both Engram tables from a read-only file-backed mmap of checkpoint
shards 47/48 themselves — no packing pass, no dedicated pack format, no
extra disk (the 189.1 GiB "pack" IS the checkpoint), no dedicated RAM tier
(kernel page cache = demand-paged, reclaimable under pressure).

## Why (vs the alternatives, all measured against 121.7 GiB unified/node)
| Placement | Cost/node | Fits? | Collectives |
|---|---|---|---|
| Stock device-sharded (upstream default) | +47.3 GiB | no | all-reduce per lookup |
| Host private shard (env host-table) | +47.3 GiB RAM | no | all-reduce per lookup |
| Host shared (memfd, upstream) | 189 GiB RAM | no | none — but impossible |
| Rank-sliced NVMe pread pool (plan's original) | 47 GiB disk | yes | none only if slices independent |
| **File-backed mmap of shard (ours)** | 0 extra | yes | **none** |

The file-backed design subsumes the plan's packer concept: the checkpoint's
weight+scale tensors are each contiguous ranges in the shard, and upstream's
`engram_gather` takes separate raw pointers — so the shard file is directly
a valid gather source. Byte-exactness is by construction (we serve the exact
checkpoint bytes; safetensors' own mmap loader reads the same pages).

## Mechanics (adapter/sglang_patch/engram_file_store.py)
- `FileEngramStore`: PROT_READ MAP_SHARED mmap; torch uint8 views over the
  two tensor ranges reinterpreted as f8_e4m3/f8_e8m0; exposes weight_ptr /
  scale_ptr (what `engram_gather` consumes on GPU, over the coherency link —
  upstream logs this mode "unpinned (ATS)").
- `_FileHostTableShim`: duck-types `_HostTable` with `layout="shared"` so
  `EngramEmbedding.forward` takes the direct full-table gather branch
  (no `tensor_model_parallel_all_reduce`) and the model's layer-14 prefetch
  CUDA stream activates (`embed._shared` true).
- `install()`: replaces `EngramEmbedding.__init__` BEFORE construction —
  the stock ctor would allocate the sharded GPU params (~55 GiB transient);
  ours mmaps instead. `weight_loader` becomes a no-op (loader's
  `safe_open.get_tensor` returns lazy mmap views; returning without copying
  reads none of the 189 GiB). `row_start=0, rows=N, tp_size=1` = full-table
  identity on every rank.
- sitecustomize installs the patch at interpreter start when
  `DSV41_ENGRAM_FILE_STORE=1` + `DSV41_MODEL_PATH` set (fail-closed with a
  named error if the path is missing).

## Evidence so far
- tests/test_engram_store.py: 4 PASS (geometry/contiguity, bit-exact
  reference gather incl. NaN semantics and out-of-range→zero, no-collective
  static contract).
- Upstream source audit at the pinned commit da64c5cb (engram.py,
  engram_gather.py, deepseek_v4.py wiring incl. the `_shared` prefetch
  branch).
- Real-checkpoint fidelity probe: to run at Phase 6 admission against the
  downloaded shard 47 (N random rows vs safetensors' own mmap reads).
- Live no-allreduce proof: profiler trace at admission (Phase 6/7).

## Risks / follow-ups
- GPU-side gather over file-backed pages = PCIe-scale latency per row read
  on first touch; page cache warms with use. If step time shows it, options
  (in order): readahead (`posix_fadvise WILLNEED` on hot regions at load),
  a small bounded mlock'd prefix, or MADV_HUGEPAGE on the mapping.
- `EngramEmbedding.__init__` signature drift across image repins — the
  install() patch asserts the ctor shape it expects and fails closed.
- CUDA graph capture with host-pointer gathers: upstream already supports
  host tables under graphs (their day-0 stack runs graphs with the host
  layout); our A/B at admission includes a graphs-on boot specifically to
  confirm replay stability.
