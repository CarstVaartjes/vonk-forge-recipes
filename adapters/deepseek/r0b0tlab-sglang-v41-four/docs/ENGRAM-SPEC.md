# Engram spec — DeepSeek-V4.1-Flash (verified from checkpoint, 2026-09-12)

Model revision: dba1be0a40aa45a94ad051997016db3960a90277

## Tables (config.text_config)
- engram_layer_ids: [1, 14]
- engram_num_embeddings: [384006168, 384016682]
- head_dim: 256, n_heads (hash): 8, max_ngram_size: 4
- FP8 block size: 32 (weight_block_size [32,32] / kernel FP8_BLOCK_SIZE)

## Tensors (safetensors headers, verified)
| tensor | shard | dtype | shape | bytes |
|---|---|---|---|---|
| layers.1.engram.embed.weight | 00047 | F8_E4M3 | [384006168, 256] | 98,305,579,008 |
| layers.1.engram.embed.scale  | 00047 | F8_E8M0 | [384006168, 8]   |  3,072,049,344 |
| layers.14.engram.embed.weight| 00048 | F8_E4M3 | [384016682, 256] | 98,308,270,592 |
| layers.14.engram.embed.scale | 00048 | F8_E8M0 | [384016682, 8]   |  3,072,133,456 |
| layers.{1,14}.engram.wkv.weight | 47/48 | F8_E4M3 | [25600, 6144] | 157,286,400 each |
| layers.{1,14}.engram.wkv.scale  | 47/48 | F8_E8M0 | [800, 192]    | 153,600 each |
| layers.{1,14}.engram.q_weight/k_weight | 47/48 | BF16 | [4, 5120] | 40,960 each |

Row width: 256 B weights + 8 B scales = 264 B/row. Total embed tables
203.07 GB = 189.1 GiB. Weight and scale tensors are each contiguous ranges
inside the shard file → the shard itself is a valid gather source.

## Upstream layout semantics (sglang @ da64c5cb, layers/engram.py)
- Default: rows sharded over TP in device memory; lookup = owned-rows gather +
  tensor_model_parallel_all_reduce.
- SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 + layout "shared": ONE copy with
  every row, mapped by all ranks (memfd via /proc/<pid>/fd — same PID ns);
  forward calls engram_gather(weight_ptr, scale_ptr, ids, out, dim, blk)
  directly, NO collective. layout "private": per-rank host shard, keeps
  all-reduce.
- engram_gather (kernels/ops/embeddings/engram_gather.py): raw pointers,
  ids [N] int, out [N, dim] bf16; ids outside [row_lo, row_hi) → zero rows;
  pointers "may live in device or host memory".
- Engram.forward → embed(hash_ids) then wkv (ReplicatedLinear, MXFP8) then
  engram_gate. Layer-14 prefetch CUDA stream activates when embed._shared.
- Loading: EngramEmbedding sets weight_loader=_load_rows on both params;
  shared layout copies rank's row range into the table.

## GB10/4-node consequences (121.7 GiB unified per node)
- Device-sharded: +47.3 GiB/rank → does not fit next to ~72 GiB/rank weights.
- Host private (anonymous 47.3 GiB/rank): same unified pool → does not fit.
- Host shared (memfd 189 GiB): per-node RAM impossible AND memfd is
  node-local — cannot span 4 nodes at all.

## Our design: DSV41_ENGRAM_FILE_STORE
Read-only file-backed mmap of checkpoint shards 47/48 per rank; torch views
over the two contiguous tensor ranges per layer; host_table shim with
layout="shared" so Engram.forward takes the no-collective gather path with
row_lo=0/row_hi=all. No packing, no extra disk, no dedicated RAM (page cache
is the evictable tier), byte-exact vs pinned checkpoint, layer-14 prefetch
path active. Loader: weight_loader no-op (safe_open passes mmap views, so
skipping the copy skips the IO).
