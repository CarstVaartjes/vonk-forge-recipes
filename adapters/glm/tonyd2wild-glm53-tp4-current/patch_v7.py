from pathlib import Path

p = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
)
s = p.read_text()

prefill_old = (
    "                pool_topk = torch.empty(\n"
    "                    (num_rows, select_k), dtype=torch.int32, device=logits.device\n"
    "                )\n"
)
prefill_new = (
    "                pool_topk = torch.full(\n"
    "                    (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n"
    "                )\n"
)
if s.count(prefill_old) != 1:
    raise SystemExit(f"prefill alloc match count: {s.count(prefill_old)}")
s = s.replace(prefill_old, prefill_new)

decode_old = (
    "            pool_topk = torch.empty(\n"
    "                (num_rows, select_k), dtype=torch.int32, device=logits.device\n"
    "            )\n"
)
decode_new = (
    "            pool_topk = torch.full(\n"
    "                (num_rows, select_k), -1, dtype=torch.int32, device=logits.device\n"
    "            )\n"
)
if s.count(decode_old) != 1:
    raise SystemExit(f"decode alloc match count: {s.count(decode_old)}")
p.write_text(s.replace(decode_old, decode_new))

p = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/glm5next/nvidia/ops/kpool_compress.py"
)
s = p.read_text()
guard_old = "    hist_out = tl.where(pid >= 0, hist_val, -1)\n"
guard_new = "    hist_out = tl.where((pid >= 0) & (pid < pool_len), hist_val, -1)\n"
if s.count(guard_old) != 1:
    raise SystemExit(f"expansion guard match count: {s.count(guard_old)}")
p.write_text(s.replace(guard_old, guard_new))
print("indexer topk init + pool clamp applied")
