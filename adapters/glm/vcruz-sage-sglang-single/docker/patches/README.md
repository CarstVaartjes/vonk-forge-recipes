# SGLang v0.5.22 patches

Applied in order by `docker/Dockerfile` (`git apply -p1` in `/sgl-workspace/sglang`). Each patch was written
against the stock `lmsysorg/sglang:v0.5.22` file (sha-checked) and served in the measured runs; comments and
debug prints were tidied for publication.

| Patch | File | What it does |
|---|---|---|
| 0001 | `srt/models/glm5_next.py` | Routes EXL3 sidecar tensors (`.trellis/.suh/.svh/.mcg/.mul1`) to the plugin's per-shard params; renames the pack's KDA `self_attn.conv1d.weight` to SGLang's `qkv_conv1d.weight` (without it the 34 KDA short-conv weights stay uninitialised and output is gibberish); fuses q_a/kv_a only for dense `.weight` tensors; asserts every `qkv_conv1d` weight loaded. |
| 0002 | `srt/models/dflash.py` | Uses `torch.topk` for the DFlash selector instead of the flashinfer `top_k` probe. |
| 0003 | `srt/speculative/dflash_worker_v2.py` | Skips the non-greedy sampling prewarm when `DF1_SKIP_DFLASH_SAMPLING_PREWARM=1` (it calls flashinfer sampling kernels). |
| 0004 | `srt/arg_groups/overrides.py` | Allows the triton DSA prefill/decode backends on CUDA when `CAMPAIGN_ALLOW_CUDA_TRITON_DSA=1`. |
| 0005 | `srt/layers/attention/dsa/dsa_backend_kpool.py` | Lets the triton DSA implementation take the full top-k (including the tail) when the k-pool is > 1. |
| 0006 | `kernels/ops/attention/dsa/triton_sparse_mla.py` | Guards the rope-tail loads/dots with `if D_TAIL > 0` (this model's DSA layers have no rope tail in that path). |
| 0007 | `kernels/ops/attention/dsa/triton_sparse_mla_decode.py` | Same guard in the decode kernels. |
| 0008 | `srt/layers/attention/trtllm_mla_backend.py` | Returns 0 for the trtllm-gen MLA counter buffer when flashinfer is unavailable (the DSA backend constructor calls it). |
| 0009 | `srt/speculative/dflash_utils.py`, `kernels/ops/speculative/dspark/dspark_accept.py` | Ours, not the creator's. When `is_flashinfer_available()` is false (`SGLANG_IS_FLASHINFER_AVAILABLE=false`) the DFlash verify path binds `top_k_renorm_prob`/`top_p_renorm_prob` to `None` (torch fallbacks) and `SoftmaxTemp` skips `flashinfer.sampling.softmax` (triton). Without it any request with temperature > 0 reaches the flashinfer stub and kills the engine (`RuntimeError: ... flashinfer.sampling.top_p_renorm_probs`); the creator only ran greedy. |
