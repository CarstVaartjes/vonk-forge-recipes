# hy4-flash-sparks patch: Hy4 flash attention on the DGX Spark's GPU (GB10, sm_121).
# Tencent's build only wires Hy4 flash to FlashMLA, which is Hopper / data-centre Blackwell only. vLLM also ships
# FlashInfer's SM120 sparse-MLA kernel (made for DeepSeek V4 on this GPU family); it takes the same packed
# fp8_ds_mla KV rows, top-k index lists and per-head sinks, so both Hy4 layer kinds can run on it:
#   sparse layers:  IndexPool's top-k rows            sliding layers: the last `window` rows, as an index list
"""Hy4 flash sparse / sliding MLA on FlashInfer's SM120 kernel."""

import torch

from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import (
    FlashInferMLASparseSM120Impl,
)


class FlashSM120Impl(FlashInferMLASparseSM120Impl):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads,
        alibi_slopes,
        sliding_window,
        kv_cache_dtype,
        logits_soft_cap,
        attn_type,
        kv_sharing_target_layer_name,
        indexer=None,
        **mla_args,
    ):
        sinks = mla_args.pop("sinks", None)
        # The window reaches the kernel as an index list (FlashWindowIndexer), not as a kernel option.
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            None,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            **mla_args,
        )
        if sinks is not None and (
            sinks.dtype != torch.float32 or tuple(sinks.shape) != (num_heads,)
        ):
            raise ValueError(f"Hy4 sinks must be float32 [{num_heads}], got {sinks.dtype} {tuple(sinks.shape)}")
        self.sinks = sinks

    def _run_mqa_kernel(self, q, kv_cache, topk_indices_physical):
        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        output = q.new_empty((q.shape[0], self.num_heads, self.kv_lora_rank), dtype=q.dtype)
        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)
        capacity = topk_indices_physical.shape[1]
        out = flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=None,
            max_seq_len=capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=capacity,
            kv_scale_format=self.kv_scale_format,
            sinks=self.sinks,
        )
        return out.squeeze(1)


class FlashSparseSM120Backend(FlashInferMLASparseSM120Backend):
    @staticmethod
    def get_impl_cls():
        return FlashSM120Impl

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_pcp(cls):
        return False

    @classmethod
    def supports_dcp(cls):
        return False


class FlashSWASM120Backend(FlashSparseSM120Backend):
    @classmethod
    def supports_sliding_window(cls):
        return True
