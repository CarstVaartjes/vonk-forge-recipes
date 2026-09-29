#!/bin/bash
set -euo pipefail

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"

echo "[mimo-v2-tp3-virtual-heads] Installing MiMo TP=3 virtual-head padding"

cat > "$SITE_PACKAGES/vllm/model_executor/virtual_tp.py" <<'PY'
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import torch


def is_virtual_tp_padded_enabled() -> bool:
    return os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1"


def pad_or_narrow_weight(
    loaded_weight: torch.Tensor,
    dim: int,
    start_idx: int,
    shard_size: int,
) -> torch.Tensor:
    if not is_virtual_tp_padded_enabled():
        return loaded_weight.narrow(dim, start_idx, shard_size)

    if loaded_weight.ndim == 0:
        return loaded_weight.narrow(dim, start_idx, shard_size)

    dim = dim if dim >= 0 else loaded_weight.ndim + dim
    if dim < 0 or dim >= loaded_weight.ndim:
        return loaded_weight.narrow(dim, start_idx, shard_size)

    available = loaded_weight.shape[dim] - start_idx
    if available >= shard_size:
        return loaded_weight.narrow(dim, start_idx, shard_size)

    shape = list(loaded_weight.shape)
    shape[dim] = shard_size
    padded_weight = torch.zeros(
        shape,
        dtype=loaded_weight.dtype,
        device=loaded_weight.device,
    )

    if available > 0:
        valid_weight = loaded_weight.narrow(dim, start_idx, available)
        padded_weight.narrow(dim, 0, available).copy_(valid_weight)

    return padded_weight
PY

python3 - <<'PY'
from pathlib import Path

site = Path("/usr/local/lib/python3.12/dist-packages")

parameter = site / "vllm/model_executor/parameter.py"
text = parameter.read_text()
if "from vllm.model_executor.virtual_tp import pad_or_narrow_weight" not in text:
    text = text.replace(
        "from vllm.logger import init_logger\n",
        "from vllm.logger import init_logger\n"
        "from vllm.model_executor.virtual_tp import pad_or_narrow_weight\n",
        1,
    )

replacements = {
    """        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
""": """        loaded_weight = pad_or_narrow_weight(
            loaded_weight, self.output_dim, self.tp_rank * shard_size, shard_size
        )
""",
    """        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
""": """        loaded_weight = pad_or_narrow_weight(
            loaded_weight, self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
""",
    """        loaded_weight = loaded_weight.narrow(
            self.output_dim, shard_id_int * shard_size, shard_size
        )

        assert param_data.shape == loaded_weight.shape
""": """        loaded_weight = pad_or_narrow_weight(
            loaded_weight, self.output_dim, shard_id_int * shard_size, shard_size
        )

        assert param_data.shape == loaded_weight.shape
""",
    """        loaded_weight = loaded_weight.narrow(
            self.input_dim, self.tp_rank * shard_size, shard_size
        )

        if len(loaded_weight.shape) == 0:
""": """        loaded_weight = pad_or_narrow_weight(
            loaded_weight, self.input_dim, self.tp_rank * shard_size, shard_size
        )

        if len(loaded_weight.shape) == 0:
""",
}
for old, new in replacements.items():
    if old in text:
        text = text.replace(old, new, 1)
parameter.write_text(text)

linear = site / "vllm/model_executor/layers/linear.py"
text = linear.read_text()
if "from vllm.model_executor.virtual_tp import pad_or_narrow_weight" not in text:
    text = text.replace(
        "from vllm.model_executor.utils import set_weight_attrs\n",
        "from vllm.model_executor.utils import set_weight_attrs\n"
        "from vllm.model_executor.virtual_tp import pad_or_narrow_weight\n",
        1,
    )

if "class MiMoV2VirtualQKVParallelLinear" not in text:
    text += r'''


def _mimo_v2_virtual_tp3_enabled(
    tp_size: int,
    total_num_heads: int,
    total_num_kv_heads: int,
) -> bool:
    import os

    return (
        os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1"
        and tp_size == 3
        and total_num_heads == 64
        and total_num_kv_heads in (4, 8)
    )


class MiMoV2VirtualQKVParallelLinear(QKVParallelLinear):
    """MiMo-V2.5 QKV projection with opt-in TP=3 virtual head padding.

    The real checkpoint has 64 Q heads.  TP=3 cannot shard that directly, so
    this keeps the TP=2 local shape per rank: 32 Q heads.  Regular layers use
    2 KV heads per rank (4 -> virtual 6); compressed/SWA layers use 4 KV heads
    per rank (8 -> virtual 12).  Out-of-range checkpoint tails are zero-filled
    by the virtual_tp pad_or_narrow loader.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ):
        tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self._mimo_v2_virtual_tp3 = _mimo_v2_virtual_tp3_enabled(
            tp_size, total_num_heads, total_num_kv_heads
        )
        if not self._mimo_v2_virtual_tp3:
            super().__init__(
                hidden_size=hidden_size,
                head_size=head_size,
                total_num_heads=total_num_heads,
                total_num_kv_heads=total_num_kv_heads,
                bias=bias,
                skip_bias_add=skip_bias_add,
                params_dtype=params_dtype,
                quant_config=quant_config,
                prefix=prefix,
                return_bias=return_bias,
                disable_tp=disable_tp,
                v_head_size=v_head_size,
            )
            return

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.v_head_size = v_head_size if v_head_size is not None else head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = 32
        self.num_kv_heads = total_num_kv_heads // 2
        self.num_kv_head_replicas = 1

        input_size = self.hidden_size
        output_size = (
            self.num_heads * self.head_size
            + self.num_kv_heads * self.head_size
            + self.num_kv_heads * self.v_head_size
        ) * tp_size
        self.output_sizes = [
            self.num_heads * self.head_size * tp_size,
            self.num_kv_heads * self.head_size * tp_size,
            self.num_kv_heads * self.v_head_size * tp_size,
        ]

        ColumnParallelLinear.__init__(
            self,
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
            disable_tp=disable_tp,
        )

    def _uses_mimo_v2_virtual_tp3(self) -> bool:
        return getattr(self, "_mimo_v2_virtual_tp3", False)

    def _mimo_v2_original_qkv_shards(self):
        return [
            ("q", 0, self.total_num_heads * self.head_size),
            (
                "k",
                self.total_num_heads * self.head_size,
                self.total_num_kv_heads * self.head_size,
            ),
            (
                "v",
                (self.total_num_heads + self.total_num_kv_heads) * self.head_size,
                self.total_num_kv_heads * self.v_head_size,
            ),
        ]

    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        if not self._uses_mimo_v2_virtual_tp3():
            super().weight_loader_v2(param, loaded_weight, loaded_shard_id)
            return

        self.validate_shard_id(loaded_shard_id)
        if loaded_shard_id is None:
            output_dim = getattr(param, "output_dim", None)
            assert output_dim is not None
            for shard_id, shard_offset, shard_size in self._mimo_v2_original_qkv_shards():
                if isinstance(param, BlockQuantScaleParameter):
                    weight_block_size = getattr(self, "weight_block_size", None)
                    shard_size, shard_offset = adjust_block_scale_shard(
                        weight_block_size, shard_size, shard_offset
                    )
                elif (
                    isinstance(param, (PackedColumnParameter, PackedvLLMParameter))
                    and getattr(param, "packed_dim", None) == output_dim
                ):
                    shard_size, shard_offset = param.adjust_shard_indexes_for_packing(
                        shard_size=shard_size, shard_offset=shard_offset
                    )
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader_v2(param, loaded_weight_shard, shard_id)
            return

        assert loaded_shard_id in ("q", "k", "v")
        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        param.load_qkv_weight(
            loaded_weight=loaded_weight,
            num_heads=1,
            shard_id=loaded_shard_id,
            shard_offset=shard_offset,
            shard_size=shard_size,
            tp_rank=self.tp_rank,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ):
        if not self._uses_mimo_v2_virtual_tp3():
            super().weight_loader(param, loaded_weight, loaded_shard_id)
            return

        self.validate_shard_id(loaded_shard_id)
        if loaded_shard_id is None:
            output_dim = getattr(param, "output_dim", None)
            assert output_dim is not None
            for shard_id, shard_offset, shard_size in self._mimo_v2_original_qkv_shards():
                if isinstance(param, BlockQuantScaleParameter):
                    weight_block_size = getattr(self, "weight_block_size", None)
                    shard_size, shard_offset = adjust_block_scale_shard(
                        weight_block_size, shard_size, shard_offset
                    )
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return
        assert loaded_shard_id in ("q", "k", "v")
        output_dim = getattr(param, "output_dim", None)
        assert output_dim is not None

        shard_offset = self._get_shard_offset_mapping(loaded_shard_id)
        shard_size = self._get_shard_size_mapping(loaded_shard_id)
        assert shard_offset is not None and shard_size is not None
        if isinstance(param, BlockQuantScaleParameter):
            weight_block_size = getattr(self, "weight_block_size", None)
            shard_size, shard_offset = adjust_block_scale_shard(
                weight_block_size, shard_size, shard_offset
            )

        param_data = param.data.narrow(output_dim, shard_offset, shard_size)
        loaded_weight = pad_or_narrow_weight(
            loaded_weight, output_dim, self.tp_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)
'''

merged_replacements = {
    """                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
""": """                loaded_weight_shard = pad_or_narrow_weight(
                    loaded_weight, output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
""",
    """            if not is_sharded_weight:
                loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
""": """            if not is_sharded_weight:
                loaded_weight = pad_or_narrow_weight(
                    loaded_weight, output_dim, start_idx, shard_size
                )
""",
    """            loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
""": """            loaded_weight = pad_or_narrow_weight(
                loaded_weight, input_dim, start_idx, shard_size
            )
""",
}
for old, new in merged_replacements.items():
    if old in text:
        text = text.replace(old, new, 1)
linear.write_text(text)

mimo = site / "vllm/model_executor/models/mimo_v2.py"
text = mimo.read_text()
if "import os\n" not in text:
    text = text.replace("from itertools import islice\n\n", "from itertools import islice\nimport os\n\n", 1)
if "MiMoV2VirtualQKVParallelLinear" not in text:
    text = text.replace(
        """from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
""",
        """from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    MiMoV2VirtualQKVParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
""",
        1,
    )

old = """        self.total_num_heads = num_heads
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
"""
new = """        self._mimo_v2_virtual_tp3 = (
            os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1"
            and tp_size == 3
            and num_heads == 64
            and num_kv_heads in (4, 8)
        )

        self.total_num_heads = num_heads
        self.num_heads = 32 if self._mimo_v2_virtual_tp3 else self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        self.num_kv_heads = (
            num_kv_heads // 2
            if self._mimo_v2_virtual_tp3
            else max(1, self.total_num_kv_heads // tp_size)
        )
"""
if old in text:
    text = text.replace(old, new, 1)
elif "self._mimo_v2_virtual_tp3 = (" not in text:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] MiMoV2Attention head-count anchor not found")

text = text.replace(
    "        self.qkv_proj = QKVParallelLinear(\n",
    "        qkv_cls = MiMoV2VirtualQKVParallelLinear if self._mimo_v2_virtual_tp3 else QKVParallelLinear\n        self.qkv_proj = qkv_cls(\n",
    1,
)
old = """        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.v_head_dim,
"""
new = """        o_proj_input_heads = (
            self.num_heads * tp_size
            if self._mimo_v2_virtual_tp3
            else self.total_num_heads
        )
        self.o_proj = RowParallelLinear(
            o_proj_input_heads * self.v_head_dim,
"""
if old in text:
    text = text.replace(old, new, 1)
elif "o_proj_input_heads =" not in text:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] MiMoV2Attention o_proj anchor not found")

old = """            if "attention_sink_bias" in name:
                total_heads = loaded_weight.shape[0]
                heads_per_rank = total_heads // tp_size
                head_start = tp_rank * heads_per_rank
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)

                param.data.copy_(narrow_weight)
                loaded_params.add(name)
            else:
"""
new = """            if "attention_sink_bias" in name:
                import os

                if os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1" and tp_size == 3:
                    shard_size = param.data.shape[0]
                    head_start = tp_rank * shard_size
                    available = loaded_weight.shape[0] - head_start
                    narrow_weight = loaded_weight.new_zeros(param.data.shape)
                    if available > 0:
                        valid = min(shard_size, available)
                        narrow_weight[:valid].copy_(
                            loaded_weight.narrow(0, head_start, valid)
                        )
                else:
                    total_heads = loaded_weight.shape[0]
                    heads_per_rank = total_heads // tp_size
                    head_start = tp_rank * heads_per_rank
                    narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)

                param.data.copy_(narrow_weight)
                loaded_params.add(name)
            else:
"""
if old in text:
    text = text.replace(old, new, 1)
elif "attention_sink_bias" in text and "new_zeros(param.data.shape)" not in text:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] attention_sink_bias loader anchor not found")

mimo.write_text(text)

mimo_mtp = site / "vllm/model_executor/models/mimo_v2_mtp.py"
text = mimo_mtp.read_text()
old = """            # attention_sink_bias is head-parallel; slice by tp
            if "attention_sink_bias" in name:
                total_heads = loaded_weight.shape[0]
                heads_per_rank = total_heads // tp_size
                loaded_weight = loaded_weight.narrow(
                    0, tp_rank * heads_per_rank, heads_per_rank
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
"""
new = """            # attention_sink_bias is head-parallel; slice by tp.
            # MiMo TP=3 virtual heads pad each local sink vector to 32.
            if "attention_sink_bias" in name:
                import os

                if os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1" and tp_size == 3:
                    shard_size = param.data.shape[0]
                    head_start = tp_rank * shard_size
                    available = loaded_weight.shape[0] - head_start
                    padded_weight = loaded_weight.new_zeros(param.data.shape)
                    if available > 0:
                        valid = min(shard_size, available)
                        padded_weight[:valid].copy_(
                            loaded_weight.narrow(0, head_start, valid)
                        )
                    loaded_weight = padded_weight
                else:
                    total_heads = loaded_weight.shape[0]
                    heads_per_rank = total_heads // tp_size
                    loaded_weight = loaded_weight.narrow(
                        0, tp_rank * heads_per_rank, heads_per_rank
                    )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
"""
if old in text:
    text = text.replace(old, new, 1)
elif "MiMo TP=3 virtual heads pad each local sink vector to 32" not in text:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] MiMoV2MTP attention_sink_bias anchor not found")
mimo_mtp.write_text(text)

mimo_omni = site / "vllm/model_executor/models/mimo_v2_omni.py"
text = mimo_omni.read_text()
if "import os\n" not in text:
    text = text.replace("import math\n", "import math\nimport os\n", 1)
helper = """
def _mimo_v2_tp3_force_vit_data_parallel() -> bool:
    return os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1"


"""
if "def _mimo_v2_tp3_force_vit_data_parallel" not in text:
    text = text.replace("\nclass MiMoVisionMLP", "\n" + helper + "class MiMoVisionMLP", 1)
text = text.replace(
    "use_data_parallel = is_vit_use_data_parallel()",
    "use_data_parallel = is_vit_use_data_parallel() or _mimo_v2_tp3_force_vit_data_parallel()",
)
mimo_omni.write_text(text)

model_config = site / "vllm/config/model.py"
text = model_config.read_text()
old = """        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )
"""
new = """        if total_num_attention_heads % tensor_parallel_size != 0:
            import os

            is_mimo_v2_tp3_virtual = (
                os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1"
                and tensor_parallel_size == 3
                and total_num_attention_heads == 64
                and (
                    getattr(self.hf_text_config, "model_type", None) == "mimo_v2"
                    or getattr(self.hf_config, "model_type", None) == "mimo_v2"
                    or any(
                        "MiMoV2" in str(arch)
                        for arch in (getattr(self.hf_config, "architectures", None) or [])
                    )
                    or any(
                        "MiMoV2" in str(arch)
                        for arch in (getattr(self.hf_text_config, "architectures", None) or [])
                    )
                )
            )
            if not is_mimo_v2_tp3_virtual:
                raise ValueError(
                    f"Total number of attention heads ({total_num_attention_heads})"
                    " must be divisible by tensor parallel size "
                    f"({tensor_parallel_size})."
                )
"""
if old in text:
    text = text.replace(old, new, 1)
elif "is_mimo_v2_tp3_virtual = (" not in text:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] ModelConfig TP divisibility anchor not found")

old = """    def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
        \"\"\"Returns the number of KV heads per GPU.\"\"\"
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1

        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the number of query
        # heads so each GPU has at least one KV head.
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
"""
if old not in text:
    old = """    def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
        \"\"\"Returns the number of KV heads per GPU.\"\"\"
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1

        total_num_kv_heads = self.get_total_num_kv_heads()
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
"""
new = """    def get_num_kv_heads(self, parallel_config: ParallelConfig) -> int:
        \"\"\"Returns the number of KV heads per GPU.\"\"\"
        if self.use_mla:
            # When using MLA during decode it becomes MQA
            return 1

        total_num_kv_heads = self.get_total_num_kv_heads()
        import os

        if (
            os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1"
            and parallel_config.tensor_parallel_size == 3
            and (
                getattr(self.hf_text_config, "model_type", None) == "mimo_v2"
                or getattr(self.hf_config, "model_type", None) == "mimo_v2"
                or any(
                    "MiMoV2" in str(arch)
                    for arch in (getattr(self.hf_config, "architectures", None) or [])
                )
                or any(
                    "MiMoV2" in str(arch)
                    for arch in (getattr(self.hf_text_config, "architectures", None) or [])
                )
            )
            and total_num_kv_heads in (4, 8)
        ):
            return total_num_kv_heads // 2

        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
"""
if old in text:
    text = text.replace(old, new, 1)
elif "VLLM_MIMO_V2_TP3_VIRTUAL_HEADS" not in text[text.find("def get_num_kv_heads"):text.find("def get_num_attention_heads")]:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] ModelConfig get_num_kv_heads anchor not found")

old = """    def get_num_attention_heads(self, parallel_config: ParallelConfig) -> int:
        num_heads = self.model_arch_config.total_num_attention_heads
        return num_heads // parallel_config.tensor_parallel_size
"""
new = """    def get_num_attention_heads(self, parallel_config: ParallelConfig) -> int:
        num_heads = self.model_arch_config.total_num_attention_heads
        import os

        if (
            os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1"
            and parallel_config.tensor_parallel_size == 3
            and (
                getattr(self.hf_text_config, "model_type", None) == "mimo_v2"
                or getattr(self.hf_config, "model_type", None) == "mimo_v2"
                or any(
                    "MiMoV2" in str(arch)
                    for arch in (getattr(self.hf_config, "architectures", None) or [])
                )
                or any(
                    "MiMoV2" in str(arch)
                    for arch in (getattr(self.hf_text_config, "architectures", None) or [])
                )
            )
            and num_heads == 64
        ):
            return 32
        return num_heads // parallel_config.tensor_parallel_size
"""
if old in text:
    text = text.replace(old, new, 1)
elif "def get_num_attention_heads" in text and "return 32" not in text[text.find("def get_num_attention_heads"):text.find("def get_num_experts")]:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] ModelConfig get_num_attention_heads anchor not found")

old = """        if total_num_attention_heads % tensor_parallel_size != 0:
            import os

            is_mimo_v2_tp3_virtual = (
                os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1"
                and tensor_parallel_size == 3
                and total_num_attention_heads == 64
                and (
                    getattr(self.hf_text_config, "model_type", None) == "mimo_v2"
                    or getattr(self.hf_config, "model_type", None) == "mimo_v2"
                    or any(
                        "MiMoV2" in str(arch)
                        for arch in (getattr(self.hf_config, "architectures", None) or [])
                    )
                    or any(
                        "MiMoV2" in str(arch)
                        for arch in (getattr(self.hf_text_config, "architectures", None) or [])
                    )
                )
            )
            if not is_mimo_v2_tp3_virtual:
                raise ValueError(
                    f"Total number of attention heads ({total_num_attention_heads})"
                    " must be divisible by tensor parallel size "
                    f"({tensor_parallel_size})."
                )
"""
new = """        if total_num_attention_heads % tensor_parallel_size != 0:
            import os

            is_mimo_v2_tp3_virtual = (
                os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1"
                and tensor_parallel_size == 3
                and total_num_attention_heads == 64
                and (
                    getattr(self.hf_text_config, "model_type", None) == "mimo_v2"
                    or getattr(self.hf_config, "model_type", None) == "mimo_v2"
                    or any(
                        "MiMoV2" in str(arch)
                        for arch in (getattr(self.hf_config, "architectures", None) or [])
                    )
                    or any(
                        "MiMoV2" in str(arch)
                        for arch in (getattr(self.hf_text_config, "architectures", None) or [])
                    )
                )
            )
            if is_mimo_v2_tp3_virtual:
                if getattr(self.hf_text_config, "intermediate_size", None) == 16384:
                    setattr(self.hf_text_config, "original_intermediate_size", 16384)
                    setattr(self.hf_text_config, "intermediate_size", 16416)
                if getattr(self.hf_text_config, "moe_intermediate_size", None) == 2048:
                    setattr(self.hf_text_config, "original_moe_intermediate_size", 2048)
                    setattr(self.hf_text_config, "moe_intermediate_size", 2112)
                if getattr(self.hf_config, "intermediate_size", None) == 16384:
                    setattr(self.hf_config, "original_intermediate_size", 16384)
                    setattr(self.hf_config, "intermediate_size", 16416)
                if getattr(self.hf_config, "moe_intermediate_size", None) == 2048:
                    setattr(self.hf_config, "original_moe_intermediate_size", 2048)
                    setattr(self.hf_config, "moe_intermediate_size", 2112)
            else:
                raise ValueError(
                    f"Total number of attention heads ({total_num_attention_heads})"
                    " must be divisible by tensor parallel size "
                    f"({tensor_parallel_size})."
                )
"""
if old in text:
    text = text.replace(old, new, 1)
elif "original_intermediate_size" not in text:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] ModelConfig intermediate padding anchor not found")
model_config.write_text(text)

vocab = site / "vllm/model_executor/layers/vocab_parallel_embedding.py"
text = vocab.read_text()
old = """        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
"""
new = """        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_embeddings = num_embeddings
        import os

        if (
            os.environ.get("VLLM_MIMO_V2_TP3_VIRTUAL_HEADS") == "1"
            and self.tp_size == 3
            and num_embeddings == 152576
        ):
            padding_size = 3
        self.padding_size = padding_size
        self.org_vocab_size = org_num_embeddings or num_embeddings
"""
if old in text:
    text = text.replace(old, new, 1)
elif "VLLM_MIMO_V2_TP3_VIRTUAL_HEADS" not in text[text.find("self.tp_size = get_tensor_model_parallel_world_size()"):text.find("num_added_embeddings =")]:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] VocabParallelEmbedding padding anchor not found")
vocab.write_text(text)

fused_moe = site / "vllm/model_executor/layers/fused_moe/layer.py"
text = fused_moe.read_text()
old = """        dims = (hidden_dim,) if shard_dim is None else (hidden_dim, shard_dim)
        if loaded_weight.ndim > 0:
            for dim in dims:
                if (
                    0 <= dim < expert_data.ndim
                    and dim < loaded_weight.ndim
                    and expert_data.shape[dim] > loaded_weight.shape[dim]
                ):
                    expert_data = expert_data.narrow(dim, 0, loaded_weight.shape[dim])
        return expert_data
"""
new = """        dims = (hidden_dim,) if shard_dim is None else (hidden_dim, shard_dim)
        if loaded_weight.ndim > 0:
            needs_padding_zero = any(
                0 <= dim < expert_data.ndim
                and dim < loaded_weight.ndim
                and expert_data.shape[dim] > loaded_weight.shape[dim]
                for dim in dims
            )
            if needs_padding_zero:
                expert_data.zero_()
            for dim in dims:
                if (
                    0 <= dim < expert_data.ndim
                    and dim < loaded_weight.ndim
                    and expert_data.shape[dim] > loaded_weight.shape[dim]
                ):
                    expert_data = expert_data.narrow(dim, 0, loaded_weight.shape[dim])
        return expert_data
"""
if old in text:
    text = text.replace(old, new, 1)
elif "needs_padding_zero" not in text:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] FusedMoE padding-zero anchor not found")

old = """        shard_size = param.shape[shard_dim]
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        param.copy_(loaded_weight)
"""
new = """        shard_size = param.shape[shard_dim]
        start_offset = shard_size * tp_rank
        available = loaded_weight.shape[shard_dim] - start_offset
        param.zero_()
        if available <= 0:
            return
        narrow_size = min(shard_size, available)
        loaded_weight = loaded_weight.narrow(
            shard_dim, start_offset, narrow_size
        )
        param.narrow(shard_dim, 0, narrow_size).copy_(loaded_weight)
"""
if old in text:
    text = text.replace(old, new, 1)
elif "start_offset = shard_size * tp_rank" not in text[text.find("def _load_combined_w13_weight_scale"):text.find("def _load_model_weight_or_group_weight_scale")]:
    raise SystemExit("[mimo-v2-tp3-virtual-heads] FusedMoE combined scale anchor not found")
fused_moe.write_text(text)
PY

python3 - <<'PY'
import py_compile
from pathlib import Path

site = Path("/usr/local/lib/python3.12/dist-packages")
for rel in [
    "vllm/model_executor/virtual_tp.py",
    "vllm/model_executor/parameter.py",
    "vllm/model_executor/layers/linear.py",
    "vllm/model_executor/models/mimo_v2.py",
    "vllm/model_executor/models/mimo_v2_mtp.py",
    "vllm/model_executor/models/mimo_v2_omni.py",
    "vllm/config/model.py",
    "vllm/model_executor/layers/vocab_parallel_embedding.py",
    "vllm/model_executor/layers/fused_moe/layer.py",
]:
    py_compile.compile(str(site / rel), doraise=True)
    print(f"[mimo-v2-tp3-virtual-heads] py_compile ok: {rel}")
PY

echo "[mimo-v2-tp3-virtual-heads] Done"
