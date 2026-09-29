#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
GLM-5.2 NVFP4 ds_mla KV-cache port — plumbing patcher.

Registers the `nvfp4_ds_mla` KV-cache dtype (400 B/token GLM record, see
LAYOUT-SPEC.md) in a vLLM 0.23.1 tree (pin ab666069) and wires the
FLASHMLA_SPARSE path to the NVFP4 store/gather/decode kernels.

Two modes:
  --image <site-packages/vllm dir>   patch the 5 in-image files
  --overlay <dir>                    patch a COPY of the glm-triton overlay dir
                                     (flashmla_sparse.py) — never the KNOWNGOOD one

Every edit is an exact-anchor replace: if an anchor is not found EXACTLY ONCE,
the patcher aborts loudly (no silent partial application). Idempotent: skips
edits whose replacement text is already present.

Credits: nvfp4_ds_mla concept/lineage: Keys' NVFP4 image (v0.21) and
danielwoz/vllm-dspark-nvfp4 (Apache-2.0); fp8_ds_mla structure this parallels:
jasl/vllm deepseek_v4 path via CosmicRaisins/glm-5.2-gb10; recipe target:
0xdfi/GLM-5.2-1M-4x-DGX-Spark.
"""

import argparse
import sys
from pathlib import Path

APPLIED, SKIPPED = [], []


def patch(path: Path, anchor: str, replacement: str, tag: str) -> None:
    text = path.read_text()
    if replacement in text:
        SKIPPED.append(tag)
        return
    n = text.count(anchor)
    if n != 1:
        sys.exit(f"FATAL [{tag}]: anchor found {n}x (need exactly 1) in {path}\n"
                 f"--- anchor ---\n{anchor}\n--------------")
    path.write_text(text.replace(anchor, replacement))
    APPLIED.append(tag)


# ============================================================================
# IMAGE-SIDE EDITS (run inside the build container against site-packages/vllm)
# ============================================================================

def patch_image(vllm_dir: Path) -> None:
    # -- 1. config/cache.py: register the dtype string --------------------------
    patch(
        vllm_dir / "config/cache.py",
        '    "fp8_ds_mla",\n',
        '    "fp8_ds_mla",\n    "nvfp4_ds_mla",\n',
        "cache.py:CacheDType",
    )

    # -- 2. utils/torch_utils.py: storage dtype + quantized predicate ----------
    patch(
        vllm_dir / "utils/torch_utils.py",
        '    "fp8_ds_mla": torch.uint8,\n',
        '    "fp8_ds_mla": torch.uint8,\n    "nvfp4_ds_mla": torch.uint8,\n',
        "torch_utils:str_to_dtype",
    )
    patch(
        vllm_dir / "utils/torch_utils.py",
        '        kv_cache_dtype.startswith("fp8")\n'
        '        or kv_cache_dtype.endswith("per_token_head")\n'
        '        or kv_cache_dtype == "nvfp4"\n',
        '        kv_cache_dtype.startswith("fp8")\n'
        '        or kv_cache_dtype.endswith("per_token_head")\n'
        '        or kv_cache_dtype == "nvfp4"\n'
        '        or kv_cache_dtype == "nvfp4_ds_mla"\n',
        "torch_utils:is_quantized_kv_cache",
    )

    # -- 3. v1/kv_cache_interface.py: 400 B/token page math --------------------
    patch(
        vllm_dir / "v1/kv_cache_interface.py",
        '        if self.cache_dtype_str == "fp8_ds_mla":\n'
        '            if self.model_version == "deepseek_v4":',
        '        if self.cache_dtype_str == "nvfp4_ds_mla":\n'
        '            # GLM NVFP4 ds_mla: 256B packed E2M1 NoPE + 16B UE8M0 per-32\n'
        '            # scales + 128B bf16 RoPE = 400B/token (61% of fp8_ds_mla 656).\n'
        '            # See LAYOUT-SPEC.md (nvfp4-port).\n'
        '            return self.block_size * 400\n'
        '        if self.cache_dtype_str == "fp8_ds_mla":\n'
        '            if self.model_version == "deepseek_v4":',
        "kv_cache_interface:page_size",
    )

    # -- 4. mla_attention.py: don't auto-rewrite nvfp4_ds_mla to fp8_ds_mla ----
    patch(
        vllm_dir / "model_executor/layers/attention/mla_attention.py",
        '            and kv_cache_dtype != "fp8_ds_mla"\n',
        '            and kv_cache_dtype not in ("fp8_ds_mla", "nvfp4_ds_mla")\n',
        "mla_attention:auto_convert_guard",
    )

    # -- 5. fusion pass: never fuse rope+store for nvfp4 (fused C++ op is fp8) -
    fusion = vllm_dir / "compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py"
    patch(
        fusion,
        "        ops.concat_and_cache_mla_rope_fused(\n",
        '        if kv_cache_dtype == "nvfp4_ds_mla":\n'
        "            raise RuntimeError(\n"
        '                "nvfp4_ds_mla must not reach the fused rope+store path; "\n'
        '                "the MLA rope fusion pass should skip nvfp4 layers"\n'
        "            )\n"
        "        ops.concat_and_cache_mla_rope_fused(\n",
        "fusion:runtime_guard",
    )
    patch(
        fusion,
        "        for _, layer in attn_layers.items():\n",
        "        for _, layer in attn_layers.items():\n"
        '            if getattr(layer, "kv_cache_dtype", None) == "nvfp4_ds_mla":\n'
        "                continue  # nvfp4: keep unfused path (Triton store)\n",
        "fusion:registration_skip",
    )


# ============================================================================
# OVERLAY-SIDE EDITS (run against a COPY of ~/glm-triton, e.g. ~/glm-triton-nvfp4)
# ============================================================================

def patch_overlay(overlay_dir: Path) -> None:
    f = overlay_dir / "flashmla_sparse.py"

    # -- a. backend dtype registration -----------------------------------------
    patch(
        f,
        '        "fp8_ds_mla",\n        "fp8",  # alias for fp8_ds_mla\n',
        '        "fp8_ds_mla",\n        "nvfp4_ds_mla",\n'
        '        "fp8",  # alias for fp8_ds_mla\n',
        "flashmla:supported_dtypes",
    )

    # -- b. cache shape: 400-byte records --------------------------------------
    patch(
        f,
        '        if cache_dtype_str == "fp8_ds_mla":\n'
        "            # V3.2 main MLA: 656-byte custom storage format. See module docstring.\n"
        "            return (num_blocks, block_size, 656)\n",
        '        if cache_dtype_str == "nvfp4_ds_mla":\n'
        "            # GLM NVFP4: 256B E2M1 + 16B UE8M0 scales + 128B bf16 rope = 400B.\n"
        "            return (num_blocks, block_size, 400)\n"
        '        if cache_dtype_str == "fp8_ds_mla":\n'
        "            # V3.2 main MLA: 656-byte custom storage format. See module docstring.\n"
        "            return (num_blocks, block_size, 656)\n",
        "flashmla:kv_cache_shape",
    )

    # -- c. metadata builder treats nvfp4 like fp8 (same scheduling/metadata) --
    patch(
        f,
        '        self.use_fp8_kv_cache = cache_config.cache_dtype == "fp8_ds_mla"\n',
        "        # nvfp4_ds_mla shares the fp8_ds_mla metadata/scheduling shape\n"
        '        self.use_fp8_kv_cache = cache_config.cache_dtype in (\n'
        '            "fp8_ds_mla",\n            "nvfp4_ds_mla",\n        )\n'
        '        self._is_nvfp4_kv = cache_config.cache_dtype == "nvfp4_ds_mla"\n',
        "flashmla:builder_flag",
    )

    # -- c2. rescue hatch: GLM52_NVFP4_FORCE_SEPARATE=1 avoids the b12x
    #        prefill_mg lane for nvfp4 (prefill then goes through the Triton
    #        gather-dequant + bf16 workspace kernel instead)
    patch(
        f,
        "        fp8_use_mixed_batch = self.num_heads < MIN_HEADS_FOR_BF16_PREFILL\n",
        "        fp8_use_mixed_batch = self.num_heads < MIN_HEADS_FOR_BF16_PREFILL\n"
        "        import os as _os\n"
        "        if (\n"
        '            getattr(self, "_is_nvfp4_kv", False)\n'
        '            and _os.getenv("GLM52_NVFP4_FORCE_SEPARATE", "0") == "1"\n'
        "        ):\n"
        "            fp8_use_mixed_batch = False\n",
        "flashmla:nvfp4_separate_hatch",
    )

    # -- d. impl init: accept nvfp4_ds_mla, same prefill workspace -------------
    patch(
        f,
        '        if is_quantized_kv_cache(kv_cache_dtype):\n'
        '            assert kv_cache_dtype == "fp8_ds_mla", (\n'
        '                "FlashMLA Sparse Attention backend fp8 only supports "\n'
        '                "fp8_ds_mla kv-cache dtype"\n'
        "            )\n",
        '        if is_quantized_kv_cache(kv_cache_dtype):\n'
        '            assert kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla"), (\n'
        '                "FlashMLA Sparse Attention backend only supports "\n'
        '                "fp8_ds_mla / nvfp4_ds_mla quantized kv-cache dtypes"\n'
        "            )\n",
        "flashmla:init_assert",
    )
    patch(
        f,
        '        if kv_cache_dtype == "fp8_ds_mla":\n'
        "            # Reserve workspace during initialization\n",
        '        if kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla"):\n'
        "            # Reserve workspace during initialization\n",
        "flashmla:workspace",
    )

    # -- e. forward flag: quantized-record path covers both --------------------
    patch(
        f,
        '        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"\n',
        '        use_fp8_cache = self.kv_cache_dtype in ("fp8_ds_mla", "nvfp4_ds_mla")\n',
        "flashmla:forward_flag",
    )

    # -- f. prefill chunk gather: nvfp4 -> Triton gather-dequant ---------------
    patch(
        f,
        "                ops.cp_gather_and_upconvert_fp8_kv_cache(\n"
        "                    kv_c_and_k_pe_cache,\n"
        "                    chunk_workspace,\n"
        "                    chunk.block_table,\n"
        "                    chunk.seq_lens,\n"
        "                    chunk.workspace_starts,\n"
        "                    len(chunk.block_table),\n"
        "                )\n",
        '                if self.kv_cache_dtype == "nvfp4_ds_mla":\n'
        "                    from vllm.v1.attention.backends.mla.nvfp4_glm_kernels import (\n"
        "                        gather_dequant_nvfp4_glm,\n"
        "                    )\n"
        "                    gather_dequant_nvfp4_glm(\n"
        "                        kv_c_and_k_pe_cache.view(torch.uint8),\n"
        "                        chunk_workspace,\n"
        "                        chunk.block_table,\n"
        "                        chunk.seq_lens,\n"
        "                        chunk.workspace_starts,\n"
        "                        len(chunk.block_table),\n"
        "                    )\n"
        "                else:\n"
        "                    ops.cp_gather_and_upconvert_fp8_kv_cache(\n"
        "                        kv_c_and_k_pe_cache,\n"
        "                        chunk_workspace,\n"
        "                        chunk.block_table,\n"
        "                        chunk.seq_lens,\n"
        "                        chunk.workspace_starts,\n"
        "                        len(chunk.block_table),\n"
        "                    )\n",
        "flashmla:prefill_gather",
    )

    # -- g. decode: b12x carries nvfp4; compiled-FlashMLA fallback is fp8-only -
    patch(
        f,
        "            b12x_result = b12x_glm_mla_attention(\n"
        "                q=q,\n"
        "                kv_cache=kv_cache_uint8,\n"
        "                topk_indices=topk_indices,\n"
        "                softmax_scale=self.softmax_scale,\n"
        "            )\n",
        "            b12x_result = b12x_glm_mla_attention(\n"
        "                q=q,\n"
        "                kv_cache=kv_cache_uint8,\n"
        "                topk_indices=topk_indices,\n"
        "                softmax_scale=self.softmax_scale,\n"
        '                kv_layout=(\n'
        '                    "nvfp4_ds_mla"\n'
        '                    if self.kv_cache_dtype == "nvfp4_ds_mla"\n'
        '                    else "fp8_ds_mla"\n'
        "                ),\n"
        "            )\n",
        "flashmla:b12x_layout",
    )
    patch(
        f,
        "        if b12x_result is None:\n"
        "            out, lse = flash_mla_with_kvcache(\n",
        "        if b12x_result is None:\n"
        '            if self.kv_cache_dtype == "nvfp4_ds_mla":\n'
        "                # The compiled FlashMLA kernel reads fp8_ds_mla 656B records\n"
        "                # only; there is no nvfp4 fallback. Fail loud (rollback =\n"
        "                # KNOWNGOOD fp8 launcher) rather than corrupt output.\n"
        "                raise RuntimeError(\n"
        '                    "b12x NVFP4 sparse decode failed and no fallback kernel "\n'
        '                    "reads nvfp4_ds_mla records — see b12x error above"\n'
        "                )\n"
        "            out, lse = flash_mla_with_kvcache(\n",
        "flashmla:no_fp8_fallback",
    )

    # -- h. write path: override do_kv_cache_update on the sparse impl ---------
    patch(
        f,
        "    def _forward_bf16_kv(\n",
        "    def do_kv_cache_update(\n"
        "        self,\n"
        "        kv_c_normed,\n"
        "        k_pe,\n"
        "        kv_cache,\n"
        "        slot_mapping,\n"
        "        kv_cache_dtype,\n"
        "        k_scale,\n"
        "    ) -> None:\n"
        '        """nvfp4_ds_mla: Triton store (LAYOUT-SPEC.md 400B record); else base."""\n'
        '        if kv_cache_dtype == "nvfp4_ds_mla":\n'
        "            if kv_cache.numel() == 0:\n"
        "                return\n"
        "            from vllm.v1.attention.backends.mla.nvfp4_glm_kernels import (\n"
        "                store_nvfp4_glm_kv,\n"
        "            )\n"
        "            store_nvfp4_glm_kv(\n"
        "                kv_c_normed,\n"
        "                k_pe.squeeze(1),\n"
        "                kv_cache.view(torch.uint8),\n"
        "                slot_mapping.flatten(),\n"
        "            )\n"
        "            return\n"
        "        super().do_kv_cache_update(\n"
        "            kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale\n"
        "        )\n"
        "\n"
        "    def _forward_bf16_kv(\n",
        "flashmla:store_override",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, help="site-packages/vllm dir to patch")
    ap.add_argument("--overlay", type=Path, help="overlay dir COPY to patch")
    args = ap.parse_args()
    if not args.image and not args.overlay:
        sys.exit("need --image and/or --overlay")
    if args.image:
        patch_image(args.image)
    if args.overlay:
        patch_overlay(args.overlay)
    print(f"applied ({len(APPLIED)}): " + ", ".join(APPLIED))
    print(f"skipped-already-present ({len(SKIPPED)}): " + ", ".join(SKIPPED))
    print("PLUMBING OK")


if __name__ == "__main__":
    main()
