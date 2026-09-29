#!/bin/bash
set -e
# DeepGEMM SM_121 Blackwell detection fix
# SM_121 (GB10/DGX Spark) has compute capability (12,1) = family 120
# vLLM only checks family 100 (10.x datacenter Blackwell) — add family 120
#
# Two patches needed:
#   1. support_deep_gemm() in vllm/platforms/cuda.py
#   2. DeepGemmFP4Experts._supports_current_device() in deep_gemm_moe.py

echo "=== Applying DeepGEMM SM_121 detection fix ==="

# Patch 1: support_deep_gemm() in cuda.py
TARGET=$(python3 -c "import vllm.platforms.cuda; print(vllm.platforms.cuda.__file__)" 2>/dev/null || true)
echo "Patch 1: $TARGET"

python3 -c "
path = '$TARGET'
with open(path) as f:
    content = f.read()

old = 'return cls.is_device_capability(90) or cls.is_device_capability_family(100)'
new = 'return (cls.is_device_capability(90) or cls.is_device_capability_family(100) or cls.is_device_capability_family(120))'

if new not in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('  Patched support_deep_gemm() — added family 120')
else:
    print('  Already patched')
"

# Patch 2: DeepGemmFP4Experts._supports_current_device()
EXPERTS_FILE=$(python3 -c "import vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe as m; print(m.__file__)" 2>/dev/null || true)
echo "Patch 2 (FP4): $EXPERTS_FILE"

python3 -c "
path = '$EXPERTS_FILE'
with open(path) as f:
    content = f.read()

old = \"current_platform.is_device_capability_family(100)\"
new = \"(current_platform.is_device_capability_family(100) or current_platform.is_device_capability_family(120))\"

if new not in content:
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('  Patched FP4 experts — added family 120')
else:
    print('  Already patched')
"

# Patch 3: Replace fp8_einsum.py with SM_121-compatible version
# SM_121 DeepGEMM JIT has ue8m0 scale type assertion — unpack scales to float32
EINSUM_FILE=$(python3 -c "import vllm.models.deepseek_v4.nvidia.ops.fp8_einsum as m; print(m.__file__)" 2>/dev/null || true)
echo "Patch 3 (fp8_einsum): $EINSUM_FILE"

# Copy the patched version from the mod directory
MOD_DIR=$(dirname "$0")
PATCHED_EINSUM="$MOD_DIR/fp8_einsum_patched.py"

if [ -f "$PATCHED_EINSUM" ] && [ -n "$EINSUM_FILE" ]; then
    cp "$PATCHED_EINSUM" "$EINSUM_FILE"
    echo "  Replaced fp8_einsum.py with SM_121-compatible version"
    grep -c "_unpack_ue8m0_scales" "$EINSUM_FILE" && echo "  Verified: _unpack_ue8m0_scales present"
else
    echo "  WARNING: fp8_einsum patch file not found, skipping"
fi

# Patch 4: Replace sparse_attn_compress_cutedsl.py with SM_121-compatible version
# Docker image has a corrupted file (starts with "==== CUDA ====") and fmin bug
SPARSE_FILE=$(python3 -c "import vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl as m; print(m.__file__)" 2>/dev/null || true)
echo "Patch 4 (fmin fix): $SPARSE_FILE"

MOD_DIR=$(dirname "$0")
PATCHED_SPARSE="$MOD_DIR/sparse_attn_compress_cutedsl_patched.py"

if [ -f "$PATCHED_SPARSE" ] && [ -n "$SPARSE_FILE" ]; then
    cp "$PATCHED_SPARSE" "$SPARSE_FILE"
    fmin_count=$(grep -c "cute.arch.fmin" "$SPARSE_FILE" 2>/dev/null || echo 0)
    echo "  Replaced sparse_attn_compress_cutedsl.py (fmin count: $fmin_count)"
else
    echo "  WARNING: sparse_attn patch file not found, skipping"
fi

# Patch 5: Preserve standard FP32 BMM weight scales for SM_121 fp8_einsum
# Upstream BMM post-process TMA-packs scales to INT32 for SM100.  The SM120
# scalar fp8_einsum path on GB10 expects [groups, out_rank/128, hidden/128]
# FP32 scales, so leave BMM scales untransformed on capability family 120.
FP8_UTILS_FILE=$(python3 -c "import vllm.model_executor.layers.quantization.utils.fp8_utils as m; print(m.__file__)" 2>/dev/null || true)
echo "Patch 5 (BMM scale layout): $FP8_UTILS_FILE"

python3 -c "
path = '$FP8_UTILS_FILE'
with open(path) as f:
    content = f.read()
old = '''        dg_ws = transform_sf_into_required_layout(
            sf=ws,
            mn=r,
            k=d,
            recipe=(1, quant_block_shape[0], quant_block_shape[1]),
            num_groups=g,
            is_sfa=False,
        )
        return wq, dg_ws
'''
new = '''        from vllm.platforms import current_platform
        if current_platform.is_device_capability_family(120):
            # SM120/SM121 DeepGEMM fp8_einsum consumes standard FP32
            # block scales [groups, ceil(out_rank/128), ceil(hidden/128)].
            # Do not TMA-pack BMM scales into INT32 UE8M0 layout here.
            return wq, ws.contiguous()

        dg_ws = transform_sf_into_required_layout(
            sf=ws,
            mn=r,
            k=d,
            recipe=(1, quant_block_shape[0], quant_block_shape[1]),
            num_groups=g,
            is_sfa=False,
        )
        return wq, dg_ws
'''
if new not in content:
    if old not in content:
        raise SystemExit('  ERROR: fp8_utils BMM scale-layout anchor not found')
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('  Patched BMM FP8 weight scales — SM_121 keeps standard FP32 layout')
else:
    print('  Already patched')
"

# Verify all patches
python3 -c "
import functools, importlib, sys
# Clear any cached deep_gemm checks  
from vllm.utils import deep_gemm as dg_utils
dg_utils.is_deep_gemm_supported.cache_clear()
from vllm.model_executor.layers.fused_moe.experts import deep_gemm_moe
importlib.reload(deep_gemm_moe)
from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import DeepGemmFP4Experts
dg = current_platform.support_deep_gemm()
fp4 = DeepGemmFP4Experts._supports_current_device()
print(f'  support_deep_gemm: {dg}')
print(f'  FP4 experts support: {fp4}')
if not dg:
    print('  WARNING: support_deep_gemm still False')
if not fp4:
    print('  NOTE: FP4 check may resolve at vLLM startup (fresh import). Patches applied.')
print('  DEEPGEMM SM_121 MOD COMPLETE')
"
