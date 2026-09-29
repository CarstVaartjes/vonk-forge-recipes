#!/bin/bash
set -e
# MoE Padding Elimination for DSV4 Flash on SM121
#
# ROOT CAUSE: compute_aligned_M() uses worst-case formula when
# expert_tokens_meta is None (which NoDPEP path always is):
#   M_sum = (M * topk) + local_experts * (alignment-1)
# For DSV4 decode (M=1, topk=6, local=128, align=128):
#   = 6 + 128*127 = 16,262 → rounds to 16,384
# This is 21x more rows than actually needed.
#
# FIX: Tighter upper bound. Only min(M*topk, local_experts) experts
# can be active, not all local_experts:
#   num_active = min(M * topk, local_num_experts)
#   M_sum = (M * topk) + num_active * (alignment - 1)
#
# For M=1: 6 + 6*127 = 768 (was 16,384) — 21x reduction
# For M=8: 48 + 48*127 = 6,144 (was 16,384) — 2.7x reduction
# For M=4096 prefill: identical (all experts active anyway)
#
# This is a PURE FORMULA change — no CPU sync, no metadata, no Triton
# kernel. Works with CUDA graphs. Does not change routing math.

echo "=== Applying MoE Padding Elimination (tighter M bound) ==="

TARGET=$(python3 -c "
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import compute_aligned_M
import inspect
print(inspect.getfile(compute_aligned_M))
" 2>/dev/null || true)

if [ -z "$TARGET" ]; then
    echo "ERROR: Could not find deep_gemm_utils.py"
    exit 1
fi

echo "Target: $TARGET"

python3 -c "
path = '$TARGET'
with open(path) as f:
    content = f.read()

MARKER = 'moe_padding_fix_tight_bound'
if MARKER in content:
    print('  Already patched — skipping')
    exit(0)

old_code = '''    # expert_num_tokens information is not available on the cpu.
    # compute the max required size.
    M_sum = (M * num_topk) + local_num_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum'''

new_code = '''    # moe_padding_fix_tight_bound: Only min(M*topk, local_num_experts)
    # experts can be active, not all local_num_experts. This is a tighter
    # upper bound that reduces DeepGEMM workspace from 16384 to 768 at c=1
    # decode (21x less MoE GEMM work) without CPU sync or metadata.
    num_active_experts = min(M * num_topk, local_num_experts)
    M_sum = (M * num_topk) + num_active_experts * (alignment - 1)
    M_sum = round_up(M_sum, alignment)
    return M_sum'''

if old_code not in content:
    print('ERROR: Could not find target code in deep_gemm_utils.py')
    exit(1)

content = content.replace(old_code, new_code)
with open(path, 'w') as f:
    f.write(content)
print('  Patched compute_aligned_M() — tighter bound active')
print('  Decode M_sum: 16384 -> 768 (c=1), 6144 (c=8)')
"

PYCACHE_DIR=$(dirname "$TARGET")/__pycache__
rm -f "$PYCACHE_DIR"/deep_gemm_utils.*.pyc 2>/dev/null || true
echo "  Cleared pycache for deep_gemm_utils"
echo "=== MoE Padding Elimination complete ==="
