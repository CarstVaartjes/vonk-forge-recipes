#!/usr/bin/env bash
set -euo pipefail
NAME=${NAME:-dgemma-vllm}
OUT=${OUT:-evidence/kernels/native-kernel-check.txt}
mkdir -p "$(dirname "$OUT")"
{
  echo "# Native kernel verification $(date -Is)"
  docker exec -i "$NAME" python3 - <<'PY'
import importlib, torch
print('capability', torch.cuda.get_device_capability())
try:
    from vllm.platforms import current_platform
    print('sm120_family', current_platform.is_device_capability_family(120))
except Exception as e:
    print('platform_check_error', repr(e))
for mod in ['vllm._C','vllm._C_stable_libtorch','vllm._moe_C']:
    try:
        importlib.import_module(mod); print(mod, 'OK')
    except Exception as e:
        print(mod, 'FAIL', repr(e))
PY
  echo '## backend log lines'
  docker logs "$NAME" 2>&1 | grep -Ei 'nvfp4|modelopt|cutlass|flashinfer|marlin|emulation|triton' || true
  echo '## forbidden fallback count (excluding potential)'
  (docker logs "$NAME" 2>&1 | grep -Ei 'marlin|emulation' | grep -vi potential || true) | wc -l
} | tee "$OUT"
