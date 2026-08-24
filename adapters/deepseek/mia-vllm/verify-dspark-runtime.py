#!/usr/bin/env python3
"""Verify the pinned Anemll runtime contains the required DSpark loader mapping."""

from __future__ import annotations

import hashlib
from pathlib import Path


TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/dspark.py"
)
EXPECTED_SHA256 = "efe33c32d37ed7f26d869d94626f1415906d31218ec0ee44d79bb2b815b8cf39"
REQUIRED = (
    '("gate_up_proj", "w1", 0),',
    '("gate_up_proj", "w3", 1),',
    'is_layer_param = name.startswith("model.layers.")',
    "name = name.replace(weight_name, param_name)",
)


if not TARGET.is_file():
    raise SystemExit(f"DSpark loader is missing: {TARGET}")
payload = TARGET.read_bytes()
actual = hashlib.sha256(payload).hexdigest()
if actual != EXPECTED_SHA256:
    raise SystemExit(
        f"DSpark loader source drift: expected {EXPECTED_SHA256}, got {actual}"
    )
source = payload.decode("utf-8")
missing = [anchor for anchor in REQUIRED if anchor not in source]
if missing:
    raise SystemExit(f"DSpark shared-expert mapping contract is missing: {missing}")
print(actual)
