"""An adapter that pins torch over an NGC PyTorch base also pins torchvision.

The NGC ``pytorch`` base ships a torchvision built for its own torch. A hash
lock that installs PyPI torch with ``--ignore-installed`` replaces torch but
leaves that torchvision, whose C++ ops then do not register
(``operator torchvision::nms does not exist``). This failed the MiniMax dual
build (2026-10-27) and the tonyd2wild GLM vision smoke (2026-11-03). Pin the
matching torchvision in the same lock, or exempt the adapter with evidence.
Bases that are pure CUDA images ship no torchvision and are out of scope.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"
NGC_PYTORCH_BASE = re.compile(r"^FROM\s+nvcr\.io/nvidia/pytorch\b", re.MULTILINE)
TORCH_PIN = re.compile(r"^torch==", re.MULTILINE)
TORCHVISION_PIN = re.compile(r"^torchvision==", re.MULTILINE)

EXEMPT = {
    "qwen/sfxnz-flash-next-tensorfold-dual": (
        "Text-only model (display/recipes/qwen3-8-flash-next-mlx-sfxnz-"
        "tensorfold-dual.json: text-only); proven on hardware; never imports "
        "torchvision."
    ),
}


def _ngc_torch_locks() -> dict[str, bool]:
    """Adapters on an NGC PyTorch base whose lock pins torch, by torchvision pin."""
    found: dict[str, bool] = {}
    for lock in sorted(ADAPTERS.glob("*/*/requirements.lock")):
        adapter = lock.parent
        dockerfiles = [p for p in adapter.glob("Dockerfile*") if p.is_file()]
        if not any(NGC_PYTORCH_BASE.search(p.read_text()) for p in dockerfiles):
            continue
        text = lock.read_text()
        if TORCH_PIN.search(text):
            found[adapter.relative_to(ADAPTERS).as_posix()] = bool(
                TORCHVISION_PIN.search(text)
            )
    return found


def test_torch_locks_over_ngc_pytorch_pin_torchvision_or_record_why_not() -> None:
    adapters = _ngc_torch_locks()
    # The scan sees real torch pins, not an empty tree.
    assert adapters
    missing = sorted(
        name for name, pinned in adapters.items() if not pinned and name not in EXEMPT
    )
    assert not missing, (
        "a torch pin over an NGC PyTorch base leaves the base's mismatched "
        f"torchvision; pin torchvision in these locks or exempt them: {missing}"
    )


def test_every_exemption_is_still_needed() -> None:
    adapters = _ngc_torch_locks()
    stale = sorted(name for name in EXEMPT if adapters.get(name) is not False)
    assert not stale, f"these exemptions no longer apply; remove them: {stale}"
