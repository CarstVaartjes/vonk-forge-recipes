"""An adapter whose lock pins torch also pins torchvision, unless its base has none.

A base that already ships torchvision (such as the NGC ``pytorch`` image) has
one built for its own torch. A hash lock that installs PyPI torch with
``--ignore-installed`` replaces torch but leaves that torchvision, whose C++ ops
then do not register (``operator torchvision::nms does not exist``); this failed
the tonyd2wild GLM vision smoke (2026-11-03). Pin the matching torchvision in the
same lock, or exempt the adapter with evidence. Bases listed in
``NO_TORCHVISION_BASES`` are plain OS or CUDA images that ship no torchvision.
A ``FROM`` that cannot be resolved counts as possibly shipping torchvision.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ROOT / "adapters"
NO_TORCHVISION_BASES = (
    "nvcr.io/nvidia/cuda:",
    "docker.io/nvidia/cuda:",
    "nvidia/cuda:",
    "ubuntu:",
    "docker.io/library/ubuntu:",
    "debian:",
    "python:",
)
FROM_LINE = re.compile(
    r"^FROM\s+(?:--\S+\s+)?(\S+)(?:\s+AS\s+(\S+))?", re.MULTILINE | re.IGNORECASE
)
ARG_LINE = re.compile(r"^ARG\s+(\w+)=(\S+)", re.MULTILINE)
VARIABLE = re.compile(r"\$\{(\w+)\}|\$(\w+)")
TORCH_PIN = re.compile(r"^torch==", re.MULTILINE)
TORCHVISION_PIN = re.compile(r"^torchvision==", re.MULTILINE)

EXEMPT = {
    "qwen/sfxnz-flash-next-tensorfold-dual": (
        "Text-only model (display/recipes/qwen3-8-flash-next-mlx-sfxnz-"
        "tensorfold-dual.json: text-only); proven on hardware; never imports "
        "torchvision."
    ),
}


def _external_bases(dockerfile: str) -> list[str]:
    """Each FROM image, variables resolved from ARG defaults ('' when unresolved)."""
    args = dict(ARG_LINE.findall(dockerfile))
    stages: set[str] = set()
    bases: list[str] = []
    for image, stage in FROM_LINE.findall(dockerfile):
        if image not in stages:
            resolved = VARIABLE.sub(
                lambda m: args.get(m.group(1) or m.group(2), "\0"), image
            )
            bases.append("" if "\0" in resolved else resolved)
        if stage:
            stages.add(stage)
    return bases


def _ships_torchvision(adapter: Path) -> bool:
    for dockerfile in adapter.glob("Dockerfile*"):
        for base in _external_bases(dockerfile.read_text()):
            if not base.startswith(NO_TORCHVISION_BASES):
                return True
    return False


def _torch_locks() -> dict[str, bool]:
    """Adapters whose lock pins torch over a base that may ship torchvision."""
    found: dict[str, bool] = {}
    for lock in sorted(ADAPTERS.glob("*/*/requirements.lock")):
        text = lock.read_text()
        if TORCH_PIN.search(text) and _ships_torchvision(lock.parent):
            found[lock.parent.relative_to(ADAPTERS).as_posix()] = bool(
                TORCHVISION_PIN.search(text)
            )
    return found


def test_torch_locks_pin_torchvision_or_record_why_not() -> None:
    adapters = _torch_locks()
    # The scan sees real torch pins, not an empty tree.
    assert adapters
    missing = sorted(
        name for name, pinned in adapters.items() if not pinned and name not in EXEMPT
    )
    assert not missing, (
        "a torch pin over a base that ships torchvision leaves the base's "
        f"mismatched torchvision; pin torchvision in these locks or exempt them: {missing}"
    )


def test_every_exemption_is_still_needed() -> None:
    adapters = _torch_locks()
    stale = sorted(name for name in EXEMPT if adapters.get(name) is not False)
    assert not stale, f"these exemptions no longer apply; remove them: {stale}"
