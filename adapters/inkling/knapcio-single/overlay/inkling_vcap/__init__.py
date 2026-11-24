"""Inkling-2 DSpark verify cap overlay (dead-row expert remap + accept cap).

Activated from sitecustomize via ``activate()``; a no-op unless INKLING_VERIFY_CAP
selects a mode.
"""

from __future__ import annotations

__all__ = ["activate"]


def activate() -> bool:
    """Parse the environment; when a mode is set, install the import hooks.
    Returns whether the overlay is armed. Raises on a malformed setting."""
    from inkling_vcap.config import parse_config

    cfg = parse_config()
    if cfg is None:
        return False
    from inkling_vcap import importhook, patches
    from inkling_vcap.state import STATE

    if STATE.config is not None:
        return True
    STATE.config = cfg
    importhook.install(patches.HOOKS)
    return True
