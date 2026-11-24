"""Draft-side byte cuts for Inkling2 MTP (DSpark) in SGLang, as an import-time
overlay. Activate with install(); configure with INKLING_DRAFT_BYTES (see
_opts.py). Unset or empty -> install() returns None and touches nothing.

Every option changes only what the draft proposes. The target forward, the
verify, and the accept rule are untouched, and the step logits the sampled
accept reads are always the ones the proposal was drawn from.
"""

from __future__ import annotations

from typing import Optional

from ._opts import Options, OptionError, parse

__all__ = ["install", "Options", "OptionError", "parse"]

_installed: Optional[Options] = None


def install(text: Optional[str] = None) -> Optional[Options]:
    """Arm the patches for the options in ``text`` (default: the env).
    Idempotent. Raises OptionError on a bad option string."""
    global _installed
    opts = parse(text)
    if not opts.active:
        return None
    if _installed is not None:
        if _installed != opts:
            raise OptionError(
                f"already installed with {_installed.describe()}, asked for {opts.describe()}"
            )
        return _installed
    from . import patches

    patches.arm(opts)
    _installed = opts
    patches.say(
        f"armed: options={opts.describe()}; patches apply when "
        f"{patches.MTP_MODULE} is imported; weights prepare at draft attach"
    )
    return opts
