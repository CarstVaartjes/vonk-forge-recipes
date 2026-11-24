"""Packed-only Inkling global KV overlay."""

import os
import sys


def enabled(name):
    return os.environ.get(name, "").strip().lower() not in ("", "0", "off", "none", "false", "no")


class IncompatibleImage(BaseException):
    """Startup mismatch that must propagate through the model registry."""


# Upper bound for the overflow and fallback slot counts: slot ids live in the
# 23-bit slot field of a page descriptor, and a count this large already means
# tens of GB per lane.
MAX_SIDE_SLOTS = 1 << 20


def side_count(name, default, minimum):
    """A validated side-slot count from the environment (startup refusal otherwise)."""
    raw = os.environ.get(name, default)
    try:
        count = int(str(raw).strip())
    except ValueError:
        count = None
    if count is None or not minimum <= count <= MAX_SIDE_SLOTS:
        raise IncompatibleImage(f"inkling_kv12s: {name}={raw!r} must be an integer in [{minimum}, {MAX_SIDE_SLOTS}]")
    return count


_installed = False


def install():
    global _installed
    if not enabled("INKLING_KV12S"):
        return
    if enabled("INKLING_KV12"):
        raise IncompatibleImage("INKLING_KV12S and INKLING_KV12 are mutually exclusive")
    if _installed:
        return
    from inkling_attn import install as install_attn, parse_levers
    from .patch import register

    levers = parse_levers(os.environ.get("INKLING_ATTN", ""))
    install_attn(tuple(dict.fromkeys((*levers, "prefill_global"))))
    register()
    _installed = True
    print(f"[inkling_kv12s pid={os.getpid()}] enabled packed-only global KV", file=sys.stderr, flush=True)
