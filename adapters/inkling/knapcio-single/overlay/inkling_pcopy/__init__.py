"""Prompt-copy forced drafts for the DFlash2 drafter (SPEC=dflash2), as an import-time overlay (the launcher turns it on by default with SPEC=dflash2).

    INKLING_DFLASH_PCOPY=12             key length of the copy match (only 12 is supported: the measured rule); 0 / unset: off
    INKLING_DFLASH_PCOPY_MIN_CTX=8192   only requests whose prompt has at least this many tokens are copied for (a short
                                        prompt gives few true copies and the forced drafts can cost acceptance)

Each decode step, a matcher finds the most recent earlier occurrence of the last 12 committed tokens in the request's
own history (prompt and output); when the match extends to at least 16 tokens, the tokens that followed it are forced
into the draft block (draft proposals only: the target verifies every one). Sampled rows give each forced slot a point
mass, which the exact accept law needs; the verify cap keeps a forced run's rows live. The draft window cannot see far
context; the copy can.

Refused (inert, logged): PD disaggregation, DP attention, mixed chunked prefill, the megakernel sampled accept,
simulated acceptance, the verify-cap steplog, the stock ragged verify planner, inkling_rsfix's selector / reject parts
not applied. Bypassed per batch (history written, nothing forced): grammar, custom logit processors, min_p. Streaming
sessions are never forced. Pinned to the image's source (exit 78 on drift; DFLASH_PCOPY=0 runs without it).
"""

import os
import sys

from . import _hook, _sources

ENV = "INKLING_DFLASH_PCOPY"
ENV_MIN_CTX = "INKLING_DFLASH_PCOPY_MIN_CTX"
DEFAULT_MIN_CTX = 8192
OFF_HINT = " (set DFLASH_PCOPY=0 to run without the prompt copy)"
STATE = {"installed": False, "key": 0}


def key_from_env(env=None):
    env = os.environ if env is None else env
    raw = env.get(ENV, "").strip()
    if raw in ("", "0"):
        return 0
    if raw != "12":
        raise RuntimeError(f"inkling_pcopy: {ENV}={raw!r}: only 12 (or 0 = off) is supported")
    return 12


def min_ctx_from_env(env=None):
    """Prompts shorter than this many tokens are never copied from (default 8192)."""
    env = os.environ if env is None else env
    raw = env.get(ENV_MIN_CTX, "").strip()
    if raw == "":
        return DEFAULT_MIN_CTX
    if not raw.isdigit():
        raise RuntimeError(f"inkling_pcopy: {ENV_MIN_CTX}={raw!r} is not a whole number of tokens")
    return int(raw)


def install(env=None, root=None):
    if STATE["installed"]:
        return STATE["key"]
    from . import patch

    key = key_from_env(env)
    min_ctx = min_ctx_from_env(env) if key else DEFAULT_MIN_CTX   # the threshold matters (and is checked) only when on
    if key:
        _sources.check(root, OFF_HINT)
        loaded = [m for m in patch.MODULES if m in sys.modules]
        if loaded:
            raise RuntimeError(f"inkling_pcopy: install before {loaded} are imported" + OFF_HINT)
    STATE.update(installed=True, key=key)
    patch.STATE["key"] = key
    patch.STATE["min_ctx"] = min_ctx
    if key:
        _hook.when_imported(patch.CONFIG_MODULE, lambda mod: patch.patch_config(mod, OFF_HINT))
        _hook.when_imported(patch.WORKER_MODULE, lambda mod: patch.patch_worker(mod, OFF_HINT))
        patch._log(f"armed: prompt-copy drafts (key {key}, forced at a 16-token match, prompts >= {min_ctx} tokens)")
    return key
