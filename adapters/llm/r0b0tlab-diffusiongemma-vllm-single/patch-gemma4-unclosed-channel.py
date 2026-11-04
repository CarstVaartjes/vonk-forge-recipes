#!/usr/bin/env python3
"""Return an unclosed Gemma4 thinking channel as answer text when thinking is off.

The creator's image ships a Gemma4 reasoning parser whose non-streaming
``_strip_disabled_thinking_channel(text, wait_for_end=False)`` returns
``prefix or None`` when ``<|channel>`` opens and ``<channel|>`` never closes, so
the sample ``<|channel>thought\\n391`` becomes ``content: null``. DiffusionGemma
ignores request temperature and seed, so such samples occur at random. This hunk
returns ``prefix + rest`` there, as the creator's earlier
``gemma4-streaming-reasoning.patch`` did, and leaves the streaming branch alone.
"""

import sys
from pathlib import Path

OLD = """        if wait_for_end and not prefix:
            return None
        return prefix or None
"""
NEW = """        if wait_for_end and not prefix:
            return None
        if not wait_for_end:
            return prefix + _strip_thought_label(rest)
        return prefix or None
"""


def patch_text(source: str) -> str:
    if source.count(OLD) != 1:
        raise ValueError("Gemma4 reasoning parser changed; review the patch")
    return source.replace(OLD, NEW, 1)


if __name__ == "__main__":
    target = Path(sys.argv[1])
    target.write_text(patch_text(target.read_text()))
