#!/usr/bin/env python3
"""Cap newly admitted Inkling prefills per scheduler tick, idempotently."""

from __future__ import annotations

import argparse
from pathlib import Path


IMPORT_ANCHOR = "import itertools\nimport time\n"
IMPORT_PATCH = "import itertools\nimport os\nimport time\n"

LOOP_ANCHOR = """            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
"""

LOOP_PATCH = """            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                # Inkling's SM121 relative-attention kernel hangs on a uniform
                # batch of four new prefills. Limit only new admissions in this
                # step; already-running decode requests and max_num_seqs are
                # unchanged, so six concurrent decode streams remain possible.
                prefill_cap = int(os.environ.get(
                    \"INKLING_MAX_NEW_PREFILLS_PER_STEP\", \"0\"
                ))
                if prefill_cap > 0 and len(scheduled_new_reqs) >= prefill_cap:
                    break

                # Paused streaming sessions (WAITING_FOR_STREAMING_REQ) are not
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("target", type=Path)
    args = parser.parse_args()

    text = args.target.read_text(encoding="utf-8")
    patched = "INKLING_MAX_NEW_PREFILLS_PER_STEP" in text
    if args.check:
        if not patched:
            raise SystemExit("Inkling prefill admission cap is not installed")
        return
    if patched:
        return
    if IMPORT_ANCHOR not in text or LOOP_ANCHOR not in text:
        raise SystemExit("Unsupported scheduler.py: expected anchors not found")
    text = text.replace(IMPORT_ANCHOR, IMPORT_PATCH, 1)
    text = text.replace(LOOP_ANCHOR, LOOP_PATCH, 1)
    args.target.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
