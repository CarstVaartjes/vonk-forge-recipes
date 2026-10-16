"""Backport vLLM PR #47953 (fixes issue #47794) onto the pinned v0.25.1 image.

vLLM #43957 (merged 2026-07-06, shipped in v0.25.1) stopped sharing the target's
embedding with a draft whose embedding width differs. Gemma 4 MTP assistants are
such drafts: their pre_projection takes 2 * backbone_hidden_size, so without
sharing the MoE 26B-A4B (V1 proposer path) fails engine init with
"a and b must have same reduction dim, but got [s47, 3840] X [5632, 1024]".
MiaAI-Lab's recipe ran the 2026-07-05 nightly, before the regression. #47953
(first released in v0.27.0) limits the width check to EAGLE drafts; this applies
its exact hunk and fails the build when the v0.25.1 text is not found.
"""

from __future__ import annotations

import importlib.util
import py_compile
from pathlib import Path

OLD = """            if share_embeddings:
                draft_embed = self.model.model.embed_tokens
                # Only share when both models use the same embedding width.
                # Guard with isinstance"""
NEW = """            if share_embeddings and hasattr(self.model, "has_own_embed_tokens"):
                # EAGLE drafts consume input embeddings at their own hidden
                # size, so only share when the widths match. MTP drafts
                # project target-width embeddings (e.g. Gemma4 MTP's
                # pre_projection takes 2 * backbone_hidden_size), so the
                # width check does not apply to them.
                draft_embed = self.model.model.embed_tokens
                # Guard with isinstance"""


def main() -> None:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("vLLM is missing from the base image")
    target = (
        Path(next(iter(spec.submodule_search_locations)))
        / "v1"
        / "spec_decode"
        / "llm_base_proposer.py"
    )
    source = target.read_text(encoding="utf-8")
    if source.count(OLD) != 1:
        raise SystemExit(f"vLLM PR #47953 anchor not found exactly once in {target}")
    target.write_text(source.replace(OLD, NEW, 1), encoding="utf-8")
    py_compile.compile(str(target), doraise=True)
    print(f"applied vLLM PR #47953 to {target}")


if __name__ == "__main__":
    main()
