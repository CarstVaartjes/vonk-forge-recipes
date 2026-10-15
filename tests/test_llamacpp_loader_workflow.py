"""Every adapter on the ggml-org llama.cpp image is proven by the loader workflow.

That image keeps its shared libraries beside ``/app/llama-server`` and resolves
them only from its ``/app`` working directory. Vonk runs workloads from ``/tmp``,
where the dynamic loader exits 127 (Nemotron 3.5 Lightning, sweep 2026-10-15).
The workflow builds and runs each such package from ``/tmp``, but a
``pull_request`` path filter cannot select by Dockerfile content, so a new
adapter must also be added to its paths.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "llamacpp-loader.yml"
GGML_ORG_BASE = re.compile(
    r"^FROM ghcr\.io/ggml-org/llama\.cpp@sha256:[0-9a-f]{64}$", re.MULTILINE
)
PATH_ENTRY = re.compile(r"^\s+- (adapters/[^\s]+)/\*\*$", re.MULTILINE)


def test_every_ggml_org_llamacpp_adapter_triggers_the_loader_proof() -> None:
    contexts = {
        dockerfile.parent.relative_to(ROOT).as_posix()
        for dockerfile in (ROOT / "adapters").glob("*/*/Dockerfile")
        if GGML_ORG_BASE.search(dockerfile.read_text(encoding="utf-8"))
    }
    assert contexts, "no adapter is built on the ggml-org llama.cpp image"
    triggered = set(PATH_ENTRY.findall(WORKFLOW.read_text(encoding="utf-8")))
    assert contexts <= triggered, (
        f"add to {WORKFLOW.name} paths: {sorted(contexts - triggered)}"
    )
