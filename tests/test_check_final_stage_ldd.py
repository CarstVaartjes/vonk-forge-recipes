"""An added runtime image that copies a binary needs the ldd check; untouched ones pass."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

from added_diff import added_diff
from repo_root import ROOT

LOADER = importlib.machinery.SourceFileLoader(
    "check_final_stage_ldd", str(ROOT / "tools/check-final-stage-ldd")
)
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
assert SPEC is not None
guard = importlib.util.module_from_spec(SPEC)
sys.modules[LOADER.name] = guard
LOADER.exec_module(guard)

NAME = "adapters/x/y/Dockerfile"
DOCKERFILE = (
    "FROM nvidia/cuda:13.0.0-devel-ubuntu24.04@sha256:a AS build\n"
    "RUN make\n"
    "FROM docker.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04@sha256:b\n"
    "COPY --from=build --chmod=0755 /src/server /opt/vonk/bin/server\n"
)
CHECKED = DOCKERFILE + 'RUN ! ldd /opt/vonk/bin/server | grep "not found"\n'


def _violations(root: Path, text: str, old_lines: int = 0) -> list[str]:
    (root / NAME).parent.mkdir(parents=True)
    (root / NAME).write_text(text)
    return guard.violations(added_diff({NAME: text}, old_lines={NAME: old_lines}), root)


def test_an_added_image_without_the_step_fails_and_with_it_passes(
    tmp_path: Path,
) -> None:
    assert len(_violations(tmp_path / "a", DOCKERFILE)) == 1
    assert _violations(tmp_path / "b", CHECKED) == []
    assert guard.scan(DOCKERFILE, {1, 2}) is True  # a build-stage change counts too
    assert guard.scan(DOCKERFILE, set()) is False  # untouched


def test_the_step_must_be_the_documented_form() -> None:
    continued = (
        DOCKERFILE + 'RUN ! ldd /opt/vonk/bin/server \\\n    | grep "not found"\n'
    )
    forced = DOCKERFILE + 'RUN ldd /opt/vonk/bin/server | grep "not found" || true\n'
    heredoc = DOCKERFILE.replace("RUN make", "RUN cat <<E\nfrom x\nE")
    platform = DOCKERFILE.replace(
        "FROM docker.io/nvidia/cuda:13.0.0-runtime",
        "FROM --platform=linux/arm64 docker.io/nvidia/cuda:13.0.0-runtime",
    )
    assert guard.scan(continued, {1}) is False
    assert guard.scan(forced, {1}) is True
    assert guard.scan(heredoc, {1}) is True  # lowercase "from" is not a stage
    assert guard.scan(platform, {1}) is True  # --platform still reads the base
